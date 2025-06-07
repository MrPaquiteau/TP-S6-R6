from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import cProfile
import sys
import csv
import os


class Net(nn.Module):
    def __init__(self, use_dropout=True):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25) if use_dropout else None
        self.dropout2 = nn.Dropout2d(0.5) if use_dropout else None
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.use_dropout and self.dropout1:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.use_dropout and self.dropout2:
            x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct, len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--task-type', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Task type: 1=Basic, 2=Reversed data, 3=No dropout, 4=Reversed+No dropout')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("CUDA is available: ", use_cuda)
    use_mps = torch.backends.mps.is_available() 
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Determine data usage based on task type
    use_dropout = args.task_type in [1, 2]
    reverse_data = args.task_type in [2, 4]
    
    print("reverse_data: ", reverse_data)
    
    print(f"Running task type: {args.task_type}")
    print(f"Using dropout: {use_dropout}")
    print(f"Using reversed data: {reverse_data}")
    
    # Load datasets
    train_dataset = datasets.MNIST('data', train=not reverse_data, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    
    test_dataset = datasets.MNIST('data', train=reverse_data, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(use_dropout=use_dropout).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_losses = []
    test_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, correct, len_dataset = test(args, model, device, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()
        
    # Define filenames based on task type
    task_desc = {
        1: "basic",
        2: "reversed",
        3: "no_dropout",
        4: "reversed_no_dropout"
    }
    
    for d in ['mnist/exports', 'mnist/exports/models', 'mnist/exports/results']:
        os.makedirs(d, exist_ok=True)

    if args.save_model:
        torch.save(model.state_dict(), f"mnist/exports/models/mnist_cnn_task{args.task_type}.pt")

    with open(f'mnist/exports/results/train_losses_task{args.task_type}_{task_desc[args.task_type]}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss'])
        for epoch, loss in enumerate(train_losses, 1):
            writer.writerow([epoch, loss])

    with open(f'mnist/exports/results/test_losses_task{args.task_type}_{task_desc[args.task_type]}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'test_loss'])
        for epoch, loss in enumerate(test_losses, 1):
            writer.writerow([epoch, loss])

if __name__ == "__main__":
    for i in range(1, 5):
        print(f"Running task type {i}")
        sys.argv = ['main.py', '--batch-size', '64', '--test-batch-size', '1000', '--epochs', '14', '--lr', '1', '--gamma', '0.7', '--seed', '1', '--log-interval', '10', '--save-model', '--task-type', str(i)]
        cProfile.run('main()', sort='time')