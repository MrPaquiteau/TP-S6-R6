import torch
from torch.utils.data import Dataset, DataLoader #Imports the Dataset and Dataloader classes
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file, header=None, 
                               names=['Path', 'Label'], delimiter=',')
        
    def __len__(self):
        return(len(self.annotations))
    
    def __getitem__(self, index):
        label = self.annotations['Label'][index]
        audio = torch.zeros((1,48000))
        data = torchaudio.load(self.annotations['Path'][index])
        audio[:, :data[0].size()[1]] = data[0][0]
        melspectrogramme = torchaudio.transforms.MelSpectrogram()(audio)
        return(melspectrogramme, label)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device)) #Use GPU if available

    audioMNIST_100 = AudioDataset('./100_fichiers.csv')
    data_loader = DataLoader(audioMNIST_100, batch_size=10, shuffle=True, num_workers=1)
    for data in data_loader:
        continue
