import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn #API for building neural networks
from torch.utils.data import Dataset, DataLoader #Imports the Dataset and Dataloader classes
import os
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file, header=None, 
                               names=['Path', 'Label'], delimiter=',')
        
    def __len__(self):
        return(len(self.annotations))
    
    def __getitem__(self, index):
        audio = torchaudio.load(self.annotations['Path'][index])
        label = self.annotations['Label'][index]
        return(audio[0][0], label)


if __name__ == "__main__":
    import time

    start_time = time.time()

    my_dataset = AudioDataset("train_audioMNIST.csv")
    for i in range(len(my_dataset)):
        sample = my_dataset[i]
        print(i, sample[0].shape, sample[1])

    execution_time = time.time() - start_time

    print(f"Le bloc de code a pris {execution_time} secondes pour s'exécuter.")