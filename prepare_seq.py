import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IMUSequence:
    def __init__(self, imu_file, vi_file, sequence_length, stride):
        imu_data = pd.read_csv(imu_file).iloc[1:, 1:].values
        vi_data = pd.read_csv(vi_file).iloc[1:, 1:].values  # change

        self.sequences = []
        self.targets = []

        for i in range(0, len(imu_data) - sequence_length, stride):
            self.sequences.append(imu_data[i:i+sequence_length])
            self.targets.append(vi_data[i+sequence_length])

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
       
       

class IMUDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = []
        self.targets = []
        for seq in sequences:
            self.sequences.extend(seq.sequences)
            self.targets.extend(seq.targets)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        self.device = get_device()
        
        # print(self.sequences.shape)
        # print(self.sequences[0].dtype)
        # print(self.sequences[0][0].dtype)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]).to(self.device), 
                torch.FloatTensor(self.targets[idx]).to(self.device))
        
        
def load_sequences(root_dir, sequence_length, stride):
    sequences = []
    for data_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, data_folder)
        if os.path.isdir(folder_path):
            
            sequences.append(IMUSequence(
                os.path.join(folder_path, 'mag.csv'),
                os.path.join(folder_path, 'gt.csv'),
                sequence_length,
                stride
            ))
    return sequences


def prepare_data(root_dir, sequence_length, batch_size, stride):
    
    all_sequences = load_sequences(root_dir, sequence_length, stride)
    train_sequences, val_sequences = train_test_split(all_sequences, test_size=0.1, random_state=42)

    train_dataset = IMUDataset(train_sequences)
    val_dataset = IMUDataset(val_sequences)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, train_dataset, val_dataset


