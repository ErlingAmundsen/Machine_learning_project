# create class data that holds fashion mnist data
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, test=False, val=False):
        data = np.load(file = f"./Data/fashion_{'test' if test else 'train'}.npy")

        # split to x and y
        images, targets = data[:, :-1], data[:, -1]
        
        # Normalize data from 0-255 to 0-1
        images = images / 255.0

        if not test:
            # split train to train and validation
            images, val_x, targets, val_y = train_test_split(images, targets, test_size=0.2, random_state=42)

            if val:
                images = val_x
                targets = val_y
        
        # convert to tensors
        self.images = torch.from_numpy(images).reshape(-1, 1, 28, 28).float()
        self.targets = torch.from_numpy(targets).long()
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]