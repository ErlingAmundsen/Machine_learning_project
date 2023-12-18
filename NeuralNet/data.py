# create class data that holds fashion mnist data
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, dataset="train"):
        train, test = np.load("./Data/fashion_train.npy"), np.load("./Data/fashion_test.npy")

        # split data
        X_data, y_data = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]

        # split training data into training and dev set
        X_train, X_dev, y_train, y_dev = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

        # normalize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.transform(X_dev)
        scaler2 = StandardScaler()
        X_data = scaler2.fit_transform(X_data)
        X_test = scaler2.transform(X_test)

        # PCA
        run_pca = True
        if run_pca:
            pca = PCA()
            X_train = pca.fit_transform(X_train)
            X_dev = pca.transform(X_dev)
            pca2 = PCA()
            X_data = pca2.fit_transform(X_data)
            X_test = pca2.transform(X_test)
        
        if dataset == "train":
            images, targets = X_train, y_train
        elif dataset == "val":
            images, targets = X_dev, y_dev
        elif dataset == "test":
            images, targets = X_test, y_test
        elif dataset == "total":
            images, targets = X_data, y_data
        else:
            raise ValueError("Dataset must be 'train', 'val', 'test' or 'total'")

        # convert to tensors
        self.images = torch.from_numpy(images).float()
        self.targets = torch.from_numpy(targets).long()
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]