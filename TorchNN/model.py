import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same")
        self.fc1 = nn.Linear(in_features=16 * 3 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=5)

        # Max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout2d = nn.Dropout2d(p=dropout)
        self.dropout = nn.Dropout(p=dropout)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(num_features=16)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.batch_norm3 = nn.BatchNorm2d(num_features=16)

        self.batch_norm4 = nn.BatchNorm1d(num_features=128)
        self.batch_norm5 = nn.BatchNorm1d(num_features=64)
    
    def forward(self, X):
        X = self.dropout2d(self.max_pool(F.relu(self.batch_norm1(self.conv1(X)))))
        X = self.dropout2d(self.max_pool(F.relu(self.batch_norm2(self.conv2(X)))))
        X = self.dropout2d(self.max_pool(F.relu(self.batch_norm3(self.conv3(X)))))
        X = X.view(-1, 16 * 3 * 3)
        X = self.dropout(F.relu(self.batch_norm4(self.fc1(X))))
        X = self.dropout(F.relu(self.batch_norm5(self.fc2(X))))
        X = self.out(X)
        return X