import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=25)
        self.fc3 = nn.Linear(in_features=25, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=5)
    
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.out(X)
        return X