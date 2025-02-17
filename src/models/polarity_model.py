import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PolarityClassifier(nn.Module):
    def __init__(self, input_size):
        super(PolarityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32) 
        self.dropout = nn.Dropout(0.5)
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return self.sigmoid(x)