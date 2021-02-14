import torch
import numpy as np

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset



class PurchaseMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(256, n_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return F.log_softmax(self.fc5(x), dim=1)

class TexasMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return F.log_softmax(self.fc3(x), dim=1)
