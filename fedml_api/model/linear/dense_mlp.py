import torch
import numpy as np
from pdb import set_trace as st

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class PurchaseMLP(nn.Module):

    layer_names = ["fc1", "fc5"]
    avgmode_to_layers = {
        "all": ["fc1.weight", "fc1.bias", "fc5.weight", "fc5.bias",],
        "top": ["fc5.weight", "fc5.bias"],
        "bottom": ["fc1.weight", "fc1.bias", ],
        "none": [],
    }
    
    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(256, n_classes)
        self.drop = nn.Dropout(p=0.5)
        self.log_penultimate_grad = False
        self.penultimate_dim = 256
        
    def open_penultimate_log(self):
        self.log_penultimate_grad = True
        
    def close_penultimate_log(self):
        self.log_penultimate_grad = False
        self.penultimate = None

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        # return F.log_softmax(self.fc5(x), dim=1)
        if self.log_penultimate_grad:
            self.penultimate = x
            self.penultimate.retain_grad()
        return self.fc5(x)

class TexasMLP(nn.Module):
    layer_names = ["fc1", "fc2", "fc3"]
    avgmode_to_layers = {
        "bottom": ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",],
        "top": ["fc3.weight", "fc3.bias"],
        "all": ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"],
        "none": [],
    }
    
    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_classes)
        self.drop = nn.Dropout(p=0.5)
        self.log_penultimate_grad = False
        self.penultimate_dim = 512
        
    def open_penultimate_log(self):
        self.log_penultimate_grad = True
        
    def close_penultimate_log(self):
        self.log_penultimate_grad = False
        self.penultimate = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        # return F.log_softmax(self.fc3(x), dim=1)
        if self.log_penultimate_grad:
            self.penultimate = x
            self.penultimate.retain_grad()
        return self.fc3(x)
