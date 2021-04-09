
import torch
from torch import nn
from pdb import set_trace as st

def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("{}".format(model_type))

class Cnn1(nn.Module):
    def __init__(self, data_size, n_classes):
        """
        """
        super(Cnn1, self).__init__()
        self.n_chan = data_size[0]
        self.n_classes = n_classes
    
        # Convolutional Layers
        self.conv1 = nn.Conv1d(self.n_chan, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1)
        self.drop = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)

        # Fully connected layers
        self.lin3 = nn.Linear(1984, 100)
        self.lin4 = nn.Linear(100, self.n_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        a = torch.relu(self.conv1(x))
        a = torch.relu(self.conv2(a))
        a = self.drop(a)
        a = self.pool(a)
        #Fully connected layers
        a = a.view((batch_size, -1))
        a = self.lin3(a)
        a = self.drop(a)
        a = self.lin4(a)
        a = self.softmax(a)

        return a

