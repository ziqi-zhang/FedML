import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pdb import set_trace as st

class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """
    
    layer_names = ["conv2d_1", "conv2d_2", "linear_1", "linear_2"]
    avgmode_to_layers = {
        "bottom": ['conv2d_1.weight', 'conv2d_1.bias', 'conv2d_2.weight', 'conv2d_2.bias'],
        "top": ['linear_1.weight', 'linear_1.bias', 'linear_2.weight', 'linear_2.bias'],
        "all": ['conv2d_1.weight', 'conv2d_1.bias', 'conv2d_2.weight', 'conv2d_2.bias', 'linear_1.weight', 'linear_1.bias', 'linear_2.weight', 'linear_2.bias'],
        "none": [],
    }
    
    blocks = ["conv2d_1", "conv2d_2", "linear_1", "linear_2"]
    feature_layers = ["conv2d_1", "conv2d_2", "linear_1"]
    
    # blocks = [["conv2d_1", "conv2d_2"], ["linear_1", "linear_2"]]
    # feature_layers = ["conv2d_2"]
    
    # blocks = [["conv2d_1", "conv2d_2", "linear_1", "linear_2"]]
    # feature_layers = []

    def __init__(self, only_digits=True, input_dim=1):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(input_dim, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        if isinstance(only_digits, bool):
            self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        elif isinstance(only_digits, int):
            # For EMNIST dataset, have 47 classes
            self.linear_2 = nn.Linear(128, only_digits)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.log_penultimate_grad = False
        self.penultimate_dim = 128
        
        self.weight_reinit()
        
    def open_penultimate_log(self):
        self.log_penultimate_grad = True
        
    def close_penultimate_log(self):
        self.log_penultimate_grad = False
        self.penultimate = None

    def forward(self, x):
        if x.dim() == 3:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        if self.log_penultimate_grad:
            self.penultimate = x
            self.penultimate.retain_grad()
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x

    def feature_forward(self, x):
        features = []
        x = self.layer_conv2d_1(x)
        if "conv2d_1" in self.feature_layers:
            features.append(x)
        x = self.layer_conv2d_2(x)
        if "conv2d_2" in self.feature_layers:
            features.append(x)
        x = self.layer_linear_1(x)
        if "linear_1" in self.feature_layers:
            features.append(x)
        x = self.layer_linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return features, x
    
    def layer_conv2d_1(self, x):
        if x.dim() == 3:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        return x
    
    def layer_conv2d_2(self, x):
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        return x
    
    def layer_linear_1(self, x):
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        return x
        
    def layer_linear_2(self, x):
        x = self.dropout_2(x)
        x = self.linear_2(x)
        return x
    
    layer_to_forward_fn = {
        "conv2d_1": layer_conv2d_1, 
        "conv2d_2": layer_conv2d_2, 
        "linear_1": layer_linear_1,
        "linear_2": layer_linear_2
    }
    
    def weight_reinit(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.reset_parameters()

class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout_1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(120, 84)
        self.dropout_2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout_1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)