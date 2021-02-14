import torch
from torch import nn

def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):

    if isinstance(module, torch.nn.modules.conv._ConvNd):
        # TO-DO: check litterature
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)

class HAR_CNN(nn.Module):
    def __init__(self, data_size, n_classes):
        """
        """
        super(HAR_CNN, self).__init__()
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
        a = torch.relu(self.lin3(a))
        a = self.drop(a)
        a = self.lin4(a)
        a = self.softmax(a)

        return a
    
    def reset_parameters(self):
        # for name, module in self.named_modules():
        #     if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        #         module.reset_parameters()
        """ Initializes the weights for each layer of the CNN"""
        self.apply(weights_init)