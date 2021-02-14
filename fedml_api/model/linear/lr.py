import torch
from pdb import set_trace as st

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, flatten=False):
        super(LogisticRegression, self).__init__()
        self.flatten = flatten
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.flatten:
            x = x.flatten(1)
        outputs = torch.sigmoid(self.linear(x))
        return outputs
