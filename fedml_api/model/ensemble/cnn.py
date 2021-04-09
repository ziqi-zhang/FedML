import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
import numpy as np

import copy
from pdb import set_trace as st

def simple_conv_str(conv_str):
    elements = conv_str.split(',')
    return f"{elements[0]},{elements[1]})"

class AdaptiveCNN(torch.nn.Module):


    blocks = ["conv2d_1_block", "conv2d_2_block", "linear_1_block", "linear_2_block"]
    feature_layers = ["conv2d_1", "conv2d_2", "linear_1"]
    
    # blocks = [["conv2d_1", "conv2d_2"], ["linear_1", "linear_2"]]
    # feature_layers = ["conv2d_2"]
    
    # blocks = [["conv2d_1_block", "conv2d_2_block", "linear_1_block", "linear_2_block"]]
    # feature_layers = []

    def __init__(self, only_digits=True, input_dim=1):
        super(AdaptiveCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv2d_1_block = nn.Sequential(*[
            torch.nn.Conv2d(input_dim, 32, kernel_size=3),
            nn.ReLU(),
        ])
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2_block = nn.Sequential(*[
            torch.nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        ])
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_1_block = nn.Sequential(*[
            self.dropout_1,
            nn.Linear(9216, 128),
            nn.ReLU(),
        ])
        
        if isinstance(only_digits, bool):
            linear_2 = nn.Linear(128, 10 if only_digits else 62)
        elif isinstance(only_digits, int):
            # For EMNIST dataset, have 47 classes
            linear_2 = nn.Linear(128, only_digits)
        self.linear_2_block = nn.Sequential(
            linear_2
        )
        self.softmax = nn.Softmax(dim=1)
        
        self.weight_reinit()
        
        self.hetero_block_fn = build_hetero_blocks
        self.hetero_block_kwargs = {
            "only_digits": only_digits,
            "input_dim": input_dim,
        }
        self.hetero_arch_fn = partial(build_hetero_archs, self)


    def forward(self, x):
        x = self.layer_conv2d_1(x)
        x = self.layer_conv2d_2(x)
        x = self.layer_linear_1(x)
        x = self.layer_linear_2(x)
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
        x = self.conv2d_1_block(x)
        return x
    
    def layer_conv2d_2(self, x):
        x = self.conv2d_2_block(x)
        x = self.max_pooling(x)
        return x
    
    def layer_linear_1(self, x):
        x = self.flatten(x)
        x = self.linear_1_block(x)
        return x
        
    def layer_linear_2(self, x):
        x = self.dropout_2(x)
        x = self.linear_2_block(x)
        return x
    
    layer_to_forward_fn = {
        "conv2d_1_block": layer_conv2d_1, 
        "conv2d_2_block": layer_conv2d_2, 
        "linear_1_block": layer_linear_1,
        "linear_2_block": layer_linear_2
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
                
    def deepen_conv(self, conv_block):
        new_conv_block = copy.deepcopy(conv_block)
        depth = len(new_conv_block)
        channel = new_conv_block[-2].out_channels
        new_conv_block.add_module(f"{depth}", torch.nn.Conv2d(channel, channel, kernel_size=3, padding=1))
        new_conv_block.add_module(f"{depth+1}", nn.ReLU())
        return new_conv_block
    
    def deepen_conv1(self):
        self.conv2d_1_block = self.deepen_conv(self.conv2d_1_block)
        
    def deepen_conv2(self):
        self.conv2d_2_block = self.deepen_conv(self.conv2d_2_block)
        
        
    def adjust_last_conv_width(self, conv_block, width):
        assert len(conv_block) > 2
        new_conv_block = copy.deepcopy(conv_block)
        in_channels = new_conv_block[-4].in_channels
        out_channels = new_conv_block[-2].out_channels
        kernel, stride, padding = new_conv_block[-4].kernel_size, new_conv_block[-4].stride, new_conv_block[-4].padding
        new_conv_block[-4] = torch.nn.Conv2d(
            in_channels, width, 
            kernel_size=kernel, stride=stride, padding=padding
        )
        new_conv_block[-2] = torch.nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
        return new_conv_block
    
    def widen_last_conv(self, conv_block):
        assert len(conv_block) > 2
        new_conv_block = copy.deepcopy(conv_block)
        channels = new_conv_block[-4].out_channels
        new_channels = channels + 16
        return self.adjust_last_conv_width(conv_block, new_channels)
    
    def shrink_last_conv(self, conv_block):
        assert len(conv_block) > 2
        new_conv_block = copy.deepcopy(conv_block)
        channels = new_conv_block[-4].out_channels
        new_channels = channels - 16
        return self.adjust_last_conv_width(conv_block, new_channels)
        
    def widen_conv1(self):
        self.conv2d_1_block = self.widen_last_conv(self.conv2d_1_block)
        
    def shrink_conv1(self):
        self.conv2d_1_block = self.shrink_last_conv(self.conv2d_1_block)
        
    def widen_conv2(self):
        self.conv2d_2_block = self.widen_last_conv(self.conv2d_2_block)
        
    def shrink_conv2(self):
        self.conv2d_2_block = self.shrink_last_conv(self.conv2d_2_block)
        
    def deepen_linear(self, linear_block):
        new_linear_block = copy.deepcopy(linear_block)
        out_channel = new_linear_block[-2].out_features
        in_channel = new_linear_block[-2].in_features
        depth = len(new_linear_block)
        new_linear_block[-2] = nn.Linear(in_channel, out_channel*4)
        new_linear_block.add_module(f"{depth}", nn.Dropout(0.25))
        new_linear_block.add_module(f"{depth+1}", nn.Linear(out_channel*4, out_channel))
        new_linear_block.add_module(f"{depth+2}", nn.ReLU())
        return new_linear_block

    
    def deepen_linear1(self):
        self.linear_1_block = self.deepen_linear(self.linear_1_block)
        
    
        
    def flatten_structure(self):
        output = f""
        conv1_str = f""
        for module in self.conv2d_1_block:
            if isinstance(module, nn.Conv2d):
                conv_str = f"{module}"
                conv1_str += f"{simple_conv_str(conv_str)}-"
        conv1_str = f"[{conv1_str[:-1]}]"
        
        conv2_str = f""
        for module in self.conv2d_2_block:
            if isinstance(module, nn.Conv2d):
                conv_str = f"{module}"
                conv2_str += f"{simple_conv_str(conv_str)}-"
        conv2_str = f"[{conv2_str[:-1]}]"
        
        linear1_str = f""
        for module in self.linear_1_block:
            if isinstance(module, nn.Linear):
                linear_str = f"{module}"
                linear1_str += f"{simple_conv_str(linear_str)}-"
        linear1_str = f"[{linear1_str[:-1]}]"
        
        linear2_str = f""
        for module in self.linear_2_block:
            if isinstance(module, nn.Linear):
                linear_str = f"{module}"
                linear2_str += f"{simple_conv_str(linear_str)}-"
        linear2_str = f"[{linear2_str[:-1]}]"
        
        output = f"{conv1_str}----{conv2_str}----{linear1_str}----{linear2_str}"
        return output
        
def build_large_cnn(*args, **kwargs):
    model = AdaptiveCNN(*args, **kwargs)
    model.deepen_conv1()
    model.widen_conv1()
    model.deepen_conv1()
    model.widen_conv1()
    model.deepen_conv1()
    model.deepen_conv2()
    model.widen_conv2()
    model.deepen_conv2()
    model.widen_conv2()
    model.deepen_conv2()
    # model.shrink_conv1()
    # model.shrink_conv2()
    model.widen_conv1()
    model.widen_conv2()
    model.deepen_linear1()
    
    return model

def build_hetero_archs(mother_model, num_branch):
    block_fn, block_kwargs = mother_model.hetero_block_fn, mother_model.hetero_block_kwargs
    block_archs = block_fn(**block_kwargs)
    
    """
        Set branch selection info
        branch_selection: 
        Dict(
            block_name (int): List(
                                block_idx_in_each_branch
                            )
        )
        """
    branch_selection = {}
    for block_name in block_archs.keys():
        branch_selection[block_name] = []
        
        repeat_times = int(num_branch / len(block_archs[block_name]))
        blocks = repeat_times * block_archs[block_name]
        # print(len(block_archs[block_name]))
        if len(block_archs[block_name]) > 1:
            quot_blocks = np.random.choice(
                block_archs[block_name], size=int(num_branch % len(block_archs[block_name])), replace=False
            ).tolist()
            blocks += quot_blocks
        np.random.shuffle(blocks)
        branch_selection[block_name] = blocks
        
    block_names = list(block_archs.keys())
    archs = []
    for branch_idx in range(num_branch):
        model = copy.deepcopy(mother_model)
        for block_name in block_names:
            model._modules[block_name] = branch_selection[block_name][branch_idx]
        model.weight_reinit()
        archs.append(model)
        # print(f"{branch_idx}: {model.flatten_structure()}")
    return archs
    

def build_hetero_blocks(*args, **kwargs):
    conv1_archs = collect_conv1_archs(*args, **kwargs)
    conv2_archs = collect_conv2_archs(*args, **kwargs)
    linear1_archs = collect_linear1_archs(*args, **kwargs)
    linear2_archs = collect_linear2_archs(*args, **kwargs)
    
    archs = {
        "conv2d_1_block": conv1_archs,
        "conv2d_2_block": conv2_archs,
        "linear_1_block": linear1_archs,
        "linear_2_block": linear2_archs,
    }
    return archs

def collect_conv1_archs(*args, **kwargs):
    archs = []
    model = AdaptiveCNN(*args, **kwargs)
    archs.append(copy.deepcopy(model.conv2d_1_block))
    model.deepen_conv1()
    archs.append(copy.deepcopy(model.conv2d_1_block))
    model.shrink_conv1()
    thin_model = copy.deepcopy(model)
    archs.append(copy.deepcopy(model.conv2d_1_block))
    model.widen_conv1()
    model.widen_conv1()
    wide_model = copy.deepcopy(model)
    archs.append(copy.deepcopy(model.conv2d_1_block))
    model.shrink_conv1()
    model.deepen_conv1()
    archs.append(copy.deepcopy(model.conv2d_1_block))
    
    thin_model.deepen_conv1()
    thin_model.shrink_conv1()
    archs.append(copy.deepcopy(thin_model.conv2d_1_block))
    wide_model.deepen_conv1()
    wide_model.widen_conv1()
    archs.append(copy.deepcopy(wide_model.conv2d_1_block))
    
    model.deepen_conv1()
    archs.append(copy.deepcopy(model.conv2d_1_block))
    thin_model.deepen_conv1()
    thin_model.shrink_conv1()
    archs.append(copy.deepcopy(thin_model.conv2d_1_block))
    wide_model.deepen_conv1()
    wide_model.widen_conv1()
    archs.append(copy.deepcopy(wide_model.conv2d_1_block))
    
    return archs

def collect_conv2_archs(*args, **kwargs):
    archs = []
    model = AdaptiveCNN(*args, **kwargs)
    archs.append(copy.deepcopy(model.conv2d_2_block))
    model.deepen_conv2()
    archs.append(copy.deepcopy(model.conv2d_2_block))
    model.shrink_conv2()
    model.shrink_conv2()
    thin_model = copy.deepcopy(model)
    archs.append(copy.deepcopy(model.conv2d_2_block))
    model.widen_conv2()
    model.widen_conv2()
    model.widen_conv2()
    model.widen_conv2()
    wide_model = copy.deepcopy(model)
    archs.append(copy.deepcopy(model.conv2d_2_block))
    model.shrink_conv2()
    model.shrink_conv2()
    model.deepen_conv2()
    archs.append(copy.deepcopy(model.conv2d_2_block))
    
    thin_model.deepen_conv2()
    thin_model.shrink_conv2()
    thin_model.shrink_conv2()
    archs.append(copy.deepcopy(thin_model.conv2d_2_block))
    wide_model.deepen_conv2()
    wide_model.widen_conv2()
    wide_model.widen_conv2()
    archs.append(copy.deepcopy(wide_model.conv2d_2_block))
    
    model.deepen_conv2()
    archs.append(copy.deepcopy(model.conv2d_2_block))
    thin_model.deepen_conv2()
    thin_model.shrink_conv2()
    thin_model.shrink_conv2()
    archs.append(copy.deepcopy(thin_model.conv2d_2_block))
    wide_model.deepen_conv2()
    wide_model.widen_conv2()
    wide_model.widen_conv2()
    archs.append(copy.deepcopy(wide_model.conv2d_2_block))
    
    return archs
    
def collect_linear1_archs(*args, **kwargs):
    archs = []
    model = AdaptiveCNN(*args, **kwargs)
    archs.append(copy.deepcopy(model.linear_1_block))
    model.deepen_linear1()
    archs.append(copy.deepcopy(model.linear_1_block))
    model.deepen_linear1()
    archs.append(copy.deepcopy(model.linear_1_block))
    
    return archs

def collect_linear2_archs(*args, **kwargs):
    archs = []
    model = AdaptiveCNN(*args, **kwargs)
    archs.append(copy.deepcopy(model.linear_2_block))

    return archs
    
    
if __name__ == "__main__":
    # model = AdaptiveCNN()
    # model.deepen_conv1()
    # model.deepen_conv2()
    # model.shrink_conv1()
    # model.shrink_conv2()
    # model.deepen_linear1()
    # print(model.flatten_structure())
    # print(model)
    # names = model.state_dict().keys()
    # print(names)
    
    # archs = collect_linear1_archs()
    # st()
    
    model = AdaptiveCNN()
    model.hetero_arch_fn(32)
