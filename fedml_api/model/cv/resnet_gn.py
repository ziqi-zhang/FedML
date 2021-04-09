import math
import logging

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pdb import set_trace as st
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

from fedml_api.model.cv.group_normalization import GroupNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def norm2d(planes, num_channels_per_group=32):
    print("num_channels_per_group:{}".format(num_channels_per_group))
    if num_channels_per_group > 0:
        return GroupNorm2d(planes, num_channels_per_group, affine=True,
                           track_running_stats=False)
    else:
        return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 group_norm=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(planes, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes, group_norm)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 group_norm=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm2d(planes * 4, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, group_norm=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm2d(64, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       group_norm=group_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       group_norm=group_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       group_norm=group_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       group_norm=group_norm)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GroupNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)
                
        self.log_penultimate_grad = False
        self.penultimate_dim = 512 * block.expansion
        
        set_block_mode(self)
        for key, params in self.avgmode_to_layers.items():
            print(key, params)
            
        
    def open_penultimate_log(self):
        self.log_penultimate_grad = True
        
    def close_penultimate_log(self):
        self.log_penultimate_grad = False
        self.penultimate = None

    def _make_layer(self, block, planes, blocks, stride=1, group_norm=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(planes * block.expansion, group_norm),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            group_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_norm=group_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.log_penultimate_grad:
            self.penultimate = x
            self.penultimate.retain_grad()
        x = self.fc(x)

        return x
    
    blocks = [["conv1", "bn1", "layer1"], "layer2", "layer3", "layer4", "fc"]
    feature_layers = ["layer1", "layer2", "layer3", "layer4",]
    
    def feature_forward(self, x):        
        features = []
        x = self.conv1_forward(x)
        x = self.bn1_forward(x)
        x = self.layer1_forward(x)
        if "layer1" in self.feature_layers:
            features.append(x)
        x = self.layer2_forward(x)
        if "layer2" in self.feature_layers:
            features.append(x)
        x = self.layer3_forward(x)
        if "layer3" in self.feature_layers:
            features.append(x)
        x = self.layer4_forward(x)
        if "layer4" in self.feature_layers:
            features.append(x)
        
        x = self.fc_forward(x)
        return features, x
        
    def conv1_forward(self, x):
        x = self.conv1(x)
        return x
    
    def bn1_forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
    def layer1_forward(self, x):
        x = self.layer1(x)
        return x
    
    def layer2_forward(self, x):
        x = self.layer2(x)
        return x

    def layer3_forward(self, x):
        x = self.layer3(x)
        return x
    
    def layer4_forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def fc_forward(self, x):
        x = self.fc(x)
        return x
    
    layer_to_forward_fn = {
        "conv1": conv1_forward,
        "bn1": bn1_forward,
        "layer1": layer1_forward, 
        "layer2": layer2_forward, 
        "layer3": layer3_forward,
        "layer4": layer4_forward,
        "fc": fc_forward,
    }

resnet_layer_modes = {
        "none": [],
        "all": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"],
        
        "to_layer1": ["conv1", "bn1", "layer1"],
        "to_layer2": ["conv1", "bn1", "layer1", "layer2"],
        "to_layer3": ["conv1", "bn1", "layer1", "layer2", "layer3"],
        "to_layer4": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
        "layer1": ["layer1"],
        "layer2": ["layer2"],
        "layer3": ["layer3"],
        "layer4": ["layer4"],
        

        "top_fc": ["fc"],
        "top_layer4": ["layer4", "fc"],
        "top_layer3": ["layer3", "layer4", "fc"],
    }

def set_block_mode(model):
    param_names = []
    for name in model.cpu().state_dict().keys():
        param_names.append(name)
    model_layer_modes = {}
    for mode_key in resnet_layer_modes.keys():
        model_layer_modes[mode_key] = []
        for mode_key_layer in resnet_layer_modes[mode_key]:
            for name in param_names:
                if mode_key_layer == name.split('.')[0]:
                    assert name not in model_layer_modes[mode_key]
                    model_layer_modes[mode_key].append(name)
    model.avgmode_to_layers = model_layer_modes

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        weights = model_zoo.load_url(model_urls['resnet18'])
        del weights['fc.weight']
        del weights['fc.bias']
        model.load_state_dict(weights, strict=False)
        logging.info("************ Load pretrained resnet18 model")
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights = model_zoo.load_url(model_urls['resnet50'])
        del weights['fc.weight']
        del weights['fc.bias']
        model.load_state_dict(weights, strict=False)
        logging.info("************ Load pretrained resnet50 model")
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
