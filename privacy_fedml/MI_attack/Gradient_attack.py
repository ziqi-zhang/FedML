import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

import logging
from pdb import set_trace as st
import os.path as osp
import os
import copy
import wandb


from fedml_api.data_preprocessing.utils import DatasetSplit
from privacy_fedml.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .MI_attack_model_trainer import MIAttackModelTrainer
from .NN_attack import NNAttack, NNAttackModel

class GradientAttackModel(nn.Module):
    def __init__(self, input_dim, grad_dim, n_classes):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        self.grad_dim = grad_dim
        self.grad_fc1 = nn.Linear(grad_dim, 256)
        self.grad_fc2 = nn.Linear(256, 128)
        
        self.fc4 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        assert x.shape[1] == self.input_dim + self.grad_dim
        x, grad = torch.split(x, (self.input_dim, self.grad_dim), dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        grad = F.relu(self.grad_fc1(grad))
        grad = F.relu(self.grad_fc2(grad))
        x = torch.cat((x, grad), dim=1)
        
        # x = self.dropout(x)
        x = self.fc4(x)
        
        # return F.log_softmax(x, dim=1)
        return x
    
class GradientAttack(NNAttack):
    def __init__(self, server, device, args, adv_client_idx=0, adv_branch_idx=0):
        super(GradientAttack, self).__init__(
            server, device, args, adv_client_idx, adv_branch_idx
        )
        self.attack_model = GradientAttackModel(
            server.output_dim, server.model_trainer.model.penultimate_dim, 2)
        self.attack_trainer = MIAttackModelTrainer(self.attack_model)

    def process_output(self, pred, target, criterion, model):
        losses = criterion(pred, target)
        for loss in losses:
            loss.backward(retain_graph=True)
        # losses[0].backward()
        p_grad = model.penultimate.grad.cpu().numpy()
        
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.detach().cpu().numpy()
        pred = -np.sort(-pred)
        output = np.concatenate((pred, p_grad), axis=1)
        
        return output
    
    def test_shadow_output(self, model, test_data):
        device = self.device
        model.to(device)
        model.eval()
        model.open_penultimate_log()

        outputs = []
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)

        for batch_idx, (x, target) in enumerate(test_data):
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            assert len(pred.shape) > 1
            
            processed_output = self.process_output(
                pred, target, criterion, model
            )
            outputs.append(processed_output)
        outputs = np.concatenate(outputs)
        model.close_penultimate_log()
        return outputs