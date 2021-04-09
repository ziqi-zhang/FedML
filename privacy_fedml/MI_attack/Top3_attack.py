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

class Top3Attack(NNAttack):
    def __init__(self, server, device, args, adv_client_idx=0, adv_branch_idx=0):
        super(Top3Attack, self).__init__(
            server, device, args, adv_client_idx, adv_branch_idx
        )
        self.attack_model = NNAttackModel(3, 2)
        self.attack_trainer = MIAttackModelTrainer(self.attack_model)

    def process_output(self, pred, target, criterion):
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.cpu().numpy()
        pred = -np.sort(-pred)
        pred = pred[:,:3]
        return pred