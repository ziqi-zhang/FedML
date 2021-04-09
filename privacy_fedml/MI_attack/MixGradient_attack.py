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

    
    
    def generate_eval_dataset(self):
        """Generate eval dataset on other clients
        """
        path = osp.join(self.save_dir, "mi_other_client_test_data.pt")
        
        # if not os.path.exists(path):
        if True:
            
            adv_client = self.server.client_list[self.adv_client_idx]
            branch_w = self.server.branches[self.adv_branch_idx]
            adv_client.model_trainer.set_model_params(branch_w)
            test_local_metrics = adv_client.local_test(True)
            acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            logging.info(f"################ {type(self).__name__} init adv client (client {self.adv_client_idx} on branch {self.adv_branch_idx}) performance {acc:.2f}")
            
            client_eval_datasets = {}
            for client_idx in range(len(self.server.client_list)):
                if client_idx == self.adv_client_idx:
                    continue
                branch_idx = self.server.client_to_branch[client_idx]
                client = self.server.client_list[client_idx]
                branch_w = self.server.branches[branch_idx]
                client.model_trainer.set_model_params(branch_w)
                test_local_metrics = client.local_test(True)
                acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
                logging.info(f"################ {type(self).__name__} generate MI eval dataset on client {client_idx} & branch {branch_idx}, performance {acc:.2f}")
                
                train_dataset, test_dataset = self.convert_dataset(
                    (client.model_trainer.model, adv_client.model_trainer.model), 
                    client, client_idx,
                )
                client_eval_datasets[client_idx] = test_dataset
            torch.save(client_eval_datasets, path)
        else:
            client_eval_datasets = torch.load(path)
        return client_eval_datasets
    
    def process_output(self, target_pred, local_pred,  label, criterion, local_model):
        losses = criterion(local_pred, label)
        for loss in losses:
            loss.backward(retain_graph=True)
        # losses[0].backward()
        p_grad = local_model.penultimate.grad.cpu().numpy()
        
        target_pred = nn.Softmax(dim=1)(target_pred)
        target_pred = target_pred.detach().cpu().numpy()
        target_pred = -np.sort(-target_pred)
        output = np.concatenate((target_pred, p_grad), axis=1)
        
        return output
    
    def test_shadow_output(self, model, test_data):
        if isinstance(model, tuple):
            target_model, local_model = model
        else:
            target_model = model
            local_model = model
        device = self.device
        
        target_model.to(device)
        target_model.eval()
        local_model.to(device)
        local_model.eval()
        local_model.open_penultimate_log()

        outputs = []
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)

        for batch_idx, (x, label) in enumerate(test_data):
            x = x.to(device)
            label = label.to(device)
            if not isinstance(model, tuple):
                target_pred = local_pred = local_model(x)
            else:
                target_pred = target_model(x)
                local_pred = local_model(x)
            assert len(local_pred.shape) > 1
            
            processed_output = self.process_output(
                target_pred, local_pred, label, criterion, local_model
            )
            outputs.append(processed_output)
        outputs = np.concatenate(outputs)
        local_model.close_penultimate_log()
        return outputs