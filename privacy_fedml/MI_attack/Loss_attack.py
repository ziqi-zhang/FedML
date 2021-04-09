import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import logging
from pdb import set_trace as st
import os.path as osp
import os
import copy
import wandb
import math

from fedml_api.data_preprocessing.utils import DatasetSplit
from privacy_fedml.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .MI_attack_model_trainer import MIAttackModelTrainer, MIAttackThred
from .NN_attack import NNAttack, NNAttackModel

class LossAttack(NNAttack):
    def __init__(self, server, device, args, adv_client_idx=0, adv_branch_idx=0):
        super(LossAttack, self).__init__(
            server, device, args, adv_client_idx, adv_branch_idx
        )
        self.attack_model = None
        self.attack_trainer = MIAttackThred(None)

    def train_attack_model(self):
        train_dataset, test_dataset = self.generate_attack_dataset()
        
        self.attack_trainer.train(train_dataset)

        metrics = self.attack_trainer.test(
            test_dataset, 
        )
        logging.info(f"################ {type(self).__name__} adversary train on client {self.adv_client_idx} test result: {metrics}")
        
        prefix = f"{type(self).__name__}/OneClientTrain"
        wandb.log({f"{prefix}/TestAcc": metrics['acc'], "round": 0})
        wandb.log({f"{prefix}/TestPrec": metrics['precision'], "round": 0})
        wandb.log({f"{prefix}/TestRecall": metrics['recall'], "round": 0})
        wandb.log({f"{prefix}/TestF1": metrics['f1'], "round": 0})
        
        weight_path = osp.join(self.save_dir, "attack_model.pth")
        state_dict = self.attack_trainer.get_model_params()
        torch.save(state_dict, weight_path)
        
    def eval_on_other_client(self, ):
        """Eval the trained attack model on other client

        Args:
            model ([type]): [description]
        """
        client_eval_datasets = self.generate_eval_dataset()
        all_metrics = {
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for client_idx, test_dataset in client_eval_datasets.items():
            metrics = self.attack_trainer.test(
                test_dataset, 
            )
            logging.info(f"################ {type(self).__name__} test on other client {client_idx}: {metrics}")
            all_metrics["acc"].append(metrics["acc"])
            all_metrics["precision"].append(metrics["precision"])
            all_metrics["recall"].append(metrics["recall"])
            all_metrics["f1"].append(metrics["f1"])
        all_metrics["acc"] = np.mean(all_metrics["acc"])
        all_metrics["precision"] = np.mean(all_metrics["precision"])
        all_metrics["recall"] = np.mean(all_metrics["recall"])
        all_metrics["f1"] = np.mean(all_metrics["f1"])
        logging.info(f"################ {type(self).__name__} test on other client average: {all_metrics}")
        
        prefix = f"{type(self).__name__}/OtherClientTest"
        wandb.log({f"{prefix}/TestAcc": all_metrics['acc'], "round": 0})
        wandb.log({f"{prefix}/TestPrec": all_metrics['precision'], "round": 0})
        wandb.log({f"{prefix}/TestRecall": all_metrics['recall'], "round": 0})
        wandb.log({f"{prefix}/TestF1": all_metrics['f1'], "round": 0})
        
    def process_output(self, pred, target, criterion):
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        conf = np.array([p[label] for p, label in zip(pred, target)])
        loss = np.array(
            [-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in conf]
        )
        return loss
    
    def test_shadow_output(self, model, test_data):
        device = self.device
        model.to(device)
        model.eval()

        outputs = []
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                assert len(pred.shape) > 1
                
                processed_output = self.process_output(
                    pred, target, criterion
                )
                outputs.append(processed_output)
        outputs = np.concatenate(outputs)
        return outputs