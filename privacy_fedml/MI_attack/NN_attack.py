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

class NNAttackModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        
        # return F.log_softmax(x, dim=1)
        return x

# class NNAttackModel(nn.Module):
#     def __init__(self, input_dim, n_classes):
#         super().__init__()

#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, n_classes)

#         self.dropout = nn.Dropout(p=0.2)

#     def forward(self, x):
        
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         return x
    

class NNAttack():
    def __init__(self, server, device, args, adv_client_idx=0, adv_branch_idx=0):
        self.server = server
        server.set_client_dataset()
        self.device = device
        self.args = args
        self.adv_client_idx = adv_client_idx
        self.adv_branch_idx = adv_branch_idx
        self.save_dir = osp.join(self.args.save_dir, f"{type(self).__name__}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.attack_model = NNAttackModel(server.output_dim, 2)
        logging.info(f"################ {type(self).__name__} build attack model")
        logging.info(self.attack_model)
        
        self.attack_trainer = MIAttackModelTrainer(self.attack_model)
        attack_model_args = copy.deepcopy(args)
        attack_model_args.lr = 1e-1
        attack_model_args.batch_size = 64
        # attack_model_args.epochs = 40
        attack_model_args.epochs = 40
        attack_model_args.optimizer = 'sgd'
        self.attack_model_args = attack_model_args
        
    def eval_attack(self):
        self.train_attack_model()
        self.eval_on_other_client()
        
    def generate_attack_dataset(
        self, 
    ):
        """
        Generate attack data from the client training and testing data
        """
        path = osp.join(self.save_dir, "mi_train_data.pt")

        if True:
        # if not os.path.exists(path):
            # Set client weight
            adv_client = self.server.client_list[self.adv_client_idx]
            branch_w = self.server.branches[self.adv_branch_idx]
            adv_client.model_trainer.set_model_params(branch_w)
            test_local_metrics = adv_client.local_test(True)
            acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            logging.info(f"################ {type(self).__name__} init adv client performance {acc:.2f}")
            
            train_dataset, test_dataset = self.convert_dataset(
                adv_client.model_trainer.model, adv_client, self.adv_client_idx,
            )
            
            
            torch.save((train_dataset, test_dataset), path)
        else:
            train_dataset, test_dataset = torch.load(path)
        logging.info(f"################ {len(train_dataset)} training data, {len(test_dataset)} test data")
        
        return train_dataset, test_dataset
    
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
                client = self.server.client_list[client_idx]
                
                
                train_dataset, test_dataset = self.convert_dataset(
                    adv_client.model_trainer.model, client, client_idx,
                )
                client_eval_datasets[client_idx] = test_dataset
            torch.save(client_eval_datasets, path)
        else:
            client_eval_datasets = torch.load(path)
        return client_eval_datasets
    
    # use other client model
    # def generate_eval_dataset(self):
    #     """Generate eval dataset on other clients
    #     """
    #     path = osp.join(self.save_dir, "mi_other_client_test_data.pt")
        
    #     # if not os.path.exists(path):
    #     if True:
            
            
    #         client_eval_datasets = {}
    #         for client_idx in range(len(self.server.client_list)):
    #             if client_idx == self.adv_client_idx:
    #                 continue
    #             branch_idx = self.server.client_to_branch[client_idx]
    #             client = self.server.client_list[client_idx]
    #             branch_w = self.server.branches[branch_idx]
    #             client.model_trainer.set_model_params(branch_w)
    #             test_local_metrics = client.local_test(True)
    #             acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
    #             logging.info(f"################ {type(self).__name__} generate MI eval dataset on client {client_idx} & branch {branch_idx}, performance {acc:.2f}")
                
    #             train_dataset, test_dataset = self.convert_dataset(
    #                 client.model_trainer.model, client, client_idx,
    #             )
    #             client_eval_datasets[client_idx] = test_dataset
    #         torch.save(client_eval_datasets, path)
    #     else:
    #         client_eval_datasets = torch.load(path)
    #     return client_eval_datasets
            
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
            test_dataloader = DataLoader(
                test_dataset, batch_size = self.attack_model_args.batch_size,
                shuffle=False,
            )
            metrics = self.attack_trainer.test(
                test_dataloader, self.device, self.args
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
        
        
    def convert_dataset(self, model, client, client_idx):
        """Convert client training/test dataset to MI dataset

        Args:
            model ([type]): [description]
            client ([type]): [description]
        """
        client_train = client.local_training_data
        client_test = client.local_test_data
        
        mem_outputs = self.test_shadow_output(model, client_train)
        nonmem_outputs = self.test_shadow_output(model, client_test)
        
        # Mem and nonmem balance to avoid trivial result
        num_select = min(len(mem_outputs), len(nonmem_outputs))
        mem_idxs = np.random.choice(len(mem_outputs), num_select)
        mem_outputs = mem_outputs[mem_idxs]
        nonmem_idxs = np.random.choice(len(nonmem_outputs), num_select)
        nonmem_outputs = nonmem_outputs[nonmem_idxs]
        
        mem_labels = np.ones(len(mem_outputs))
        nonmem_labels = np.zeros(len(nonmem_outputs))
        
        mem_train_data, mem_test_data, mem_train_label, mem_test_label,  = train_test_split(
            mem_outputs, mem_labels, train_size=0.8, random_state=1,
        )
        nonmem_train_data, nonmem_test_data, nonmem_train_label, nonmem_test_label,  = train_test_split(
            nonmem_outputs, nonmem_labels, train_size=0.8, random_state=1,
        )
        logging.info(f"Client {client_idx} generate attack dataset mem_train: {len(mem_train_data)}, nonmem_train: {len(nonmem_train_data)}, mem_test: {len(mem_test_data)}, nonmen_test: {len(nonmem_test_data)}")

        train_data = np.concatenate([mem_train_data, nonmem_train_data])
        train_label = np.concatenate([mem_train_label, nonmem_train_label])
        test_data = np.concatenate([mem_test_data, nonmem_test_data])
        test_label = np.concatenate([mem_test_label, nonmem_test_label])
        
        train_data = torch.Tensor(train_data)
        train_label = torch.Tensor(train_label).long()
        test_data = torch.Tensor(test_data)
        test_label = torch.Tensor(test_label).long()
        train_dataset = TensorDataset(train_data, train_label)
        test_dataset = TensorDataset(test_data, test_label)
        return train_dataset, test_dataset
        
    def train_attack_model(self):
        train_dataset, test_dataset = self.generate_attack_dataset()
        train_dataloader = DataLoader(
            train_dataset, batch_size = self.attack_model_args.batch_size,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size = self.attack_model_args.batch_size,
            shuffle=False,
        )
        logging.info(f"################ {type(self).__name__} train attack model")
        self.attack_trainer.train(
            train_dataloader, self.device, self.attack_model_args
        )
        logging.info(f"################ {type(self).__name__} finish train")
        
        metrics = self.attack_trainer.test(
            test_dataloader, self.device, self.args
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
        
    def process_output(self, pred, target, criterion):
        pred = nn.Softmax(dim=1)(pred)
        pred = pred.cpu().numpy()
        pred = -np.sort(-pred)
        return pred
    
    def test_shadow_output(self, model, test_data):
        device = self.device
        model.to(device)
        model.eval()

        outputs = []
        criterion = nn.CrossEntropyLoss().to(device)
        
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
    
