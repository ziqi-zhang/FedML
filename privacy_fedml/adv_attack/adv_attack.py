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
from privacy_fedml.two_model_trainer import TwoModelWarpper
    
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2CarliniWagnerAttack
import eagerpy as ep

from privacy_fedml.model.hetero_feat_avg import HeteroFeatAvgEnsembleDefense


# def accuracy(fmodel: Model, inputs: Any, labels: Any) -> float:
#     inputs_, labels_ = ep.astensors(inputs, labels)
#     del inputs, labels

#     predictions = fmodel(inputs_).argmax(axis=-1)
#     accuracy = (predictions == labels_).float32().mean()
#     return accuracy.item()

class AdvAttack():
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
            
        
        # set adv client model
        params = self.set_attack_params()
        ensemble_info = self.prepare_advclient_model()
        self.prepare_server_model(ensemble_info)
        
        # test clean accuracys)
        # clean_acc = self.check_accuracy(
        #     self.advclient_model, self.server.test_global, self.device
        # )
        # logging.info(f"clean accuracy:  {clean_acc * 100:.1f} %")
        # st()
        
        self.attack_fn = LinfPGD(**params)
        # self.attack_fn = L2CarliniWagnerAttack(**params)
        self.attack()
        

    def set_attack_params(self):
        if self.args.dataset in ["mnist", "fmnist", "emnist"]:
            range = 1 / 0.3081
            params = {
                "steps": 10, "rel_stepsize": 0.1,
            }
            self.epsilon = range * 0.3
            bound_min = (0-0.1307) / 0.3081
            bound_max = (1-0.1307) / 0.3081
            self.bounds = (bound_min, bound_max)
            self.preprocessing = None
            

            
        elif self.args.dataset in ['cifar10', 'cifar100']:
            CIFAR_MEAN = torch.Tensor([0.49139968, 0.48215827, 0.44653124]).view(1,3,1,1).to(self.device)
            CIFAR_STD = torch.Tensor([0.24703233, 0.24348505, 0.26158768]).view(1,3,1,1).to(self.device)
            range = 1 / 0.3081
            params = {
                "steps": 20, "rel_stepsize": 0.1,
            }
            self.epsilon = 8/255
            # bound_min = (0-0.1307) / 0.3081
            # bound_max = (1-0.1307) / 0.3081
            self.bounds = (0, 1)
            self.preprocessing = {
                "mean": CIFAR_MEAN, "std": CIFAR_STD
            }
        else:
            raise NotImplementedError
        return params
    
    def check_accuracy(self, fmodel, dataloader, device) -> float:
        total, correct = 0., 0.
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x = x.to(device)
                target = target.to(device)
                if self.preprocessing is not None:
                    x = x * self.preprocessing["std"] + self.preprocessing["mean"]
                predictions = fmodel(x).argmax(axis=-1)
                
                total += len(x)
                correct += (predictions == target).sum()
                
        accuracy = correct / total
        return accuracy.item()
        
    def prepare_advclient_model(self):
        if self.args.aggr == "fedavg":
            adv_client = self.server.client_list[self.adv_client_idx]
            branch_w = self.server.branches[self.adv_branch_idx]
            adv_client.model_trainer.set_model_params(branch_w)
            self.advclient_model = copy.deepcopy(adv_client.model_trainer.model)
            ensemble_info = None
        elif self.args.aggr == "blockensemble":
            state_dict_pair, ensemble_info = self.server.prepare_branch_dict()
            self.adv_client = self.server.client_list[self.adv_client_idx]
            self.adv_client.model_trainer.set_model_params(state_dict_pair)
            # self.advclient_model = TwoModelWarpper(
            #     self.adv_client.model_trainer.model1,
            #     self.adv_client.model_trainer.model2,
            # )
            self.advclient_model = self.adv_client.model_trainer.model2
        elif self.args.aggr == "heteroensemble":
            model_pair, ensemble_info = self.server.prepare_branch_dict(client_idx=0)
            self.adv_client = self.server.client_list[self.adv_client_idx]
            self.advclient_model = TwoModelWarpper(
                model_pair[0],
                model_pair[1],
            )
            # self.advclient_model = model_pair[0]
        else:
            raise NotImplementedError
        
        self.advclient_model.eval()
        self.advclient_model = PyTorchModel(
            self.advclient_model, bounds = self.bounds,
            preprocessing = self.preprocessing
        )
        return ensemble_info
        
    def prepare_server_model(self, ensemble_info):
        if self.args.aggr == "fedavg":
            server = self.server.client_list[self.adv_client_idx]
            branch_w = self.server.branches[self.adv_branch_idx]
            server.model_trainer.set_model_params(branch_w)
            self.server_model = copy.deepcopy(server.model_trainer.model)
        elif self.args.aggr == "blockensemble":
            self.server_model = self.server.server_model
        elif self.args.aggr == "heteroensemble":
            self.server_model = self.server.server_model
            
            # server_model = self.server.server_model
            # self.server_model = HeteroFeatAvgEnsembleDefense(
            #     server_model, ensemble_info
            # )
        else:
            raise NotImplementedError
        
        self.server_model.eval()
        self.server_model = PyTorchModel(
            self.server_model, bounds = self.bounds,
            preprocessing = self.preprocessing
        )
        
    def attack(self):
        total, correct, success, server_success = 0, 0, 0, 0
        client_total, server_total = 0, 0
        fmodel = self.advclient_model 
        
        server_model = self.server_model
        dataloader = self.server.test_global
        for batch_idx, (x, target) in enumerate(dataloader):
            x = x.to(self.device)
            target = target.to(self.device)
            if self.preprocessing is not None:
                x = x * self.preprocessing["std"] + self.preprocessing["mean"]
                # x = x * 255
            
            clean_pred = fmodel(x).argmax(axis=-1)
            pred_correct = clean_pred == target
            correct += (pred_correct).sum().float()
            
            raw_advs, clipped_advs, _success = self.attack_fn(fmodel, x, clean_pred, epsilons=self.epsilon)
            
            
            # total += len(x)
            # success += (_success).sum().float()
            
            # server_pred = server_model(clipped_advs).argmax(axis=-1)
            # server_wrong = (server_pred != target)
            # server_success += server_wrong.sum().float()
            
            
            # success += (_success*pred_correct).sum().float()
            server_clean_pred = server_model(x).argmax(axis=-1)
            client_valid = (clean_pred == target)
            client_total += client_valid.sum().float()
            success += (_success * client_valid).sum().float()
            server_adv_pred = server_model(clipped_advs).argmax(axis=-1)
            server_valid = (server_clean_pred == target)
            server_total += server_valid.sum().float()
            server_wrong = (server_adv_pred != target) * server_valid
            server_success += server_wrong.sum().float()
            # print(f"Total {total}, success {success}, server success {server_success}")
            # st()
            # diff = (x-clipped_advs).max()
            
            
            # for MNIST
            # x = x * 0.3081 + 0.1307
            # clipped_advs = clipped_advs * 0.3081 + 0.1307
            
            # for idx in range(5):
            #     img, adv_img = x[idx], clipped_advs[idx]
            #     img = (img*255).cpu().numpy().astype(np.uint8)
            #     adv_img = (adv_img*255).cpu().numpy().astype(np.uint8)
            #     img = np.concatenate([img, adv_img], axis=2)
            #     img = np.transpose(img, (1,2,0))
            #     path = osp.join("debug", f"{idx}.png")
            #     from PIL import Image
            #     for MNIST
            #     img = img.repeat(3, axis=2)
                
            #     im = Image.fromarray(img,)
            #     im.save(path)
                
            # diff = (x - clipped_advs).max()*255
            # st()
            
            
        success_rate = (success / client_total).item()
        server_success_rate = (server_success / server_total).item()
        clean_acc = (correct / total).item()
        logging.info(f"clean acc: {clean_acc*100:.1f}%, client success rate:  {success_rate * 100:.1f}%, server success rate: {server_success_rate*100:.1f}%" )
        return clean_acc, success_rate, server_success_rate
