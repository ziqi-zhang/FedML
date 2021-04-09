import copy
import logging
import random
import os
import os.path as osp

import numpy as np
import torch
import wandb
from pdb import set_trace as st

from privacy_fedml.client import Client
from .fedavg_api import FedAvgAPI
from .model.hetero_feat_avg import HeteroFeatAvgEnsemble
from privacy_fedml.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from privacy_fedml.one_model_trainer import OneModelTrainer
from privacy_fedml.two_model_trainer import TwoModelTrainer
from privacy_fedml.three_model_trainer import ThreeModelTrainer

class HeteroEnsembleAPI(FedAvgAPI):
    def __init__(self, dataset, device, args, model_trainer, output_dim=10):
        super(HeteroEnsembleAPI, self).__init__(dataset, device, args, model_trainer, output_dim)
        
        
        self.branch_to_client, self.client_to_branch = {}, {}

        self.hetero_archs = model_trainer.model.hetero_arch_fn(args.branch_num)
        logging.info(f"############# Branch Archs")
        for branch_idx, model in enumerate(self.hetero_archs):
            logging.info(f"Branch {branch_idx}: {model.flatten_structure()}")
        logging.info(f"#############")
        
        if args.ensemble_method == "featavg":
            self.server_model = HeteroFeatAvgEnsemble(
                self.hetero_archs, self.branches, self.client_list
            )
        else:
            raise NotImplementedError
        # self.server_trainer = copy.deepcopy(model_trainer)
        self.server_trainer = MyModelTrainerCLS(self.server_model)
        self.server_trainer.model = self.server_model
        
        """
        self.branches: List[
                state_dict: Dict[param_name(str): weight(tensor)]
            ]
        self.branch_state_dict_keys: List[ [str] ]
        """
        self.branches = []
        self.branch_state_dict_keys = []
        for branch_idx in range(args.branch_num):
            self.hetero_archs[branch_idx].weight_reinit()
            weights = copy.deepcopy(self.hetero_archs[branch_idx].state_dict())
            self.branches.append(weights)
            self.branch_state_dict_keys.append(weights.keys())
        # print(self.branches[0]['conv2d_1_block.0.weight'][0])
        # print(self.branches[1]['conv2d_1_block.0.weight'][0])
        # st()
        # self.branches = [copy.deepcopy(self.model_trainer.get_model_params()[0]) for branch in range(args.branch_num)]
        
        # self.branches = [self.model_trainer.get_model_params() for branch in range(args.branch_num)]
        self.model_class = type(model_trainer.model)
        self.model_class.blocks = [
            block if isinstance(block, list) else [block]
            for block in self.model_class.blocks
        ]
        self.fake_model = copy.deepcopy(model_trainer.model)
        
        # List[str]
        # self.model_class.state_dict_keys = self.branches[0].keys()
        
        """
        Set clients per branch trained on, theoretically the max number is round * 2 
        """
        if args.clients_per_branch == -1:
            self.branch_to_client_mapping = None
            self.client_to_branch_mapping = None
        else:
            assert args.clients_per_branch < args.client_num_per_round
            self.branch_to_client_mapping = []
            self.client_to_branch_mapping = [[] for _ in range(args.client_num_in_total)]
            for branch_idx in range(args.branch_num):
                selected_clients = np.random.choice(np.arange(args.client_num_per_round), size=args.clients_per_branch, replace=False)
                self.branch_to_client_mapping.append(selected_clients)
                for client_idx in selected_clients:
                    self.client_to_branch_mapping[client_idx].append(branch_idx)
            for client_idx in range(args.client_num_in_total):
                if len(self.client_to_branch_mapping[client_idx]) < 2:
                    selected_branch = np.random.choice(np.arange(args.branch_num), size=2, replace=False)
                    self.client_to_branch_mapping[client_idx] += list(selected_branch)
                    for branch_idx in selected_branch:
                        self.branch_to_client_mapping[branch_idx] = np.append(self.branch_to_client_mapping[branch_idx], client_idx)
            logging.info(f"######## Client to branch setting")
            for client_idx in range(args.client_num_in_total):
                logging.info(f"Client {client_idx}: {self.client_to_branch_mapping[client_idx]}")
            logging.info(f"#################################")
            logging.info(f"######## Branch to client setting")
            for branch_idx in range(args.branch_num):
                logging.info(f"Branch {branch_idx}: {self.branch_to_client_mapping[branch_idx]}")
            logging.info(f"#################################")
        
        
        """
        Set branch training info count
        self.branch_trained_count: 
        Dict(
            branch_idx (int): Dict(
                                block_idx(int): count(int)
                            )
        )
        """
        self.branch_trained_count = {}
        for branch_idx in range(self.branch_num):
            self.branch_trained_count[branch_idx] = {}
            for block_idx in range(len(self.model_class.blocks)):
                self.branch_trained_count[branch_idx][block_idx] = 0
                
        # Set block to param_name
        """
        self.block_to_param_name:   
        Dict(
            block_idx(int): List(param_name)
        )
        """
        
        self.branch_block_to_param_name = []
        for branch_idx in range(self.branch_num):
            block_to_param_name = {}
            for block_idx, blocks in enumerate(self.model_class.blocks):
                block_to_param_name[block_idx] = []
                for block in blocks:
                    for param_name in self.branch_state_dict_keys[branch_idx]:
                        if block == param_name.split('.')[0]:
                            assert param_name not in block_to_param_name[block_idx]
                            block_to_param_name[block_idx].append(param_name)
            self.branch_block_to_param_name.append(block_to_param_name)
        # assert args.feat_lmda != 0
        
        num_local_test = min(20, len(self.client_list))
        self.local_test_clients = np.random.choice(
            np.arange(len(self.client_list)), size=num_local_test, replace=False
        )
        

    def reset_branch_count(self):
        for branch_idx in range(self.branch_num):
            for block_idx in range(len(self.model_class.blocks)):
                self.branch_trained_count[branch_idx][block_idx] = 0
                
        
    def ensemble_branch(self, ensemble_block_info):
        
        ensemble_param_info = {}
        for block_idx, (blocks, branch_idx) in ensemble_block_info.items():
            ensemble_param_info[block_idx] = (
                blocks, branch_idx, self.branch_block_to_param_name[branch_idx][block_idx]
            )

        for block_idx, (blocks, branch_idx, names) in ensemble_param_info.items():
            logging.info(f"Block {blocks} from branch {branch_idx}, params: {names}")
        model = copy.deepcopy(self.fake_model)
        state_dict = {}
        for block_idx, (blocks, branch_idx, names) in ensemble_param_info.items():
            block_arch = self.branch_block_to_param_name[branch_idx][block_idx]
            for block in blocks:
                block_arch = self.hetero_archs[branch_idx]._modules[block]
                model._modules[block] = block_arch
            
            for name in names:
                state_dict[name] = copy.deepcopy(self.branches[branch_idx][name])
                # logging.info(f"{name} from branch {branch_idx}")
        model.load_state_dict(state_dict)
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(name)
        # st()
        
        return model, ensemble_param_info
        
    def update_training_branch_count(self, ensemble_block_info):
        for block_idx, (block, branch_idx) in ensemble_block_info.items():
            self.branch_trained_count[branch_idx][block_idx] += 1
        
    def prepare_branch_dict(self, client_idx):
        branch_select_info = {
            block_idx: (block, np.random.choice(self.client_to_branch_mapping[client_idx], size=2, replace=False))
            for block_idx, block in enumerate(self.model_class.blocks)
        }
        
        # branch_select_info = {
        #     block_idx: (block, [0,1])
        #     for block_idx, block in enumerate(self.model_class.blocks)
        # }
        
        """
        ensemble_block_info1: 
        Dict(
            block_idx (int): tuple(block, branch_idx)
        )
        """
        ensemble_block_info1 = {
            block_idx: (block, branch[0]) for block_idx, (block, branch) in branch_select_info.items()
        }
        """
        ensemble_info1: 
        Dict(
            block_idx (int): tuple(block, branch_idx, param_names)
        )
        """
        logging.info(f"========= Ensemble branches 1st path =========")
        model1, ensemble_info1 = self.ensemble_branch(ensemble_block_info1)
        logging.info(f"Arch: {model1.flatten_structure()}")
        self.update_training_branch_count(ensemble_block_info1)
        
        if isinstance(self.model_trainer, TwoModelTrainer):
            ensemble_block_info2 = {
                block_idx: (block, branch[1]) for block_idx, (block, branch) in branch_select_info.items()
            }
            
            logging.info(f"========= Ensemble branches 2nd path =========")
            model2, ensemble_info2 = self.ensemble_branch(ensemble_block_info2)
            logging.info(f"Arch: {model2.flatten_structure()}")
            self.update_training_branch_count(ensemble_block_info2)
            
            return (model1, model2), (ensemble_block_info1, ensemble_block_info2)
        elif isinstance(self.model_trainer, ThreeModelTrainer):
            ensemble_block_info2 = {
                block_idx: (block, branch[1]) for block_idx, (block, branch) in branch_select_info.items()
            }
            logging.info(f"========= Ensemble branches 2nd path =========")
            model2, ensemble_info2 = self.ensemble_branch(ensemble_block_info2)
            logging.info(f"Arch: {model2.flatten_structure()}")
            self.update_training_branch_count(ensemble_block_info2)
            
            ensemble_block_info3 = {
                block_idx: (block, branch[1]) for block_idx, (block, branch) in branch_select_info.items()
            }
            
            logging.info(f"========= Ensemble branches 3nd path =========")
            model3, ensemble_info3 = self.ensemble_branch(ensemble_block_info2)
            logging.info(f"Arch: {model3.flatten_structure()}")
            self.update_training_branch_count(ensemble_block_info2)
            
            return (model1, model2, model3), (ensemble_block_info1, ensemble_block_info2, ensemble_block_info3)
        
        
        elif isinstance(self.model_trainer, OneModelTrainer):
            return (model1, ), (ensemble_block_info1,)
    
    def reset_updated_branch_params(self):
        self.updated_branches = copy.deepcopy(self.branches)
        for branch_idx in range(self.branch_num):
            for name in self.updated_branches[branch_idx].keys():
                self.updated_branches[branch_idx][name].zero_()
                
    def update_branch_params(self, state_dict_list, ensemble_info_list):
        
        for state_dict, info in zip(state_dict_list, ensemble_info_list):
            for block_idx, (blocks, branch_idx) in info.items():
                param_names = self.branch_block_to_param_name[branch_idx][block_idx]
                
                for name in param_names:
                    self.updated_branches[branch_idx][name] += state_dict[name]
    # def update_branch_params(self, state_dict, info):
    #     for block_idx, (blocks, branch_idx) in info.items():
    #         param_names = self.block_to_param_name[block_idx]
    #         for name in param_names:
    #             self.updated_branches[branch_idx][name] += state_dict[name]
            
    def average_updated_branch_params(self):
        for branch_idx in range(self.branch_num):
            for block_idx, param_names in self.branch_block_to_param_name[branch_idx].items():
                cnt = self.branch_trained_count[branch_idx][block_idx]
                for name in param_names:
                    if cnt != 0:
                        self.updated_branches[branch_idx][name] /= cnt
                    else:
                        self.updated_branches[branch_idx][name] = copy.deepcopy(self.branches[branch_idx][name])
        
    def train(self):
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))
            self.reset_updated_branch_params()
            self.reset_branch_count()
            if (
                round_idx == self.args.comm_round - 1 or 
                (round_idx+1) % self.args.frequency_of_the_test == 0
            ):
                post_client_global_acc, pre_client_global_acc = [], []
            
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                
                # train on new dataset
                logging.info(f"############### Round {round_idx} client {idx} train ")
                model_pair, ensemble_info = self.prepare_branch_dict(client_idx=idx)
                if (
                    (round_idx == self.args.comm_round - 1 or 
                    (round_idx+1) % self.args.frequency_of_the_test == 0) and
                    idx in self.local_test_clients
                ):
                    client.model_trainer.set_model_arch_weights(model_pair)
                    client.update_local_dataset(
                        0, None,self.test_global, None
                    )
                    # test data
                    test_local_metrics = client.local_test(True)
                    pre_acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
                    pre_client_global_acc.append(pre_acc)
                    ...
                    
                client.update_local_dataset(
                    client_idx, self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )
                trained_model_pair = client.train_with_arch(model_pair)
                
                if (
                    (round_idx == self.args.comm_round - 1 or 
                    (round_idx+1) % self.args.frequency_of_the_test == 0) and
                    idx in self.local_test_clients
                ):
                    client.update_local_dataset(
                        0, None,self.test_global, None
                    )
                    # test data
                    test_local_metrics = client.local_test(True)
                    post_acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
                    post_client_global_acc.append(post_acc)
                    ...
                
                self.update_branch_params(trained_model_pair, ensemble_info)
            
            self.average_updated_branch_params()
            self.branches = self.updated_branches
            
            # test results
            # at last round
            if (
                round_idx == self.args.comm_round - 1 or 
                (round_idx+1) % self.args.frequency_of_the_test == 0
            ):
                self.server_test_on_global_dataset(round_idx)
                
                client_acc = np.mean(post_client_global_acc)
                stats = {'test_acc': client_acc, }
                wandb.log({"TestPostClientGlobalDataset/Acc": client_acc, "round": round_idx})
                logging.info("################LOCAL_POST_test_on_GLOBAL_DATASET : {}".format(round_idx))
                logging.info(stats)
                
                client_acc = np.mean(pre_client_global_acc)
                stats = {'test_acc': client_acc, }
                wandb.log({"TestPreClientGlobalDataset/Acc": client_acc, "round": round_idx})
                logging.info("################LOCAL_PRE_test_on_GLOBAL_DATASET : {}".format(round_idx))
                logging.info(stats)


    def set_server_weight(self):
        self.server_trainer.model.load_branch_to_models(self.branches, self.client_list)
        
    def server_test_on_global_dataset(self, round_idx):
        logging.info("################SERVER_test_on_GLOBAL_DATASET : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        self.set_server_weight()
        
        metrics = self.server_trainer.test(self.test_global, self.device, self.args)
        # test_metrics['num_samples'].append(copy.deepcopy(metrics['test_total']))
        # test_metrics['num_correct'].append(copy.deepcopy(metrics['test_correct']))
        # test_metrics['losses'].append(copy.deepcopy(metrics['test_loss']))

        # test on test dataset
        # test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        # test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_acc = metrics['test_correct'] / metrics['test_total']

        stats = {'test_acc': test_acc, }
        wandb.log({"TestServerGlobalDataset/Acc": test_acc, "round": round_idx})
        logging.info(stats)

    def save_branch_state(self):
        
        path = osp.join(self.args.save_dir, "branches.pt")
        logging.info(f"################Save branch states to {path}")
        torch.save(self.branches, path)
        path = osp.join(self.args.save_dir, "archs.pt")
        logging.info(f"################Save archs to {path}")
        torch.save(self.hetero_archs, path)
        # path = osp.join(self.args.save_dir, "client_branch_map.pt")
        # data = (self.client_to_branch, self.branch_to_client)
        # torch.save(data, path)
        
        
    def load_branch_state(self):
        path = osp.join(self.args.save_dir, "branches.pt")
        logging.info(f"################Load branch states from {path}")
        self.branches = torch.load(path)
        path = osp.join(self.args.save_dir, "archs.pt")
        logging.info(f"################Load archs states from {path}")
        self.hetero_archs = torch.load(path)
        if self.args.ensemble_method == "featavg":
            self.server_model = HeteroFeatAvgEnsemble(
                self.hetero_archs, self.branches, self.client_list
            )
        else:
            raise NotImplementedError
        self.set_server_weight()
        # self._set_client_branch(0)