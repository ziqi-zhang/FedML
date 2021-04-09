import copy
import logging
import random

import numpy as np
import torch
import wandb
from pdb import set_trace as st

from privacy_fedml.client import Client
from .fedavg_api import FedAvgAPI
from .model.pred_avg import PredAvgEnsemble
from .model.pred_vote import PredVoteEnsemble
from .model.pred_weight import PredWeight
from .model.pred_weight_class import PredWeightClass

from .server_data import load_server_data

from .my_model_trainer_classification import MyModelTrainer


class PredWeightAPI(FedAvgAPI):
    def __init__(self, dataset, device, args, model_trainer, output_dim=10):
        super(PredWeightAPI, self).__init__(dataset, device, args, model_trainer, output_dim)
        if args.ensemble_method == "predavg":
            self.server_model = PredAvgEnsemble(self.client_list)
        elif args.ensemble_method == "predvote":
            self.server_model = PredVoteEnsemble(self.client_list)
        elif args.ensemble_method == "predweight":
            self.server_model = PredWeight(self.branches, self.client_list, args, output_dim)
        elif args.ensemble_method == "predweightclass":
            self.server_model = PredWeightClass(self.branches, self.client_list, args, output_dim)
        else:
            raise NotImplementedError
        
        self.server_trainer = MyModelTrainer(model=self.server_model)
        self.server_train_data, self.server_test_data = load_server_data(args)
        
        # self.set_server_weight()
        # self.train_server_weight(0)
        # self.server_test_on_global_dataset(0)
        # self._set_client_branch(0)
        
    def _set_client_branch(self, round_idx):
        self.branch_to_client, self.client_to_branch = {}, {}
        client_per_branch = self.args.client_per_branch
        for idx in range(self.args.client_num_per_round):
            branch_idx = (idx- (round_idx%client_per_branch) ) % self.branch_num
            # self.branch_to_client[branch_idx] = idx
            if branch_idx not in self.branch_to_client:
                self.branch_to_client[branch_idx] = [idx]
            elif idx not in self.branch_to_client[branch_idx]:
                self.branch_to_client[branch_idx].append(idx)
            self.client_to_branch[idx] = branch_idx
        logging.info(f"Client branch map: {self.client_to_branch}")
        logging.info(f"Branch client map: {self.branch_to_client}")


    def train(self):
        # w_global = self.model_trainer.get_model_params()
        # self.branches = [copy.deepcopy(w_global) for _ in self.client_list]
        for round_idx in range(self.args.comm_round):
            self._set_client_branch(round_idx)
            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                logging.info(f"Round {round_idx} client {idx} train branch {self.client_to_branch[idx]}")
                branch_w = self.branches[self.client_to_branch[idx]]
                w = client.train(branch_w)
                # self.logger.info("local weights = " + str(w))
                # w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                self.branches[self.client_to_branch[idx]] = copy.deepcopy(w)

            # update global weights
            # w_global = self._aggregate(w_locals)
            # self.model_trainer.set_model_params(w_global)
            # self.branches = [w[1] for w in w_locals]
            
            # test results
            # at last round
            self.train_server_weight(round_idx)
            if round_idx == self.args.comm_round - 1:
                self.server_test_on_global_dataset(round_idx)
                self.local_test_on_global_dataset(round_idx)
                self.local_test_on_next_client_dataset(round_idx)
                self._local_test_on_all_clients(round_idx)
                self.local_test_on_other_client_dataset(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self.server_test_on_global_dataset(round_idx)
                    self.local_test_on_next_client_dataset(round_idx)
                    self._local_test_on_all_clients(round_idx)
                    

    def train_server_weight(self, round_idx):
        logging.info("################Train server weight : {}".format(round_idx))
        
        self.server_model.update_clients(self.branches, self.client_list)
        server_args = copy.deepcopy(self.args)
        server_args.epochs = self.args.server_epoch
        self.server_trainer.train(self.server_train_data, self.device, server_args)
        logging.info(self.server_model.branch_weight)
        # self.server_model.reset_module_grad()
        

    def set_client_weight(self, client, branch_idx):
        w_client = self.branches[branch_idx]
        client.model_trainer.set_model_params(w_client)
        
    def set_server_weight(self):
        self.server_trainer.model.update_clients(self.branches, self.client_list)
        
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
