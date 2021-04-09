import copy
import logging
import random

import numpy as np
import torch
import wandb
from pdb import set_trace as st

from privacy_fedml.client import Client
from .fedavg_api import FedAvgAPI
from .predavg_api import PredAvgAPI
from .model.pred_avg import PredAvgEnsemble
from .model.pred_vote import PredVoteEnsemble
from .model.pred_weight import PredWeight
from .model.pred_weight_class import PredWeightClass

from .server_data import load_server_data

from .my_model_trainer_classification import MyModelTrainer


class BlockAvgAPI(PredAvgAPI):
    def __init__(self, dataset, device, args, model_trainer, output_dim=10):
        super(BlockAvgAPI, self).__init__(dataset, device, args, model_trainer, output_dim)
        self.server_model = PredAvgEnsemble(self.client_list)
        self.branch_num_samples = [0 for branch in range(args.branch_num)]
        
        # param_names = ""
        # for name, _ in self.model_trainer.model.state_dict().keys():
        #     param_names += f"\"{name}\", "
        # print(param_names)
        # st()
        
        
        assert args.avg_mode in self.model_trainer.model.avgmode_to_layers.keys()
        self.avg_param_names = self.model_trainer.model.avgmode_to_layers[args.avg_mode]
        separate_param_names = []
        for name in self.model_trainer.model.cpu().state_dict().keys():
            if name not in self.avg_param_names:
                separate_param_names.append(name)
        logging.info(f"====== Block avg params: {self.avg_param_names}")
        logging.info(f"====== Separate params: {separate_param_names}")
        # print(len(self.avg_param_names))
        
        # self.set_server_weight()
        # self.train_server_weight(0)
        # self.server_test_on_global_dataset(0)
        # self._set_client_branch(0)
        


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
                self.branch_num_samples[self.client_to_branch[idx]] = client.get_sample_number()

            # update global weights
            # print("0:  ", self.branches[0]["conv2d_1.bias"])
            # print("1:  ", self.branches[1]["conv2d_1.bias"])
            # st()
            w_global = self._aggregate()
            # print("0:  ", self.branches[0]["conv2d_1.bias"])
            # print("1:  ", self.branches[1]["conv2d_1.bias"])
            # st()
            # self.model_trainer.set_model_params(w_global)
            # self.branches = [w[1] for w in w_locals]
            
            # test results
            # at last round
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
                    

    def _aggregate(self):
        training_num = 0
        for sample_num in self.branch_num_samples:
            training_num += sample_num
        
        for k in self.avg_param_names:
            for i in range(0, self.branch_num):
                local_sample_number = self.branch_num_samples[i]
                branch_param = self.branches[i][k]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_param = branch_param * w
                else:
                    averaged_param += branch_param * w
            for i in range(0, self.branch_num):
                # self.branches[i][k] = copy.deepcopy(averaged_param)
                self.branches[i][k] = averaged_param
            logging.info(f"====== Average {k}")


