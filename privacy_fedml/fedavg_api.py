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


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer, output_dim=10):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.output_dim = output_dim

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.branches = [self.model_trainer.get_model_params() for branch in range(args.branch_num)]
        self.branch_num = args.branch_num
        # self._set_client_branch(0)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")
        
    def _set_client_branch(self, round_idx):
        self.branch_to_client, self.client_to_branch = {}, {}
        for idx in range(self.args.client_num_per_round):
            branch_idx = idx % self.branch_num
            # self.branch_to_client[branch_idx] = idx
            if branch_idx not in self.branch_to_client:
                self.branch_to_client[branch_idx] = [idx]
            else:
                self.branch_to_client[branch_idx].append(idx)
            self.client_to_branch[idx] = branch_idx
        
    def reset_accumulate_weight(self):
        self.accumulate_state_dict = {}
        for key, tensor in self.model_trainer.get_model_params().items():
            self.accumulate_state_dict[key] = copy.deepcopy(tensor)
            self.accumulate_state_dict[key].zero_()
        
    def update_accumulate_weight(self, w_locals):
        for key, tensor in w_locals.items():
            self.accumulate_state_dict[key] += tensor
            
    def average_accumulate_weight(self):
        for key, tensor in self.accumulate_state_dict.items():
            self.accumulate_state_dict[key] = tensor / self.args.client_num_per_round
        return self.accumulate_state_dict

    def train(self):
        # w_global = self.model_trainer.get_model_params()
        # self.branches = [w_global for _ in self.client_list]
        for round_idx in range(self.args.comm_round):
            
            logging.info("################Communication round : {}".format(round_idx))
            self._set_client_branch(round_idx)
            w_locals = []
            self.reset_accumulate_weight()

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
                branch_w = self.branches[self.client_to_branch[idx]]
                w = client.train(branch_w)
                # self.logger.info("local weights = " + str(w))
                # w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                # self.branches[self.client_to_branch[idx]] = w_locals[-1][1]
                self.update_accumulate_weight(w)
                logging.info(f"Round {round_idx}, client {idx}")

            # update global weights
            # w_global = self._aggregate(w_locals)
            w_global = self.average_accumulate_weight()
            self.model_trainer.set_model_params(w_global)
            self.branches = [w_global for _ in self.client_list]

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif (round_idx+1) % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
            
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_own_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            self.set_client_weight(client, client_idx)
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
            
            acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            logging.info(f"Client {client_idx}, other dataset {client_idx}, acc {acc:.2f}")

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)



    def set_client_weight(self, client, branch_idx):
        # w_global = self.model_trainer.get_model_params()
        client.model_trainer.set_model_params(self.branches[branch_idx])
        # client.model_trainer.set_model_params(w_global)
        
    def set_server_weight(self, client):
        w_global = self.model_trainer.get_model_params()
        client.model_trainer.set_model_params(w_global)
        
    def local_test_on_global_dataset(self, round_idx):
        logging.info("################local_test_on_GLOBAL_DATASET : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': [],
            'accs': [],
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # Set up client weight
            self.set_client_weight(client, client_idx)
            if self.test_data_local_dict[client_idx] is None:
                logging.info("Local test None")
                continue
            client.update_local_dataset(0, None,
                                        self.test_global,
                                        None)

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
            acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            test_metrics['accs'].append(acc)

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on test dataset
        # test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        # test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_acc = np.mean(test_metrics['accs'])

        stats = {'test_acc': test_acc}
        wandb.log({"TestGlobalDataset/Acc": test_acc, "round": round_idx})
        # wandb.log({"TestGlobalDataset/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        
    def local_test_on_next_client_dataset(self, round_idx):
        OTHER_CLIENT_TEST_NUM = 10
        # Only test OTHER_CLIENT_TEST_NUM clients for simplicity
        logging.info("################local_test_on_NEXT_CLIENT_DATASET : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': [],
            'accs': [],
        }

        client = self.client_list[0]
        
        for client_idx in range(min(OTHER_CLIENT_TEST_NUM, len(self.client_list))):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # Set up client weight
            self.set_client_weight(client, client_idx)
            test_dataset_idx = (client_idx+1) % len(self.client_list)

            client.update_local_dataset(0, None,
                                        self.test_data_local_dict[test_dataset_idx],
                                        None)

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
            acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            test_metrics['accs'].append(acc)
            logging.info(f"Client {client_idx}, other dataset {test_dataset_idx}, acc {acc:.2f}")

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            
        # test on test dataset
        # test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_acc = np.mean(test_metrics['accs'])
        # test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'test_acc': test_acc}
        wandb.log({"TestNextClientDataset/Acc": test_acc, "round": round_idx})
        # wandb.log({"TestOtherClientDataset/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        
    def local_test_on_other_client_dataset(self, round_idx):
        OTHER_CLIENT_TEST_NUM = 10
        # Only test OTHER_CLIENT_TEST_NUM clients for simplicity
        logging.info("################local_test_on_OTHER_CLIENT_DATASET : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': [],
            'accs': [],
        }

        client = self.client_list[0]
        
        for branch_idx in range(self.branch_num):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # Set up client weight
            self.set_client_weight(client, branch_idx)
            for test_dataset_idx in range(len(self.client_list)):
                if test_dataset_idx in self.branch_to_client[branch_idx]:
                    continue

                client.update_local_dataset(0, None,
                                            self.test_data_local_dict[test_dataset_idx],
                                            None)

                # test data
                test_local_metrics = client.local_test(True)
                acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
                test_metrics['accs'].append(acc)
                logging.info(f"Branch {branch_idx}, other dataset (client) {test_dataset_idx}, acc {acc:.2f}")

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break
            
        # test on test dataset
        test_acc = np.mean(test_metrics['accs'])

        stats = {'test_acc': test_acc}
        wandb.log({"TestOtherClientDataset/Acc": test_acc, "round": round_idx})
        logging.info(stats)

    def server_test_on_global_dataset(self, round_idx):
        ...
        
    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!"%self.args.dataset)

        logging.info(stats)
        
    def save_branch_state(self):
        
        path = osp.join(self.args.save_dir, "branches.pt")
        logging.info(f"################Save branch states to {path}")
        torch.save(self.branches, path)
        path = osp.join(self.args.save_dir, "client_branch_map.pt")
        data = (self.client_to_branch, self.branch_to_client)
        torch.save(data, path)
        
        
    def load_branch_state(self):
        path = osp.join(self.args.save_dir, "branches.pt")
        logging.info(f"################Load branch states from {path}")
        self.branches = torch.load(path)
        
        self._set_client_branch(0)
        
    def set_client_dataset(self):
        # Load the client dataset
        
        client_indexes = self._client_sampling(
            0, self.args.client_num_in_total,
            self.args.client_num_per_round
        )
        logging.info("client_indexes = " + str(client_indexes))
        for idx, client in enumerate(self.client_list):
            # update dataset
            client_idx = client_indexes[idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])