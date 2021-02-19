import logging
from pdb import set_trace as st
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# from .datasets import CIFAR10_truncated
from ..utils import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_mnist_dataset(dataset, data_dir):
    
    # data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                    transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                    transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)
    elif dataset == "emnist":
        train_dataset = datasets.EMNIST(data_dir, train=True, download=True,
                                    transform=apply_transform, split="balanced")

        test_dataset = datasets.EMNIST(data_dir, train=False, download=True,
                                        transform=apply_transform, split="balanced")
    else:
        raise NotImplementedError
    return train_dataset, test_dataset
    
def load_mnist_data(dataset, data_dir):
    train_dataset, test_dataset = load_mnist_dataset(dataset, data_dir)
    
    X_train, y_train = train_dataset.data, train_dataset.targets
    X_test, y_test = test_dataset.data, test_dataset.targets

    return (X_train, y_train, X_test, y_test)


# for centralized training
def get_dataloader(dataset, data_dir, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    train_dataset, test_dataset = load_mnist_dataset(dataset, data_dir)
    
    if train_dataidxs is not None:
        train_dataset = DatasetSplit(train_dataset, train_dataidxs)
        test_dataset = DatasetSplit(test_dataset, test_dataidxs)
    train_dl = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False, drop_last=True)
    return train_dl, test_dl


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_mnist_data(dataset, datadir)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    if partition == "homo":
        train_net_dataidx_map = homo_partition(n_train, n_nets)
        test_net_dataidx_map = homo_partition(n_test, n_nets)
        
    elif partition == "p-hetero":
        train_net_dataidx_map = p_hetero_partition(n_nets, y_train, alpha)
        test_net_dataidx_map = p_hetero_partition(n_nets, y_test, alpha)

    traindata_cls_counts = record_net_data_stats(
        y_train, train_net_dataidx_map, "Train")
    testdata_cls_counts = record_net_data_stats(
        y_test, test_net_dataidx_map, "Test")
    
    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map, traindata_cls_counts, testdata_cls_counts




def load_partition_data_mnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, \
    train_net_dataidx_map, test_net_dataidx_map, \
    traindata_cls_counts, testdata_cls_counts = partition_data(dataset,
                                                            data_dir,
                                                            partition_method,
                                                            client_number,
                                                            partition_alpha)
    class_num = len(np.unique(y_train))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    # logging.info("testdata_cls_counts = " + str(testdata_cls_counts))
    train_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        train_dataidxs = train_net_dataidx_map[client_idx]
        train_local_data_num = len(train_dataidxs)
        test_dataidxs = test_net_dataidx_map[client_idx]
        test_local_data_num = len(test_dataidxs)
        data_local_num_dict[client_idx] = train_local_data_num
        
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size,
            train_dataidxs, test_dataidxs)
        logging.info(f"client_idx = {client_idx}, train sample = {train_local_data_num}, test sample = {test_local_data_num}, "
            +f"batch_num_train_local = {len(train_data_local)}, batch_num_test_local = {len(test_data_local)}")

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
