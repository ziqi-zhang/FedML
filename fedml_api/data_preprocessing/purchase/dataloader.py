import logging
from pdb import set_trace as st
import os.path as osp
import numpy as np
import torch
import pickle
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

# from .datasets import CIFAR10_truncated
from ..utils import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

TRAIN_PER_CLIENT, TEST_PER_CLIENT = 1000, 4000
def load_purchase(dataset, datadir):
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    
    if dataset == "purchase100":
        data_path = osp.join(datadir, "purchase_100_not_normalized_features.p")
        label_path = osp.join(datadir, "purchase_100_not_normalized_labels.p")
        
    elif dataset == "texas100":
        data_path = osp.join(datadir, "texas_100_not_normalized_features.p")
        label_path = osp.join(datadir, "texas_100_not_normalized_labels.p")
        
    with open(data_path, "rb") as f:
            data = pickle.load(f)
    with open(label_path, "rb") as f:
            label = pickle.load(f)
            
            
    # PredAvg setting, training dataset and test dataset have 10k data
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.8,
                                                        random_state=1,
                                                        stratify=label)
    if dataset == "purchase100":
        x_train = x_train[:10000]
        y_train = y_train[:10000]
        x_test = x_test[:10000]
        y_test = y_test[:10000]
        ...
    elif dataset == "texas100":
        # texas has 50k train and 10k test, sample 10k for target model train
        x_train = x_train[:10000]
        y_train = y_train[:10000]
        x_test = x_test[:10000]
        y_test = y_test[:10000]
        ...
    
    # x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.5,
    #                                                     random_state=1,
    #                                                     stratify=label)
    # if dataset == "purchase100":
    #     x_train = x_train[:10000]
    #     y_train = y_train[:10000]
    #     x_test = x_test[:10000]
    #     y_test = y_test[:10000]
    #     ...
    # elif dataset == "texas100":
    #     # texas has 50k train and 10k test, sample 10k for target model train
    #     x_train = x_train[:10000]
    #     y_train = y_train[:10000]
    #     x_test = x_test[:10000]
    #     y_test = y_test[:10000]
    #     ...
        
    # len_train = min(len(x_train), 50000)
    # len_test = min(len(x_test), 50000)
    # if dataset == "purchase100":
    #     x_train = x_train[:len_train]
    #     y_train = y_train[:len_train]
    #     x_test = x_test[:len_test]
    #     y_test = y_test[:len_test]
        
    # elif dataset == "texas100":
    #     x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.5,
    #                                                     random_state=1,
    #                                                     stratify=label)
    #     len_train = min(len(x_train), 50000)
    #     len_test = min(len(x_test), 50000)
    #     # texas has 50k train and 10k test, sample 10k for target model train
    #     x_train = x_train[:len_train]
    #     y_train = y_train[:len_train]
    #     x_test = x_test[:len_test]
    #     y_test = y_test[:len_test]
    
    # x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.2,
    #                                                     random_state=1,
    #                                                     stratify=label)
    
    # n_train, n_test = TRAIN_PER_CLIENT*n_nets, TEST_PER_CLIENT*n_nets
    # if dataset == "purchase100":
    #     x_train = x_train[:10000]
    #     y_train = y_train[:10000]
    #     # x_test = x_train[:10000]
    #     # y_test = y_train[:10000]
    #     ...
    # elif dataset == "texas100":
    #     # texas has 50k train and 10k test, sample 10k for target model train
    #     # x_train = x_train[:10000]
    #     # y_train = y_train[:10000]
    #     x_train = x_train[:n_train]
    #     y_train = y_train[:n_train]
    #     x_test = x_test[:n_test]
    #     y_test = y_test[:n_test]
    # st()
    
    return x_train, y_train, x_test, y_test
    


class PurchaseSplit(data.Dataset):

    def __init__(self, data, target, idxs=None, transform=None):

        self.data, self.target = data, target
        self.transform = transform
        if idxs is not None:
            self.data, self.target = data[idxs], target[idxs]
        else:
            self.data, self.target = data, target
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        # img = torch.Tensor(img).unsqueeze(0)
        # target = torch.Tensor(target).unsqueeze(0)
        # print(img.shape)

        
        return img.astype(np.float32), target.astype(np.int64)



# for centralized training
def get_dataloader(
    x_train, y_train, x_test, y_test, train_bs, test_bs, 
    train_dataidxs=None, test_dataidxs=None,
    training_data_ratio=1,
):
    train_transform = transforms.Compose([

        transforms.ToTensor(),
        # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # train_dataset, test_dataset = load_mnist_dataset(dataset, data_dir)
    train_dataset = PurchaseSplit(x_train, y_train, idxs=train_dataidxs,)
    test_dataset = PurchaseSplit(x_test, y_test, idxs=test_dataidxs)
    # train_dataset[0]
    # print(len(train_dataset))
    # st()
    train_dl = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False, drop_last=False)
    return train_dl, test_dl


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_purchase(
        dataset, datadir
    )
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




def load_partition_data_purchase(
    dataset, data_dir, partition_method, partition_alpha, client_number, batch_size,
    training_data_ratio=1,
):
    if training_data_ratio != 1:
        raise NotImplementedError
    
    X_train, y_train, X_test, y_test, \
    train_net_dataidx_map, test_net_dataidx_map, \
    traindata_cls_counts, testdata_cls_counts = partition_data(
        dataset, data_dir, partition_method, client_number,
        partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        X_train, y_train, X_test, y_test, 
        batch_size, batch_size,
        training_data_ratio=training_data_ratio,
    )
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
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, train_local_data_num))
        
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            X_train, y_train, X_test, y_test, 
            batch_size, batch_size,
            train_dataidxs, test_dataidxs,
        )
        logging.info(f"client_idx = {client_idx}, train sample = {train_local_data_num}, test sample = {test_local_data_num}, "
            +f"batch_num_train_local = {len(train_data_local)}, batch_num_test_local = {len(test_data_local)}")

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def load_server_data_purchase(
    dataset, datadir, server_data_ratio, batch_size
):
    logging.info("*********partition server data***************")
    X_train, y_train, X_test, y_test = load_purchase(dataset, datadir)
    num_data = int(len(X_train) * server_data_ratio)
    sample_indices = np.random.choice(len(X_train), num_data)
    
    train_data, test_data = get_dataloader(
            X_train, y_train, X_test, y_test, 
            batch_size, batch_size,
            train_dataidxs=sample_indices)
    return train_data, test_data