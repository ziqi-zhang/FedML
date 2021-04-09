import logging
from pdb import set_trace as st
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

# from .datasets import CIFAR10_truncated
from ..utils import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def load_CH_MNIST():
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """

    # Initialize Data
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)
    # 5k in total

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.3,
                                                        random_state=1,
                                                        stratify=labels.numpy())
    
    x_train = tf.image.resize(x_train, (32, 32))
    # y_train = tf.keras.utils.to_categorical(y_train-1, num_classes=8)
    m_train = np.ones(y_train.shape[0])
    
    x_test = tf.image.resize(x_test, (32, 32))
    # y_test = tf.keras.utils.to_categorical(y_test-1, num_classes=8)
    m_test = np.zeros(y_test.shape[0])
    
    return x_train.numpy().astype(np.uint8), y_train, x_test.numpy().astype(np.uint8), y_test
    
    # x_train = np.repeat(x_train, 5, axis=0)
    # m_train = np.repeat(m_train, 5, axis=0)
    # x_test = np.repeat(x_test, 5, axis=0)
    # m_test = np.repeat(m_test, 5, axis=0)
    
    # return x_train.astype(np.uint8), y_train, x_test.astype(np.uint8), y_test

class CHMNISTSplit(data.Dataset):
    def __init__(self, data, target, idxs=None, transform=None):
        
        self.transform = transform
        if idxs is not None:
            
            self.data, self.target = data[idxs], target[idxs]
        else:
            self.data, self.target = data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# for centralized training
def get_dataloader(
    x_train, y_train, x_test, y_test, train_bs, test_bs, 
    dataidxs_train=None, dataidxs_test=None
):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    # train_dataset, test_dataset = load_mnist_dataset(dataset, data_dir)
    train_dataset = CHMNISTSplit(x_train, y_train, idxs=dataidxs_train, transform=train_transform)
    test_dataset = CHMNISTSplit(x_test, y_test, idxs=dataidxs_test, transform=test_transform)
    # train_dataset[0]
    # print(len(train_dataset))
    # st()
    train_dl = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False, drop_last=False)
    return train_dl, test_dl


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_CH_MNIST()
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




def load_partition_data_chmnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, \
    train_net_dataidx_map, test_net_dataidx_map, \
    traindata_cls_counts, testdata_cls_counts = partition_data(
        dataset, data_dir, partition_method, client_number,
        partition_alpha,)
    class_num = len(np.unique(y_train))
    train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = sum([len(test_net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        X_train, y_train, X_test, y_test, batch_size, batch_size
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
        
        train_data_local, test_data_local = get_dataloader(
            X_train, y_train, X_test, y_test, batch_size, batch_size,
            train_dataidxs, test_dataidxs)
        logging.info(f"client_idx = {client_idx}, train sample = {train_local_data_num}, test sample = {test_local_data_num}, "
            +f"batch_num_train_local = {len(train_data_local)}, batch_num_test_local = {len(test_data_local)}")

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        
    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def load_server_data_chmnist(
    dataset, datadir, server_data_ratio, batch_size
):
    logging.info("*********partition server data***************")
    X_train, y_train, X_test, y_test = load_CH_MNIST()
    num_data = int(len(X_train) * server_data_ratio)
    sample_indices = np.random.choice(len(X_train), num_data)
    train_data, test_data = get_dataloader(
            X_train, y_train, X_test, y_test, 
            batch_size, batch_size,
            dataidxs_train=sample_indices)
    return train_data, test_data