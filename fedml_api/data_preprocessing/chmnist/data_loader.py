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

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    
    # for net_i, dataidx_i in net_dataidx_map.items():
    #     for net_j, dataidx_j in net_dataidx_map.items():
    #         if net_i == net_j:
    #             continue
    #         inter = np.intersect1d(dataidx_i, dataidx_j)
    #         print(net_i, net_j, len(inter))
    # st()

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def load_CH_MNIST():
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """

    # Initialize Data
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.8,
                                                        random_state=1,
                                                        stratify=labels.numpy())
    
    x_train = tf.image.resize(x_train, (32, 32))
    # y_train = tf.keras.utils.to_categorical(y_train-1, num_classes=8)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.image.resize(x_test, (32, 32))
    # y_test = tf.keras.utils.to_categorical(y_test-1, num_classes=8)
    m_test = np.zeros(y_test.shape[0])
    
    return x_train.numpy().astype(np.uint8), y_train, x_test.numpy().astype(np.uint8), y_test

class CHMNISTSplit(data.Dataset):

    def __init__(self, data, target, idxs=None, transform=None):

        self.data, self.target = data, target
        self.transform = transform
        if idxs is not None:
            self.idxs = idxs
        else:
            self.idxs = np.arange(len(data))
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):

        img, target = self.data[self.idxs[index]], self.target[self.idxs[index]]

        if self.transform is not None:
            img = self.transform(img)


        return img, target



# for centralized training
def get_dataloader(x_train, y_train, x_test, y_test, train_bs, test_bs, dataidxs=None):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    # train_dataset, test_dataset = load_mnist_dataset(dataset, data_dir)
    train_dataset = CHMNISTSplit(x_train, y_train, idxs=dataidxs, transform=train_transform)
    test_dataset = CHMNISTSplit(x_test, y_test, transform=test_transform)
    # train_dataset[0]
    # print(len(train_dataset))
    # st()
    train_dl = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False, drop_last=True)
    return train_dl, test_dl


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_CH_MNIST()
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    
    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "p-hetero":
        num_group = num_class = len(np.unique(y_train))
        client_per_group = int(n_nets / num_group)
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}
        
        idx_group = [[] for _ in range(num_group)]
        for k in range(num_class):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            
            split_idx = int(alpha*len(idx_k))
            dense_idxs = idx_k[:split_idx]
            sparse_idxs = idx_k[split_idx:]
            idx_group[k].append(dense_idxs)
            
            sparse_idxs = np.array_split(sparse_idxs, num_group-1)
            
            idx = 0
            for sparse_k in range(num_class):
                if k == sparse_k:
                    continue
                idx_group[sparse_k].append(sparse_idxs[idx])
                idx += 1
        for group in range(num_group):
            idx_group[group] = np.concatenate(idx_group[group])
            np.random.shuffle(idx_group[group])
        
        idx_batch = [[] for _ in range(n_nets)]
        if n_nets >= num_class:
            for group in range(num_group):
                group_split = np.array_split(idx_group[group], client_per_group)
                for batch in range(client_per_group):
                    idx_batch[group*client_per_group+batch] = group_split[batch]
        else:
            group_split = np.array_split(idx_group, n_nets)
            for i in range(n_nets):
                idx_batch[i] = np.concatenate(group_split[i])
        
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts




def load_partition_data_chmnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        X_train, y_train, X_test, y_test, 
        batch_size, batch_size
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)
    
    
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            X_train, y_train, X_test, y_test, 
            batch_size, batch_size,
            dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
