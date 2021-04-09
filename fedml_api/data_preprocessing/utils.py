import logging
from pdb import set_trace as st
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def homo_partition(total_num, n_nets):
    idxs = np.random.permutation(total_num)
    batch_idxs = np.array_split(idxs, n_nets)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    return net_dataidx_map

def p_hetero_partition(n_nets, y_train, alpha):
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
    return net_dataidx_map

def record_net_data_stats(y_train, net_dataidx_map, tag=""):
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
    logging.debug(f'{tag} Data statistics: {str(net_cls_counts)}' )
    return net_cls_counts


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        # return torch.tensor(image), torch.tensor(label)
        return image, label