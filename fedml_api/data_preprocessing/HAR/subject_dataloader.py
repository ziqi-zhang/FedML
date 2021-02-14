import logging
from pdb import set_trace as st
import numpy as np
import os
import os.path as osp
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

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

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded

# load a single file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


class HumanActivityRecognition(Dataset):
    """Human activity recognition dataset"""
    data_size = (9, 128)
    n_classes = 6
    classes = ['walking',
               'walking upstairs',
               'walking downstairs',
               'sitting',
               'standing',
               'laying']

    def __init__(self, root,
                 is_train=True,
                 is_standardized=False):
        """
        Parameters
        ----------

        root : string
            Path to the csv file with annotations.
        is_train : bool
            Chooses train or test set
        is_standardized : bool
            Chooses whether data is standardized
        """
        if is_train:
            image_set = 'train'
        else:
            image_set = 'test'

        data_train = self.load_dataset(root, 'train')
        if is_standardized and image_set == 'train':
            print("Loading Human Activity Recognition train dataset ...")
            X = self.standardize_data(data_train[0])
            self.X = torch.from_numpy(X).permute(0,2,1).float()
            self.Y = torch.from_numpy(data_train[1]).flatten().long()
        elif is_standardized and image_set == 'test':
            print("Loading Human Activity Recognition test dataset ...")
            data_test = self.load_dataset(root, 'test')
            X =  self.standardize_data(data_train[0], data_test[0])
            self.X = torch.from_numpy(X).permute(0, 2, 1).float()
            self.Y = torch.from_numpy(data_test[1]).flatten().long()
        else:
            print("Loading Human Activity Recognition %s dataset ..." % image_set)
            data = self.load_dataset(root, image_set)
            self.X = torch.from_numpy(data[0]).permute(0, 2, 1).float()
            self.Y = torch.from_numpy(data[1]).flatten().long()
            self.subject = torch.from_numpy(data[2]).flatten().long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.X[idx,:,:]
        target = self.Y[idx]

        return input, target

    # load a dataset group, such as train or test
    # borrowed methods from the tutorial
    def load_dataset_group(self, group, prefix=''):
        # filepath = prefix + group + '/Inertial Signals/'
        filepath = os.path.join(prefix, group, 'Inertial Signals/')
        
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
        # body acceleration
        filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
        # body gyroscope
        filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
        # load input data
        X = load_group(filenames, filepath)
        
        # load class output
        path = osp.join(prefix, group, 'y_' + group + '.txt')
        Y = load_file(path)
        
        path = osp.join(prefix, group, 'subject_'+group+'.txt')
        subject = load_file(path)
        return X, Y, subject

    def split_data(self, data, image_set):
        l = int(0.75*len(data))
        if image_set == 'train':
            train = data[:l]
            return train
        elif image_set == "test":
            return data[l:]
            
    # load the dataset, returns train and test X and y elements
    def load_dataset(self, root='', image_set='train'):
        # load all train
        train_X, train_Y, train_s = self.load_dataset_group('train', root)
        test_X, test_Y, test_s = self.load_dataset_group('test', root)
        X = np.concatenate([train_X, test_X], axis=0)
        Y = np.concatenate([train_Y, test_Y], axis=0)
        subject = np.concatenate([train_s, test_s], axis=0)
        
        # zero-offset class values
        Y = Y - 1
        subject = subject - 1
        
        new_X, new_Y, new_s = [], [], []
        for s in np.unique(subject):
            idx = np.where(subject == s)[0]
            # train_idx = self.split_data(idx, "train")
            # test_idx = self.split_data(idx, "test")
            # print(len(train_idx), len(test_idx))
            idx = self.split_data(idx, image_set)
            new_X.append(X[idx])
            new_Y.append(Y[idx])
            new_s.append(subject[idx])
        new_X = np.concatenate(new_X)
        new_Y = np.concatenate(new_Y)
        new_s = np.concatenate(new_s)
        
        
        
        return new_X, new_Y, new_s

    # standardize data
    def standardize_data(self, X_train, X_test=None):
        """
        Standardizes the dataset

        If X_train is only passed, returns standardized X_train

        If X_train and X_test are passed, returns standardized X_test
        -------
        """
        # raise Exception("need to standardize the test set with the mean and stddev of the train set!!!!!!!")
        # remove overlap
        cut = int(X_train.shape[1] / 2)
        longX_train = X_train[:, -cut:, :]
        # flatten windows
        longX_train = longX_train.reshape((longX_train.shape[0] * longX_train.shape[1], longX_train.shape[2]))
        # flatten train and test
        flatX_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))

        # standardize
        s = StandardScaler()
        # fit on training data
        s.fit(longX_train)
        # apply to training and test data
        if X_test is not None:
            print("Standardizing test set")
            flatX_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
            flatX_test = s.transform(flatX_test)
            flatX_test = flatX_test.reshape((X_test.shape))
            return flatX_test
        else:
            print("Standardizing train set")
            # reshape
            flatX_train = s.transform(flatX_train)
            flatX_train = flatX_train.reshape((X_train.shape))
            return flatX_train
        

def load_har_dataset(dataset, data_dir):

    train_dataset = HumanActivityRecognition(root=data_dir, is_train=True)

    test_dataset = HumanActivityRecognition(root=data_dir, is_train=False)

    return train_dataset, test_dataset
    
def load_har_data(dataset, data_dir):
    train_dataset, test_dataset = load_har_dataset(dataset, data_dir)
    
    X_train, y_train, s_train = train_dataset.X, train_dataset.Y, train_dataset.subject
    X_test, y_test, s_test = test_dataset.X, test_dataset.Y, test_dataset.subject
    
    return (X_train, y_train, s_train, X_test, y_test, s_test)


# for centralized training
def get_dataloader(dataset, data_dir, train_bs, test_bs, dataidxs=None):
    train_dataset, test_dataset = load_har_dataset(dataset, data_dir)
    
    if dataidxs is not None:
        train_dataset = DatasetSplit(train_dataset, dataidxs)
    train_dl = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False, drop_last=True)
    return train_dl, test_dl


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, s_train, X_test, y_test, s_test = load_har_data(dataset, datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    
    
    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "p-hetero":
        num_group = num_subject = len(np.unique(s_train))
        
        N = s_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}
        
        idx_group = [[] for _ in range(num_group)]
        for k in range(num_subject):
            idx_k = np.where(s_train == k)[0]
            np.random.shuffle(idx_k)
            
            split_idx = int(alpha*len(idx_k))
            dense_idxs = idx_k[:split_idx]
            sparse_idxs = idx_k[split_idx:]
            idx_group[k].append(dense_idxs)
            
            sparse_idxs = np.array_split(sparse_idxs, num_group-1)
            
            idx = 0
            for sparse_k in range(num_subject):
                if k == sparse_k:
                    continue
                idx_group[sparse_k].append(sparse_idxs[idx])
                idx += 1
        for group in range(num_group):
            idx_group[group] = np.concatenate(idx_group[group])
            np.random.shuffle(idx_group[group])
        
        idx_batch = [[] for _ in range(n_nets)]
        if n_nets >= num_subject:
            client_per_group = int(n_nets / num_group)
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
            
        
    

    traindata_cls_counts = record_net_data_stats(s_train, net_dataidx_map)
    
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts




def load_partition_data_ucihar_subject(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    assert len(net_dataidx_map) == client_number
    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        # train_data_local = DatasetSplit(train_data_global, dataidxs)
        # test_data_local = test_data_global
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
