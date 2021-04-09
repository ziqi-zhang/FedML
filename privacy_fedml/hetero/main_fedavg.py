import argparse
import logging
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import wandb
from pdb import set_trace as st

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.vgg import VGG
from fedml_api.model.cv.cnn import CNN_DropOut, CNNCifar
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from fedml_api.model.linear.dense_mlp import *

from fedml_api.data_preprocessing.UCIAdult.dataloader import load_partition_data_uciadult
from fedml_api.data_preprocessing.purchase.dataloader import load_partition_data_purchase
from fedml_api.data_preprocessing.chmnist.data_loader import load_partition_data_chmnist
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.data_preprocessing.HAR.data_loader import load_partition_data_ucihar
from fedml_api.data_preprocessing.HAR.subject_dataloader import load_partition_data_ucihar_subject
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.linear.har_cnn import HAR_CNN
from fedml_api.model.cv.resnet_gn import resnet18, resnet50
from fedml_api.model.cv.resnet_cifar import *

# from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
# from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from fedml_api.standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG

from privacy_fedml.fedavg_api import FedAvgAPI
from privacy_fedml.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from privacy_fedml.predavg_api import PredAvgAPI
from privacy_fedml.predweight_api import PredWeightAPI
from privacy_fedml.blockavg_api import BlockAvgAPI
from privacy_fedml.blockensemble_api import BlockEnsembleAPI
from privacy_fedml.two_model_trainer import TwoModelTrainer

from privacy_fedml.MI_attack.NN_attack import NNAttack
from privacy_fedml.MI_attack.Top3_attack import Top3Attack
from privacy_fedml.MI_attack.Loss_attack import LossAttack
# from privacy_fedml.MI_attack.MixGradient_attack import GradientAttack
from privacy_fedml.MI_attack.Gradient_attack import GradientAttack

from privacy_fedml.adv_attack.adv_attack import AdvAttack

from fedml_api.model.ensemble.cnn import AdaptiveCNN, build_large_cnn

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    parser.add_argument('--run_tag', type=str, default="test")
    parser.add_argument('--aggr', type=str, default='fedavg')
    parser.add_argument('--branch_num', type=int, default=10)
    parser.add_argument('--client_per_branch', type=int, default=1)
    parser.add_argument('--ensemble_method', type=str, default="predavg")
    parser.add_argument('--server_data_ratio', type=float, default=0.1)
    parser.add_argument('--server_epoch', type=int, default=1)
    parser.add_argument('--disable_server_train', default=False, action='store_true')
    parser.add_argument('--training_data_ratio', type=float, default=1)
    parser.add_argument('--avg_mode', type=str, default="none")
    parser.add_argument('--no_mi_attack', action='store_true')
    
    parser.add_argument('--feat_lmda', type=float, default=0)
    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name in ["mnist", "fmnist", "emnist"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size,
            training_data_ratio=args.training_data_ratio,
        )
        
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num
    elif dataset_name == "har":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ucihar(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size
        )
        
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num
    
    elif dataset_name == "har_subject":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ucihar_subject(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size
        )
        
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num
    elif dataset_name == "chmnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_chmnist(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size
        )
        
        args.client_num_in_total = client_num
    elif dataset_name == "adult":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_uciadult(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size
        )
        
        args.client_num_in_total = client_num
    elif dataset_name in ["purchase100", "texas100"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_purchase(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size
        )
        args.client_num_in_total = client_num
        
    elif dataset_name == "femnist":
        
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(
            args.dataset, args.data_dir, 
            client_num_in_total = args.client_num_in_total,
        )
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None, partition_alpha=None,
                                                 client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size,
            training_data_ratio=args.training_data_ratio)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset in ["mnist", "fmnist", "emnist"]:
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim, flatten=True)
    elif model_name == "cnn" and args.dataset in ["mnist", "fmnist", "emnist"]:
        if args.dataset in ["mnist", "fmnist"]:
            logging.info("CNN + MNIST")
            model = build_large_cnn(True)
        elif args.dataset == "emnist":
            logging.info("CNN + MNIST")
            model = build_large_cnn(only_digits=47)
    elif model_name == "cnn" and args.dataset in ["har", "har_subject"]:
        logging.info("CNN + HAR")
        model = HAR_CNN(data_size=(9, 128), n_classes=6)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "cnn" and args.dataset == "cifar10":
        logging.info("CNN + CIFAR10")
        model = CNNCifar()
    elif model_name == "purchasemlp":
        if args.dataset == "purchase100":
            model = PurchaseMLP(input_dim=600, n_classes=100)
    elif model_name == "texasmlp":
        if args.dataset == "texas100":
            model = TexasMLP(input_dim=6169, n_classes=100)
    elif model_name == 'lr' and args.dataset == "adult":
        model = LogisticRegression(105, 2, flatten=False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "vgg11":
        model = VGG("VGG11")
    elif model_name == "resnet20":
        if args.dataset == "cifar10":
            model = resnet20_cifar(num_classes=10)
        elif args.dataset == "cifar100":
            model = resnet20_cifar(num_classes=100)
        elif args.dataset == "chmnist":
            model = resnet20_cifar(num_classes=8)
    elif model_name == "resnet18_gn":
        if args.dataset == "cifar10":
            model = resnet18(num_classes=10)
        elif args.dataset == "cifar100":
            model = resnet18(num_classes=100)
        elif args.dataset == "chmnist":
            model = resnet18(num_classes=8)
    elif model_name == "resnet50":
        if args.dataset == "cifar10":
            model = resnet50(num_classes=10)
        elif args.dataset == "cifar100":
            model = resnet50(num_classes=100)
        elif args.dataset == "chmnist":
            model = resnet50(pretrained=True, num_classes=8)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    else:
        raise NotImplementedError
    return model


def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    elif args.aggr == "blockensemble":
        return TwoModelTrainer(model)
        # return MyModelTrainerCLS(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)

def load_server(args, dataset ,device, model_trainer, output_dim):
    if args.aggr == "fedavg":
        server = FedAvgAPI(dataset, device, args, model_trainer, output_dim)
    elif args.aggr == "predavg":
        server = PredAvgAPI(dataset, device, args, model_trainer, output_dim)
    elif args.aggr == "predweight":
        server = PredWeightAPI(dataset, device, args, model_trainer, output_dim)
    elif args.aggr == "blockavg":
        server = BlockAvgAPI(dataset, device, args, model_trainer, output_dim)
    elif args.aggr == "blockensemble":
        server = BlockEnsembleAPI(dataset, device, args, model_trainer, output_dim)
    else:
        raise NotImplementedError
    return server

if __name__ == "__main__":
    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()

    
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)
    # args.exp_name = (
    #     f"FedAVG[{args.run_tag}]{args.dataset}{args.training_data_ratio}-{args.model}-" + 
    #     f"{args.partition_method}{str(args.partition_alpha)}-" +
    #     f"[branch{args.branch_num}-per-{args.client_per_branch}clients]-" +
    #     f"[{args.aggr}-{args.ensemble_method}-{args.avg_mode}]-" +
    #     f"[server-data{args.server_data_ratio}-e{args.server_epoch}]-"+
    #     f"r{str(args.comm_round)}" +
    #     f"-e{str(args.epochs)}-lr{str(args.lr)}-bs{str(args.batch_size)}"
    # )
    args.exp_name = (
        f"FedAVG[{args.run_tag}]{args.dataset}-{args.training_data_ratio}-{args.model}-" + 
        f"{args.partition_method}{str(args.partition_alpha)}-" +
        f"[client{args.client_num_in_total}-branch{args.branch_num}]-" +
        f"[{args.aggr}-{args.ensemble_method}-{args.avg_mode}]-" +
        f"feat{str(args.feat_lmda)}-"
        f"r{str(args.comm_round)}-e{str(args.epochs)}-lr{str(args.lr)}-bs{str(args.batch_size)}"
    )

    dir_path = osp.join("results", args.run_tag, args.exp_name)
    run_tags = args.run_tag.split(',')
    args.save_dir = dir_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    wandb.init(
        project="fedml",
        dir=dir_path,
        tags=run_tags,
        name=args.exp_name,
        config=args
    )
    
    
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    # load data
    dataset = load_data(args, args.dataset)
    # dataset = [6]*8

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)
    
    server = load_server(args, dataset, device, model_trainer, output_dim=dataset[7])
    if not args.disable_server_train:
        server.train()
        server.save_branch_state()
    else:
        server.load_branch_state()
    
    # if not args.no_mi_attack:
    #     nn_attack = NNAttack(server, device, args, 0, 0)
    #     nn_attack.eval_attack()
        
    #     top3_attack = Top3Attack(server, device, args, 0, 0)
    #     top3_attack.eval_attack()
        
    #     loss_attack = LossAttack(server, device, args, 0, 0)
    #     loss_attack.eval_attack()
        
    #     gradient_attack = GradientAttack(server, device, args, 0, 0)
    #     gradient_attack.eval_attack()
    
    # adv = AdvAttack(server, device, args, 0, 0)
    