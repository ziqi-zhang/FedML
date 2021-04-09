from fedml_api.data_preprocessing.MNIST.data_loader import load_server_data_mnist
from fedml_api.data_preprocessing.purchase.dataloader import load_server_data_purchase
from fedml_api.data_preprocessing.chmnist.data_loader import load_server_data_chmnist
from fedml_api.data_preprocessing.cifar10.data_loader import load_server_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_server_data_cifar100

def load_server_data(args):
    dataset_name = args.dataset
    
    if dataset_name in ["mnist", "fmnist", "emnist"]:
        server_train_data, server_test_data = load_server_data_mnist(
            args.dataset, args.data_dir, args.server_data_ratio,
            args.batch_size
        )
    elif dataset_name in ["purchase100", "texas100"]:
        server_train_data, server_test_data = load_server_data_purchase(
            args.dataset, args.data_dir, args.server_data_ratio,
            args.batch_size
        )
    elif dataset_name == "chmnist":
        server_train_data, server_test_data = load_server_data_chmnist(
            args.dataset, args.data_dir, args.server_data_ratio,
            args.batch_size
        )
    elif dataset_name == "cifar10":
        server_train_data, server_test_data = load_server_data_cifar10(
            args.dataset, args.data_dir, args.server_data_ratio,
            args.batch_size
        )
    elif dataset_name == "cifar100":
        server_train_data, server_test_data = load_server_data_cifar100(
            args.dataset, args.data_dir, args.server_data_ratio,
            args.batch_size
        )
    else:
        raise NotImplementedError
    
    return server_train_data, server_test_data