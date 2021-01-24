from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .custom import Custom_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'custom')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'custom':
        dataset = Custom_Dataset(root=data_path, normal_class=normal_class)
        # dataset = Custom_Dataset(root=train_test_data_path, apply_model_root=apply_model_data_path, normal_class=normal_class)

    return dataset
