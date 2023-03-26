
from datasets.imagenet import get_imagenet, get_imagenet_kaggle, get_imagenet_small
from datasets.cifar import get_imagenet_cifar10, get_imagenet_cifar100
from datasets.ffcv_utils import loader_ffcv_dataset
import torch.utils.data as data


def select_dataset(dataset: str, dataset_path: str, dataset_val_path: str, batch_size: int):
    ds, ds_val = None, None
    if dataset == 'imagenet':
        ds, ds_val = get_imagenet(dataset_path)
    elif dataset == 'imagenet_small':
        ds, ds_val = get_imagenet_small(dataset_path)
    elif dataset == 'imagenet_kaggle':
        ds, ds_val = get_imagenet_kaggle(dataset_path)
    elif dataset == 'ffcv':
        ds = loader_ffcv_dataset(dataset_path, batch_size)
        if dataset_val_path:
            ds_val = loader_ffcv_dataset(dataset_val_path, batch_size)
    elif dataset == 'cifar10':
        ds, ds_val = get_imagenet_cifar10(dataset_path)
    elif dataset == 'cifar100':
        ds, ds_val = get_imagenet_cifar100(dataset_path)
    else:
        raise ValueError('Unknown Dataset')

    if dataset == 'ffcv':
        num_classes = 1000
    else:
        num_classes = len(ds.classes)

    # train_data_loader, val_data_loader = None, None
    if dataset != 'ffcv':
        train_data_loader = data.DataLoader(ds, batch_size=batch_size)
        val_data_loader = data.DataLoader(ds_val, batch_size=batch_size) if ds_val else None
    else:
        train_data_loader = ds
        val_data_loader = ds_val

    return train_data_loader, val_data_loader, num_classes
