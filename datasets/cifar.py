from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import json
import os
import argparse

from typing import Tuple
from datasets.ffcv_utils import convert_to_ffcv


mean = (0.5,0.5,0.5)
std = (0.5,0.5,0.5)

tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

tfs_ffcv = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

def get_imagenet_cifar10(path: str, ffcv: bool = False) -> Tuple[Dataset,Dataset]:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # TODO move into context
    imagenet = datasets.CIFAR10(root=path, transform=tfs, train=True, download=True)
    imagenet_val =  datasets.CIFAR10(root=path, transform=tfs, train=False, download=True)
    # imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet, imagenet_val

def get_imagenet_cifar100(path: str, ffcv: bool = False) -> Tuple[Dataset,Dataset]:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # TODO move into context
    imagenet = datasets.CIFAR100(root=path, transform=tfs if not ffcv else tfs_ffcv, train=True, download=True)
    imagenet_val =  datasets.CIFAR100(root=path, transform=tfs if not ffcv else tfs_ffcv, train=False, download=True)
    # imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet, imagenet_val

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Dataset to process')
    parser.add_argument('--path', type=str, help='Path to the dataset', required=True)

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        ds, ds_val = get_imagenet_cifar10(args.path,True)
    elif args.dataset == 'cifar100':
        ds, ds_val =  get_imagenet_cifar100(args.path, True)
    else:
        raise ValueError('Dataset not found')

    train_path_out = f'{args.path}_train.beton'
    val_path_out = f'{args.path}_val.beton'

    convert_to_ffcv(train_path_out, ds)
    convert_to_ffcv(val_path_out,ds_val)