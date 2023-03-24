from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import json
import os

from typing import Tuple

mean = (0.5,0.5,0.5)
std = (0.5,0.5,0.5)

tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_imagenet_cifar10(path: str) -> Tuple[Dataset,Dataset]:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # TODO move into context
    imagenet = datasets.CIFAR10(root=path, transform=tfs, train=True, download=True)
    imagenet_val =  datasets.CIFAR10(root=path, transform=tfs, train=False, download=True)
    # imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet, imagenet_val

def get_imagenet_cifar100(path: str) -> Tuple[Dataset,Dataset]:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # TODO move into context
    imagenet = datasets.CIFAR100(root=path, transform=tfs, train=True, download=True)
    imagenet_val =  datasets.CIFAR100(root=path, transform=tfs, train=False, download=True)
    # imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet, imagenet_val