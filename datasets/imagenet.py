

from torchvision import  datasets
from torch.utils.data import Dataset


def get_imagenet(path: str) -> Dataset:
    imagenet = datasets.ImageNet(path)
    return imagenet