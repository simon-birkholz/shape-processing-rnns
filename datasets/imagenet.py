from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_imagenet(path: str) -> Dataset:
    tfs = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet = datasets.ImageNet(path, transform=tfs)
    # imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet  # , imagenet_val


def get_imagenet_small(path: str) -> Dataset:
    tfs = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    small_ds = datasets.ImageFolder(path, transform=tfs)
    return small_ds
