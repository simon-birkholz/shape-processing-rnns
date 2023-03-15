

from torchvision import  datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_imagenet(path: str) -> Dataset:
    tfs = transforms.Compose([
            transforms.Resize((50,50)),
            transforms.ToTensor(),
    ])

    imagenet = datasets.ImageNet(path,transform=tfs)
    #imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet #, imagenet_val