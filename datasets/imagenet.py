from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import json
import os

from typing import Tuple

# taken from https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]

def get_imagenet_kaggle(path: str) -> Tuple[Dataset,Dataset]:
    tfs = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet = ImageNetKaggle(path, transform=tfs, split='train')
    imagenet_val = ImageNetKaggle(path, transform=tfs, split='val')
    # imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet, imagenet_val

def get_imagenet(path: str) -> Tuple[Dataset,Dataset]:
    tfs = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet = datasets.ImageNet(path, transform=tfs, split='train')
    imagenet_val = datasets.ImageNet(path, transform=tfs, split='val')
    # imagenet_val = datasets.ImageNet(path, split='val')
    return imagenet , imagenet_val


def get_imagenet_small(path: str) -> Tuple[Dataset,Dataset]:
    tfs = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #val_to_label = {}
    #with open(os.path.join(path, "Labels.json"), "r") as f:
    #   val_to_label = json.load(f)

    #def to_label(intern: str) -> str:
    #    return val_to_label[intern]

    small_ds = datasets.ImageFolder(os.path.join(path, 'train'), transform=tfs)
    small_ds_val = datasets.ImageFolder(os.path.join(path, 'val'), transform=tfs)
    return small_ds, small_ds_val
