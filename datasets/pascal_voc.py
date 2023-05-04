import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import json
import torch
import os

from utils import traverse_obj

import xmltodict


class PascalVoc(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.bndboxes = []
        self.targets = []
        self.transform = transform
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']

        with open(os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")) as f:
            files = [l.strip() for l in f]

        image_dir = os.path.join(root, "JPEGImages")
        annotations_dir = os.path.join(root, "Annotations")
        for file in files:
            with open(os.path.join(annotations_dir, f"{file}.xml")) as f:
                annotations = xmltodict.parse(f.read())

            objects = traverse_obj(annotations, 'annotation', 'object')
            if isinstance(objects, list):
                # more than one object
                continue

            if objects['truncated'] == '1' or objects['difficult'] == '1':
                continue

            bnbbox = traverse_obj(objects, 'bndbox')

            bbox = [int(bnbbox['xmin']), int(bnbbox['ymin']), int(bnbbox['xmax']) - int(bnbbox['xmin']), int(bnbbox['ymax']) - int(bnbbox['ymin'])]
            # the last two fields are now width and height of the bounding box

            bbox = torch.tensor(bbox, dtype=torch.float32)
            self.bndboxes.append(bbox)

            target = self.classes.index(objects['name'])
            self.targets.append(target)

            sample_path = os.path.join(image_dir, f'{file}.jpg')
            self.samples.append(sample_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        bbox = self.bndboxes[idx]
        if self.transform:
            x,bbox = self.transform(x,bbox)
        else:
            x = transforms.ToTensor()(x)
        return (x,bbox), self.targets[idx]
