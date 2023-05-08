import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import os
import numpy as np

import torchvision.transforms as tfs
import torchvision.transforms.functional as F

from utils import traverse_obj

import xmltodict


def white_out(mask):
    mask_array = np.array(mask).astype(np.uint8)
    mask_array[(mask_array != (0, 0, 0)).any(axis=-1)] = (255, 255, 255)
    return F.to_pil_image(mask_array)


def remove_border(img):
    target_color = (224, 224, 192)
    image_array = np.array(img).astype(np.uint8)
    image_array[(image_array == target_color).all(axis=-1)] = (0, 0, 0)
    return F.to_pil_image(image_array)


class PascalVoc(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.bndboxes = []
        self.segmasks = []
        self.targets = []
        self.transform = transform
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']

        with open(os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")) as f:
            files = [l.strip() for l in f]

        image_dir = os.path.join(root, "JPEGImages")
        annotations_dir = os.path.join(root, "Annotations")
        segmask_dir = os.path.join(root, "SegmentationClass")
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

            bbox = [int(bnbbox['xmin']), int(bnbbox['ymin']), int(bnbbox['xmax']) - int(bnbbox['xmin']),
                    int(bnbbox['ymax']) - int(bnbbox['ymin'])]
            # the last two fields are now width and height of the bounding box

            bbox = torch.tensor(bbox, dtype=torch.float32)
            self.bndboxes.append(bbox)

            target = self.classes.index(objects['name'])
            self.targets.append(target+1) # Zero class means not in pascal voc

            sample_path = os.path.join(image_dir, f'{file}.jpg')
            self.samples.append(sample_path)

            segmask_path = os.path.join(segmask_dir, f'{file}.png')
            self.segmasks.append(segmask_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        mask = Image.open(self.segmasks[idx]).convert("RGB")

        mask = white_out(remove_border(mask))

        bbox = self.bndboxes[idx]
        if self.transform:
            return self.transform((x, bbox, mask)), self.targets[idx]
        else:
            x = tfs.ToTensor()(x)
        return (x, bbox, mask), self.targets[idx]
