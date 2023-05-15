from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Convert, Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder

import torchvision as tv

from torch.utils.data import Dataset
import torch

import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def convert_to_ffcv(write_path: str, ds: Dataset):
    writer = DatasetWriter(write_path, {'image': RGBImageField(
        max_resolution=256,
        jpeg_quality=100
    ), 'label': IntField()})
    writer.from_indexed_dataset(ds)


def loader_ffcv_dataset(path: str, batch_size: int):
    decoder = RandomResizedCropRGBImageDecoder((224, 224))
    # decoder = SimpleRGBImageDecoder()

    device = torch.device('cuda')

    # Data decoding and augmentation
    image_pipeline = [decoder, ToTensor(), ToDevice(device, non_blocking=True), ToTorchImage(), Convert(torch.float),
                      tv.transforms.Normalize(mean=mean, std=std), tv.transforms.RandomHorizontalFlip(p=0.5)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    loader = Loader(path, batch_size=batch_size, num_workers=8,
                    order=OrderOption.RANDOM, pipelines=pipelines,
                    os_cache=True, drop_last=True)
    return loader
