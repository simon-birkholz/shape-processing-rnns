from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from models.cells import get_maybe_normalization, get_maybe_padded_conv

KernelArg = Union[int, Sequence[int]]
import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, activation, normalization):
        super(ConvBlock, self).__init__()
        self.conv = get_maybe_padded_conv(in_channels,out_channels,kernel,stride)
        self.norm = get_maybe_normalization(normalization,out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)

        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        return out

def get_dev_testing_architecture():

    a = F.relu
    n = "batchnorm"

    model = nn.Sequential(
        ConvBlock(3,64,7,2,a,n),
        nn.MaxPool2d(2),
        ConvBlock(64, 128, 7, 1, a, n),
        nn.MaxPool2d(2),
        ConvBlock(128, 256, 3, 1, a, n),
        nn.MaxPool2d(2),
        ConvBlock(256, 256, 3, 1, a, n),
        nn.MaxPool2d(2),
        ConvBlock(256, 512, 3, 1, a, n),
        nn.MaxPool2d(2),
        ConvBlock(512, 1024, 2, 1, a, n),
        nn.Flatten(),
        nn.Linear(3*3*1024,1000),
        #nn.Softmax(dim=1)
    )


    return model