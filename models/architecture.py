from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from .utils import _pair
from models.cells import ConvRNNCell, ConvGruCell, ConvLSTMCell, ReciprocalGatedCell

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
        if stride > 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding='same')
        if normalization == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


NORMAL_FILTERS = [64, 128, 256, 256, 512]
WIDER_FILTERS = [128, 512, 512, 512, 1024]
DEEPER_FILTERS = [64, 64, 64, 128, 128, 256, 256, 512, 512, 512]


class AxuiliaryClassifier(torch.nn.Module):

    def __init__(self, in_channels, out_classes, activation, normalization):
        super().__init__()
        self.model = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                   ConvBlock(in_channels, 128, 1, 1, activation, normalization), nn.Flatten(),
                                   nn.Linear(1152, 1024), nn.Linear(1024, 1000), nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)


class FeedForwardTower(torch.nn.Module):
    def __init__(self,
                 tower_type='normal',
                 cell_type='conv',
                 activation='relu',
                 num_classes=1000,
                 cell_kernel=3,
                 auxiliary_classifier=False):
        super().__init__()
        self.cell_type = cell_type
        self.num_classes = num_classes
        self.cell_kernel = cell_kernel
        self.auxiliary_classifier = auxiliary_classifier

        if self.cell_type == 'conv':
            def get_cell(*args, **kwargs):
                return torch.nn.Conv2d(*args, **kwargs, padding='same')
        elif self.cell_type == 'gru':
            def get_cell(*args, **kwargs):
                return ConvGruCell(*args, **kwargs)
        elif self.cell_type == 'rnn':
            def get_cell(*args, **kwargs):
                return ConvRNNCell(*args, **kwargs)
        elif self.cell_type == 'lstm':
            def get_cell(*args, **kwargs):
                return ConvLSTMCell(*args, **kwargs)
        elif self.cell_type == 'reciprocal':
            def get_cell(*args, **kwargs):
                return ReciprocalGatedCell(*args, **kwargs)
        else:
            raise ValueError('Unknown ConvRNN cell type')

        if tower_type == 'normal':
            filter_counts = [3] + NORMAL_FILTERS
        elif tower_type == 'wider':
            filter_counts = [3] + WIDER_FILTERS
        elif tower_type == 'deeper':
            filter_counts = [3] + DEEPER_FILTERS
        else:
            raise ValueError('Unknown Tower type')

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = functools.partial(F.elu, alpha=1.0)
        else:
            raise ValueError('Unknown activation function')

        kernel_sizes = [7] + [3] * (len(filter_counts) - 2)
        stride_sizes = [2] + [1] * (len(filter_counts) - 2)
        print(kernel_sizes)

        self.conv_blocks = nn.ModuleList(
            [ConvBlock(ins, outs, ks, ss, self.activation, 'batchnorm') for (ins, outs), ks, ss in
             zip(pairwise(filter_counts), kernel_sizes, stride_sizes)])

        self.last_conv = nn.Conv2d(filter_counts[-1], self.num_classes, 2, 1, padding='same')

        self.flatten = nn.Flatten()
        self.pooling = nn.MaxPool2d(kernel_size=2)

        #if self.auxiliary_classifier:
        #    self.layer_two_thirds = int(len(filter_counts) * (2/3)) -1
        #    self.aux_cls = AxuiliaryClassifier(filter_counts[self.layer_two_thirds],num_classes,self.activation,'batchnorm')



    def forward(self, x):

        for idx, block in enumerate(self.conv_blocks):
            x = block(x)
            x = self.pooling(x)
            #if self.auxiliary_classifier and self.layer_two_thirds == idx:
            #    aux_input = x
        x = self.last_conv(x)
        x = self.flatten(x)

        x = F.softmax(x, dim=1)

        #if self.auxiliary_classifier:
        #    aux_output = self.aux_cls(aux_input)
        #    return x, aux_output

        return x
