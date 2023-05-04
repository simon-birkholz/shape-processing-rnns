from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.jit as jit
from .utils import _pair
from models.cells import ConvRNNCell, ConvGruCell, ConvLSTMCell, ReciprocalGatedCell

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


class ConvWrapper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, activation, normalization):
        super(ConvWrapper, self).__init__()

        self.conv = get_maybe_padded_conv(in_channels, out_channels, kernel, stride)
        self.norm = get_maybe_normalization(normalization, out_channels)
        self.activation = activation

    def forward(self, x, hx=None):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        return out


NORMAL_FILTERS = [64, 128, 256, 256, 512]
NORMAL_POOLS = [True, True, True, True, True]

MORE_FILTERS = [64, 64, 128, 128, 256, 256, 512, 512]
MORE_POOLS = [True, False, True, False, True, False, True, True]
WIDER_FILTERS = [128, 512, 512, 512, 1024]
DEEPER_FILTERS = [64, 64, 64, 128, 128, 256, 256, 512, 512, 512]

class FeedForwardTower(torch.nn.Module):
    def __init__(self,
                 tower_type='normal',
                 cell_type='conv',
                 activation='relu',
                 num_classes=1000,
                 cell_kernel=7,
                 time_steps=1,
                 normalization='batchnorm',
                 auxiliary_classifier=False,
                 classifier_head=False,
                 **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            print(f"Unknown Parameter {k}:{v}")
        self.cell_type = cell_type
        self.num_classes = num_classes
        self.cell_kernel = cell_kernel
        self.auxiliary_classifier = auxiliary_classifier
        self.classifier_head = classifier_head
        self.time_steps = time_steps

        if self.cell_type == 'conv':
            self.time_steps = 1  # no unrollment necessary

            def get_cell(*args, **kwargs):
                return ConvWrapper(*args, **kwargs)
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
            self.do_pooling = NORMAL_POOLS
        elif tower_type == 'more':
            filter_counts = [3] + MORE_FILTERS
            self.do_pooling = MORE_POOLS
        elif tower_type == 'wider':
            filter_counts = [3] + WIDER_FILTERS
            self.do_pooling = NORMAL_POOLS
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

        # self.conv_blocks = nn.ModuleList(
        #    [ConvBlock(ins, outs, ks, ss, self.activation, normalization) for (ins, outs), ks, ss in
        #     zip(pairwise(filter_counts), kernel_sizes, stride_sizes)])

        self.cell_blocks = nn.ModuleList(
            [get_cell(ins, outs, ks, ss, self.activation, normalization) for (ins, outs), ks, ss in
             zip(pairwise(filter_counts), kernel_sizes, stride_sizes)])

        self.last_conv = nn.Conv2d(filter_counts[-1], self.num_classes, 2, 1, padding='same')

        self.flatten = nn.Flatten()

        if self.classifier_head:
            self.fc = nn.Linear(self.num_classes * 3 * 3,self.num_classes)
        self.pooling = nn.MaxPool2d(kernel_size=2)

        if self.cell_type in ['conv', 'rnn', 'gru']:
            self.get_x = lambda out: out
        elif self.cell_type in ['lstm', 'reciprocal']:
            self.get_x = lambda out: out[0]

    def forward(self, input):
        x = input
        hidden = [None] * len(self.cell_blocks)
        for t in range(0, self.time_steps):
            x = input
            for i in range(len(self.cell_blocks)):
                # x = self.conv_blocks[i](x)
                x = self.cell_blocks[i](x, hidden[i])
                hidden[i] = x
                x = self.get_x(x)

                if self.do_pooling[i]:
                    x = self.pooling(x)

        x = self.last_conv(x)
        x = self.flatten(x)

        if self.classifier_head:
            x = self.fc(x)

        x = F.softmax(x, dim=1)

        return x
