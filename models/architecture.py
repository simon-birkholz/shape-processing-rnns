from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.jit as jit
from .utils import _pair
from models.cells import ConvRNNCell, ConvGruCell, ConvLSTMCell, ReciprocalGatedCell
import serrelabmodels.gamanet

from models.cells import get_maybe_normalization, get_maybe_padded_conv
from serrelabmodels.layers import fgru_cell, hgru_cell

KernelArg = Union[int, Sequence[int]]
import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class SpatialDropout(nn.Module):

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.dropout = nn.Dropout2d(p=p)  # internally use the normal dropout2d layer

    def forward(self, x, mask=None):
        h = None
        if isinstance(x, tuple):
            x, h = x
        if mask is None:
            mask = torch.ones(*x.shape)
            mask = mask.to(x.get_device())
            mask = self.dropout(mask)

        x = mask * x
        if h is not None:
            h = mask * h
            return (x, h), mask
        return x, mask
    # When no mask supplied generate a new one


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel: KernelArg,
                 stride: KernelArg,
                 activation,
                 normalization):
        super(ConvBlock, self).__init__()
        self.conv = get_maybe_padded_conv(in_channels, out_channels, kernel, stride)
        self.norm = get_maybe_normalization(normalization, out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)

        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        return out


class ConvWrapper(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel: KernelArg,
                 stride: KernelArg,
                 activation,
                 normalization):
        super(ConvWrapper, self).__init__()

        self.conv = get_maybe_padded_conv(in_channels, out_channels, kernel, stride)
        self.norm = get_maybe_normalization(normalization, out_channels)
        self.activation = activation

    def forward(self, x, hx=None, t=0):
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
DEEPER_POOLS = [True, True, True, True, True]


class FeedForwardTower(torch.nn.Module):
    def __init__(self,
                 tower_type: str = 'normal',
                 cell_type: str = 'conv',
                 activation: str = 'relu',
                 num_classes: int = 1000,
                 cell_kernel: KernelArg = 3,
                 time_steps: int = 1,
                 normalization: str = 'batchnorm',
                 dropout: float = 0.0,
                 dropout_recurrent: bool = False,
                 skip_first: bool = False,
                 do_preconv: bool = True,
                 **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            print(f"Unknown Parameter {k}:{v}")
        self.cell_type = cell_type
        self.num_classes = num_classes
        self.cell_kernel = cell_kernel
        self.time_steps = time_steps
        self.skip_first = skip_first
        self.do_preconv = do_preconv
        self.dropout_recurrent = dropout_recurrent

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
                return ReciprocalGatedCell(*args, do_preconv=self.do_preconv, **kwargs)
        elif self.cell_type == 'hgru':
            def get_cell(ins, outs, ks, ss, ac, nn, **kwargs):
                return hgru_cell.hGRUCell(ins, outs, ks, nn, timesteps=time_steps, **kwargs)
        elif self.cell_type == 'fgru':
            def get_cell(*args, **kwargs):
                return fgru_cell.fGRUCell(*args, **kwargs)
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
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise ValueError('Unknown activation function')

        kernel_sizes = [7] + [3] * (len(filter_counts) - 2)
        stride_sizes = [2] + [1] * (len(filter_counts) - 2)
        print(kernel_sizes)

        if self.skip_first:
            # perhaps we want to conciously skip the first cell for a normal convolution
            self.cell_blocks = nn.ModuleList(
                [ConvWrapper(filter_counts[0], filter_counts[1], kernel_sizes[0], stride_sizes[0],
                             self.activation, normalization)] +
                [get_cell(ins, outs, ks, ss, self.activation, normalization) for (ins, outs), ks, ss in
                 zip(pairwise(filter_counts[1:]), kernel_sizes[1:], stride_sizes[1:])])
        else:
            self.cell_blocks = nn.ModuleList(
                [get_cell(ins, outs, ks, ss, self.activation, normalization) for (ins, outs), ks, ss in
                 zip(pairwise(filter_counts), kernel_sizes, stride_sizes)])

        self.last_conv = nn.Conv2d(filter_counts[-1], self.num_classes, 3, 1)

        self.flatten = nn.Flatten()

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.dropout = SpatialDropout(p=dropout)

        if self.cell_type in ['conv', 'rnn', 'gru', 'hgru']:
            self.get_x = lambda out: out
        elif self.cell_type in ['lstm', 'reciprocal']:
            def get_x(x):
                if isinstance(x, tuple):
                    return x[0]
                else:
                    return x

            self.get_x = get_x

        # do He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=activation)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input,
                return_hidden=False,
                time_steps=-1):
        if time_steps == -1:
            time_steps = self.time_steps

        x = input
        hidden = [None] * len(self.cell_blocks)
        dropout_mask = None
        for t in range(0, time_steps):
            x = input
            for i in range(len(self.cell_blocks)):
                # x = self.conv_blocks[i](x)
                x = self.cell_blocks[i](x, hidden[i], t)
                hidden[i] = x
                x = self.get_x(x)

                if self.do_pooling[i]:
                    x = self.pooling(x)

            if self.dropout_recurrent:
                hidden[-1], dropout_mask = self.dropout(hidden[-1], dropout_mask)

        if self.dropout_recurrent:
            x, _ = self.dropout(x, self.pooling(dropout_mask))
        else:
            x, _ = self.dropout(x, dropout_mask)
        x = self.last_conv(x)
        x = self.flatten(x)

        if return_hidden:
            return x, hidden

        return x


class GammaNetWrapper(torch.nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 time_steps: int = 1,
                 **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            print(f"Unknown Parameter {k}:{v}")
        self.time_steps = time_steps
        self.network = serrelabmodels.gamanet.BaseGN(timesteps=time_steps)
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.cls = nn.Linear(224 * 224, self.num_classes)

    def forward(self, input,
                return_hidden=False,
                time_steps=-1):
        hidden = None
        if time_steps == -1:
            time_steps = self.time_steps
        if return_hidden:
            x, hidden = self.network(input, return_hidden=True)
        else:
            x = self.network(input, return_hidden=False)

        x = self.flatten(x)
        x = self.cls(x)

        if return_hidden:
            return x, hidden
        else:
            return x
