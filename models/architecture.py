from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import _pair

KernelArg = Union[int, Sequence[int]]


class FeedForwardTower(torch.nn.Module):
    def __init__(self,
                 cell_type='conv',
                 num_classes=1000):
        super().__init__()
        self.cell_type = cell_type
        self.num_classes = num_classes

        if self.cell_type == 'conv':
            def get_cell(*args,**kwargs):
                return torch.nn.Conv2d(*args,**kwargs)
        elif self.cell_type == 'gru':
            def get_cell(*args,**kwargs):
                return torch.nn.Conv2d(*args,**kwargs)
        elif self.cell_type == 'rnn':
            def get_cell(*args,**kwargs):
                return torch.nn.Conv2d(*args,**kwargs)
        elif self.cell_type == 'lstm':
            def get_cell(*args,**kwargs):
                return torch.nn.Conv2d(*args,**kwargs)
        elif self.cell_type == 'reciprocal':
            def get_cell(*args,**kwargs):
                return torch.nn.Conv2d(*args,**kwargs)
        else:
            raise ValueError('Unknown ConvRNN cell type')

        self.conv1 = nn.Conv2d(3,64,7,2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1)
        self.conv6 = nn.Conv2d(512, self.num_classes, 1, 2)
        self.flatten = nn.Flatten()

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.activation = F.relu


    def forward(self,x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = F.softmax(x,dim=1)
        return x

