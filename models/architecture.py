from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import _pair
from models.cells import ConvRNNCell, ConvGruCell, ConvLSTMCell, ReciprocalGatedCell

KernelArg = Union[int, Sequence[int]]


class FeedForwardTower(torch.nn.Module):
    def __init__(self,
                 cell_type='conv',
                 num_classes=1000,
                 cell_kernel=3,
                 classifier_head=False):
        super().__init__()
        self.cell_type = cell_type
        self.num_classes = num_classes
        self.classifier_head=classifier_head
        self.cell_kernel=cell_kernel

        if self.cell_type == 'conv':
            def get_cell(*args,**kwargs):
                return torch.nn.Conv2d(*args,**kwargs)
        elif self.cell_type == 'gru':
            def get_cell(*args,**kwargs):
                return ConvGruCell(*args,**kwargs)
        elif self.cell_type == 'rnn':
            def get_cell(*args,**kwargs):
                return ConvRNNCell(*args,**kwargs)
        elif self.cell_type == 'lstm':
            def get_cell(*args,**kwargs):
                return ConvLSTMCell(*args,**kwargs)
        elif self.cell_type == 'reciprocal':
            def get_cell(*args,**kwargs):
                return ReciprocalGatedCell(*args,**kwargs)
        else:
            raise ValueError('Unknown ConvRNN cell type')

        self.conv1 = nn.Conv2d(3,64,7,2)
        self.cell1 = get_cell(64,64,self.cell_kernel)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.cell2 = get_cell(128, 128,self.cell_kernel)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.cell3 = get_cell(256, 256,self.cell_kernel)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1)
        self.cell4 = get_cell(256, 256,self.cell_kernel)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1)
        self.cell5 = get_cell(512, 512,self.cell_kernel)
        self.bn5 = nn.BatchNorm2d(512)

        if self.classifier_head:
            self.conv6 = nn.Conv2d(512, 512, 1, 2)
            self.classifier = nn.Linear(512,self.num_classes)
        else:
            self.conv6 = nn.Conv2d(512, self.num_classes, 1, 2)

        self.flatten = nn.Flatten()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.activation = F.relu


    def forward(self,x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.cell1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.cell2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.cell3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.pooling(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.cell4(x)
        x = self.activation(x)
        x = self.bn4(x)
        x = self.pooling(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.cell5(x)
        x = self.activation(x)
        x = self.bn5(x)
        x = self.pooling(x)

        if self.classifier_head:
            x = self.conv6(x)
            x = self.activation(x)
            x = self.flatten(x)
            x = self.classifier(x)
        else:
            x = self.conv6(x)
            x = self.flatten(x)

        x = F.softmax(x,dim=1)
        return x

