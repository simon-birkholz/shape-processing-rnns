from typing import Union, Sequence

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import _pair

KernelArg = Union[int, Sequence[int]]

class ConvRNNCell(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: KernelArg,
                 bias: bool=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.nonlinearity = 'relu'

        self.ndim = 2

        ntuple = _pair

        self.kernel_size = ntuple(kernel_size)
        #self.stride = ntuple(stride)
        #self.dilation = ntuple(dilation)

        #TODO there are multiple options for combining the hidden state and the input state
        # 1. Concat beforehand and then convolution
        # 2. Elementwise addition and then convolution (Hidden State and Input state would need the same dimensions
        # 3. Two convolution and then elementwise addition
        # In our formulas option 3 is used for all gates

        # todo padding und bias
        self.x2h = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size)

        self.h2h = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size)

    def forward(self,input ,hx=None):
        # Inputs:
        # input: of shape (batch_size, input_size,height_size, width_size)
        # hx: of shape (batch_size, hidden_size,height_size, width_size)
        # Outputs:
        # hy: of shape (batch_size, hidden_size,height_size, width_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.out_channels, input.size(2), input.size(3)))
        hy = (self.x2h(input) + self.h2h(hx))
        # TODO support different activation functions
        hy = F.relu(hy)
        return hy

class ConvGruCell(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: KernelArg,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.nonlinearity = 'relu'

        # reset gate
        self.wr = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size)
        self.ur = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        # update gate
        self.wz = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size)
        self.uz = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        # state candidate
        self.can = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self,input ,hx=None):
        # Inputs:
        # input: of shape (batch_size, input_size,height_size, width_size)
        # hx: of shape (batch_size, hidden_size,height_size, width_size)
        # Outputs:
        # hy: of shape (batch_size, hidden_size,height_size, width_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.out_channels, input.size(2), input.size(3)))

        reset_gate = F.sigmoid(self.wr(input) + self.ur(hx))
        update_gate = F.sigmoid(self.wz(input) + self.uz(hx))

        recalled = reset_gate * hx
        combined = torch.cat([input,recalled], dim=1)
        
        hy = self.can(combined)
        # TODO support different activation functions
        hy = F.relu(hy)

        hy = (1 * update_gate) * hx + update_gate * hy
        return hy