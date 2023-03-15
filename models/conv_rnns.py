from typing import Union, Sequence

import torch
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F


from .utils import _pair

KernelArg = Union[int, Sequence[int]]

class ConvRNNCell(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: KernelArg,
                 bias: bool=True,
                 ndim: int=2,
                 stride: KernelArg=1,
                 dilation: KernelArg=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.ndim = ndim

        ntuple = _pair

        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)
