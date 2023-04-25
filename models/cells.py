from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.jit as jit


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
        self.x2h = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size,padding='same')

        self.h2h = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding='same')

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
        self.wr = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.ur = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')

        # update gate
        self.wz = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.uz = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')

        # state candidate
        self.can = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')


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


class ConvLSTMCell(torch.nn.Module):
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

        # reset/forget gate
        self.wf = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.uf = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.vf = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')

        # input gate
        self.wi = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.ui = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.vi = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')

        # output gate
        self.wo = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.uo = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.vo = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')

        # state candidate
        self.can = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')


    def forward(self,input ,hidden_state=None):
        # Inputs:
        # input: of shape (batch_size, input_size,height_size, width_size)
        # hx: of shape (batch_size, hidden_size,height_size, width_size)
        # Outputs:
        # hy: of shape (batch_size, hidden_size,height_size, width_size)

        if hidden_state is None:
            h_cur = Variable(input.new_zeros(input.size(0), self.out_channels, input.size(2), input.size(3)))
            c_cur = Variable(input.new_zeros(input.size(0), self.out_channels, input.size(2), input.size(3)))
        else:
            h_cur, c_cur = hidden_state

        # TODO important LSTM has the output gate on the new state candidate and with hadamard product

        forget_gate = F.sigmoid(self.wf(input) + self.uf(h_cur) + self.vf(c_cur))
        input_gate = F.sigmoid(self.wi(input) + self.ui(h_cur) + self.vi(c_cur))
        output_gate = F.sigmoid(self.wo(input) + self.uo(h_cur) + self.vo(c_cur))

        combined = torch.cat([input, h_cur],dim=1)

        candidate = F.tanh(self.can(combined))

        c_next = forget_gate * c_cur + input_gate * candidate

        # TODO support different activation functions
        h_next = output_gate * F.tanh(c_next)
        return h_next, c_next


class ReciprocalGatedCell(torch.nn.Module):
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

        # output gating
        self.wch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.whh = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')

        # memory gating
        self.whc = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')
        self.wcc = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same')

    def forward(self,input ,hidden_state=None):

        if hidden_state is None:
            h_cur = Variable(input.new_zeros(input.size(0), self.out_channels, input.size(2), input.size(3)))
            c_cur = Variable(input.new_zeros(input.size(0), self.out_channels, input.size(2), input.size(3)))
        else:
            h_cur, c_cur = hidden_state

        # TODO support different activation functions

        # output gating
        h_next = (1 - F.sigmoid(self.wch(c_cur))) * input + (1 - F.sigmoid(self.whh(h_cur))) * h_cur
        h_next = F.tanh(h_next)

        # memory gating
        c_next = (1 - F.sigmoid(self.whc(h_cur))) * input + (1 - F.sigmoid(self.wcc(c_cur))) * c_cur
        c_next = F.tanh(c_next)

        return h_next, c_next