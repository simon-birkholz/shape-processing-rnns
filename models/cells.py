from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.jit as jit

from .utils import _pair, calculate_output_dimension

KernelArg = Union[int, Sequence[int]]


def get_maybe_padded_conv(in_channels: int, out_channels: int, kernel_size: KernelArg, stride: int):
    if stride > 1:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride,
                         padding='same')


class ConvRNNCell(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: KernelArg,
                 stride,
                 activation,
                 normalization,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.normalization = normalization
        self.ndim = 2

        ntuple = _pair

        # self.kernel_size = ntuple(kernel_size)
        # self.stride = ntuple(stride)
        # self.dilation = ntuple(dilation)

        # TODO there are multiple options for combining the hidden state and the input state
        # 1. Concat beforehand and then convolution
        # 2. Elementwise addition and then convolution (Hidden State and Input state would need the same dimensions
        # 3. Two convolution and then elementwise addition
        # In our formulas option 3 is used for all gates

        # todo padding und bias
        self.x2h = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride)

        self.h2h = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding='same')

    def forward(self, input, hx=None):
        # Inputs:
        # input: of shape (batch_size, input_size,height_size, width_size)
        # hx: of shape (batch_size, hidden_size,height_size, width_size)
        # Outputs:
        # hy: of shape (batch_size, hidden_size,height_size, width_size)

        if hx is None:
            if self.stride == 1:
                os1 = input.size(2)
                os2 = input.size(3)
            else:
                os1 = calculate_output_dimension(input.size(2), self.kernel_size, 0, self.stride)
                os2 = calculate_output_dimension(input.size(3), self.kernel_size, 0, self.stride)
            hx = Variable(input.new_zeros(input.size(0), self.out_channels, os1, os2))
        hy = (self.x2h(input) + self.h2h(hx))
        # TODO support different activation functions
        hy = self.activation(hy)
        return hy


class ConvGruCell(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: KernelArg,
                 stride,
                 activation,
                 normalization,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.nonlinearity = activation
        self.normalization = normalization

        # reset gate
        self.wr = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride)
        self.ur = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')

        # update gate
        self.wz = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride)
        self.uz = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')

        # state candidate
        self.wcan = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride)
        self.ucan = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding='same')

        # self.can = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size,
        #                     padding='same')

    def forward(self, input, hx=None):
        # Inputs:
        # input: of shape (batch_size, input_size,height_size, width_size)
        # hx: of shape (batch_size, hidden_size,height_size, width_size)
        # Outputs:
        # hy: of shape (batch_size, hidden_size,height_size, width_size)

        if hx is None:
            if self.stride == 1:
                os1 = input.size(2)
                os2 = input.size(3)
            else:
                os1 = calculate_output_dimension(input.size(2), self.kernel_size, 0, self.stride)
                os2 = calculate_output_dimension(input.size(3), self.kernel_size, 0, self.stride)

            hx = Variable(input.new_zeros(input.size(0), self.out_channels, os1, os2))

        reset_gate = F.sigmoid(self.wr(input) + self.ur(hx))
        update_gate = F.sigmoid(self.wz(input) + self.uz(hx))

        recalled = reset_gate * hx
        # combined = torch.cat([input, recalled], dim=1)

        # we now have to add elementwise
        hy = self.wcan(input) + self.ucan(recalled)

        # TODO support different activation functions
        hy = F.relu(hy)

        hy = (1 * update_gate) * hx + update_gate * hy
        return hy


class ConvLSTMCell(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: KernelArg,
                 stride,
                 activation,
                 normalization,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalization = normalization
        self.bias = bias
        self.nonlinearity = activation

        # reset/forget gate
        self.wf = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride)
        self.uf = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')
        self.vf = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')

        # input gate
        self.wi = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride)
        self.ui = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')
        self.vi = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')

        # output gate
        self.wo = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride)
        self.uo = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')
        self.vo = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            padding='same')

        # state candidate
        self.wcan = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride)
        self.ucan = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding='same')

    def forward(self, input, hidden_state=None):
        # Inputs:
        # input: of shape (batch_size, input_size,height_size, width_size)
        # hx: of shape (batch_size, hidden_size,height_size, width_size)
        # Outputs:
        # hy: of shape (batch_size, hidden_size,height_size, width_size)

        if hidden_state is None:
            if self.stride == 1:
                os1 = input.size(2)
                os2 = input.size(3)
            else:
                os1 = calculate_output_dimension(input.size(2), self.kernel_size, 0, self.stride)
                os2 = calculate_output_dimension(input.size(3), self.kernel_size, 0, self.stride)

            h_cur = Variable(input.new_zeros(input.size(0), self.out_channels, os1, os2))
            c_cur = Variable(input.new_zeros(input.size(0), self.out_channels, os1, os2))
        else:
            h_cur, c_cur = hidden_state

        # TODO important LSTM has the output gate on the new state candidate and with hadamard product

        forget_gate = F.sigmoid(self.wf(input) + self.uf(h_cur) + self.vf(c_cur))
        input_gate = F.sigmoid(self.wi(input) + self.ui(h_cur) + self.vi(c_cur))
        output_gate = F.sigmoid(self.wo(input) + self.uo(h_cur) + self.vo(c_cur))

        # combined = torch.cat([input, h_cur], dim=1)

        # we now have to add elementwise
        combined = self.wcan(input) + self.ucan(h_cur)
        candidate = F.tanh(combined)

        c_next = forget_gate * c_cur + input_gate * candidate

        # TODO support different activation functions
        h_next = output_gate * F.tanh(c_next)
        return h_next, c_next


class ReciprocalGatedCell(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: KernelArg,
                 stride,
                 activation,
                 normalization,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.nonlinearity = activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalization = normalization

        # employ a convolution before the reciprocal gated cell because the input gets gated by the hidden state, unlike all other cells

        self.preconv = get_maybe_padded_conv(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride)

        # output gating
        self.wch = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding='same')
        self.whh = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding='same')

        # memory gating
        self.whc = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding='same')
        self.wcc = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding='same')

    def forward(self, input, hidden_state=None):

        if hidden_state is None:
            if self.stride == 1:
                os1 = input.size(2)
                os2 = input.size(3)
            else:
                os1 = calculate_output_dimension(input.size(2), self.kernel_size, 0, self.stride)
                os2 = calculate_output_dimension(input.size(3), self.kernel_size, 0, self.stride)

            h_cur = Variable(input.new_zeros(input.size(0), self.out_channels, os1, os2))
            c_cur = Variable(input.new_zeros(input.size(0), self.out_channels, os1, os2))
        else:
            h_cur, c_cur = hidden_state

        # TODO support different activation functions
        x = self.preconv(input)

        # output gating
        h_next = (1 - F.sigmoid(self.wch(c_cur))) * x + (1 - F.sigmoid(self.whh(h_cur))) * h_cur
        h_next = F.tanh(h_next)

        # memory gating
        c_next = (1 - F.sigmoid(self.whc(h_cur))) * x + (1 - F.sigmoid(self.wcc(c_cur))) * c_cur
        c_next = F.tanh(c_next)

        return h_next, c_next
