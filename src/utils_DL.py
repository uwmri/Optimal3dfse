import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np

class RunningAverage:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def reset(self):
        self.count = 0
        self.sum = 0

    def avg(self):
        return self.sum / self.count



class ComplexConv3d(nn.Module):
    """
        do conv on real and imag separately
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(ComplexConv3d, self).__init__()
        self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return self.conv_r(input.real) + 1j*self.conv_r(input.imag)


class TrueComplexConv3d(nn.Module):
    """
        do conv on real and imag separately
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(TrueComplexConv3d, self).__init__()
        self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_y = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        re = self.conv_x(input.real) - self.conv_y(input.imag)
        im = self.conv_x(input.imag) + self.conv_y(input.real)
        return re + 1j * im
    

class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.act = nn.ReLU(inplace=False)
    def forward(self, input):
        return self.act(input.real) + 1j*self.act(input.imag)