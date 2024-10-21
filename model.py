import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from torch.fx.experimental.unification.utils import xfail

from utils_DL import ComplexConv3d, ComplexReLU

class ResBlock(nn.Module):
    """
    a residual connection with checkpointing that deals with a block of the image
    """
    def __init__(self, in_channels, out_channels, fmaps=16, kernel_size=3, padding=1, stride=1, bias=False):
        super(ResBlock, self).__init__()

        self.conv0 = ComplexConv3d(in_channels, fmaps, kernel_size=7, padding='same', bias=bias)
        self.conv1 = ComplexConv3d(in_channels, fmaps, kernel_size, padding=padding, bias=bias)
        self.conv2 = ComplexConv3d(fmaps, fmaps, kernel_size, padding=padding, stride=stride, bias=bias)
        self.conv8 = ComplexConv3d(fmaps, out_channels, kernel_size, padding=padding, bias=bias)
        self.activation = ComplexReLU()
    def run_function(self):
        def custom_forward(L):
            x = self.conv1(L)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.activation(x)
            x = self.conv8(x)
            torch.cuda.empty_cache()
            return  x
        return custom_forward

    def forward(self, vol_input):
        out = checkpoint.checkpoint(self.run_function(), vol_input)
        return  torch.squeeze(out) + torch.squeeze(vol_input)


class BlockWiseCNN(nn.Module):
    """
    Only supports dividing the volume in half in each dimension.
    Otherwise the overlap region would be different for the middle blocks than the edges.
    Args:
        cnn: the cnn that will be applied to each block
        patch_size: size of the block. a list length 3.
        overlap: The overlap should be larger than the receptive field of the CNN

    """
    def __init__(self, cnn: nn.Module, patch_size: list, overlap: list, offset: list, use_reentrant=False):
        super(BlockWiseCNN, self).__init__()
        self.cnn = cnn
        self.patch_size = patch_size
        self.overlap = overlap
        self.offset=offset
        self.use_reentrant = use_reentrant
    def run_function(self, x):
        return self.cnn(x)
    def forward(self, x):
        # Size of block fed to network
        N = [i + j for i, j in zip(self.patch_size, self.overlap)]
        Ns = [i + j for i, j in zip(self.patch_size, self.offset)]

        # Patch into 8 overlapping blocks, the blocks are larger than the patch size
        # such that they have enough overlap to cover the receptive field of the network.
        # The patches are checkpointed to save memory and stiched back together.
        out = torch.zeros_like(x)

        patch0 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], 0:N[1], 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], 0:Ns[1], 0:Ns[2]] = patch0[..., 0:Ns[0], 0:Ns[1], 0:Ns[2]]

        patch1 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], 0:N[1], -N[2]:], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], 0:Ns[1], -Ns[2]:] = patch1[..., 0:Ns[0], 0:Ns[1], -Ns[2]:]

        patch2 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], -N[1]:, 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], -Ns[1]:, 0:Ns[2]] = patch2[..., 0:Ns[0], -Ns[1]:, 0:Ns[2]]

        patch3 = checkpoint.checkpoint(self.run_function, x[..., 0:N[0], -N[1]:, -N[2]:], use_reentrant=self.use_reentrant)
        out[..., 0:Ns[0], -Ns[1]:, -Ns[2]:] = patch3[..., 0:Ns[0], -Ns[1]:, -Ns[2]:]

        patch4 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, 0:N[1], 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, 0:Ns[1], 0:Ns[2]] = patch4[..., -Ns[0]:, 0:Ns[1], 0:Ns[2]]

        patch5 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, 0:N[1], -N[2]:], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, 0:Ns[1], -Ns[2]:] = patch5[..., -Ns[0]:, 0:Ns[1], -Ns[2]:]

        patch6 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, -N[1]:, 0:N[2]], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, -Ns[1]:, 0:Ns[2]] = patch6[..., -Ns[0]:, -Ns[1]:, 0:Ns[2]]

        patch7 = checkpoint.checkpoint(self.run_function, x[..., -N[0]:, -N[1]:, -N[2]:], use_reentrant=self.use_reentrant)
        out[..., -Ns[0]:, -Ns[1]:, -Ns[2]:] = patch7[..., -Ns[0]:, -Ns[1]:, -Ns[2]:]

        return out



