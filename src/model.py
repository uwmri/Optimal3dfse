import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from utils_DL import ComplexConv3d, TrueComplexConv3d, ComplexReLU




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fmaps=32, kernel_size=3, padding=1, stride=1, bias=False, true_complex=False):
        super(ResBlock, self).__init__()
        if true_complex:
            self.conv1 = TrueComplexConv3d(in_channels, fmaps, kernel_size, padding=padding, bias=bias)
            self.conv2 = TrueComplexConv3d(fmaps, fmaps*2, kernel_size, padding=padding, stride=stride, bias=bias)
            self.conv3 = TrueComplexConv3d(fmaps*2, fmaps*2, kernel_size,padding=padding, bias=bias)
            self.conv4 = TrueComplexConv3d(fmaps*2, fmaps*2, kernel_size, padding=padding, bias=bias)
            self.conv8 = TrueComplexConv3d(fmaps*2, out_channels, kernel_size, padding=padding, bias=bias)

        else:
            self.conv0 = ComplexConv3d(in_channels, fmaps, 7, padding='same', bias=bias)
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

            return x

        return custom_forward


    def forward_nonblock(self, vol_input):
        out_final = checkpoint.checkpoint(self.run_function(), vol_input)
        return torch.squeeze(out_final) + torch.squeeze(vol_input)

    def forward(self, x, edge=20, blocks_per_dim=2):
            
            return self.forward_nonblock(x)
 

class BlockWiseCNN(nn.Module):
    def __init__(self, denoiser, patch_size=None, overlap=None, use_reentrant=False):
        super(BlockWiseCNN, self).__init__()
        self.denoiser = denoiser
        self.patch_size = patch_size
        self.overlap = overlap
        self.use_reentrant = use_reentrant

    def _get_starts(self, dim, patch):
        if dim <= patch:
            return [0]
        starts = list(range(0, dim, patch))
        if starts[-1] != dim - patch:
            starts[-1] = dim - patch
        return starts

    def run_function(self, x):
        return self.denoiser(x, mode='nonblock')

    def forward(self, x):

        out = torch.zeros_like(x)

        patch_in = [p + o for p, o in zip(self.patch_size, self.overlap)]
        half_overlap = [o // 2 for o in self.overlap]


        # reflection pad the volume so that patches near the borders have enough context
        pad = (
            half_overlap[2], half_overlap[2],
            half_overlap[1], half_overlap[1],
            half_overlap[0], half_overlap[0],
        )
        x_pad = F.pad(x, pad=pad, mode="reflect")



        dims = x.shape[-3:]
        starts = [self._get_starts(d, p) for d, p in zip(dims, self.patch_size)]

        for z in starts[0]:
            for y in starts[1]:
                for x0 in starts[2]:
                    patch = x_pad[
                            ...,
                            z: z + patch_in[0],
                            y: y + patch_in[1],
                            x0: x0 + patch_in[2],
                            ]
                    patch = checkpoint.checkpoint(
                        self.run_function, patch, use_reentrant=self.use_reentrant
                    )
                    out[
                    ...,
                    z: z + self.patch_size[0],
                    y: y + self.patch_size[1],
                    x0: x0 + self.patch_size[2],
                    ] = patch[
                        ...,
                        half_overlap[0]: half_overlap[0] + self.patch_size[0],
                        half_overlap[1]: half_overlap[1] + self.patch_size[1],
                        half_overlap[2]: half_overlap[2] + self.patch_size[2],
                        ]

        return out