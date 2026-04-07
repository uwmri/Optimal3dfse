"""
Usage: run
"""

import time
import torch
import numpy as np
import cupy as cp
import math
from pathlib import Path
import sigpy as sp
import sigpy.mri as mri
from sigpy.pytorch import to_pytorch, from_pytorch
from sigpy import backend
from sigpy.fourier import _scale_coord, _apodize, _get_oversamp_shape
import h5py
import os
import glob
import matplotlib.pyplot as plt
from math import ceil
from src.utils import pca_coil_compression
from src.model import ResBlock, BlockWiseCNN
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use the worst case as test
scan_dir = rf'S:\Opt3dfse_ADRC_add_on\CKD_09583_2025-03-31\09583_00012_Optimal_3DFLAIR\raw_data'

recon_type='train'
Nrep = 100
level = 5

# for Opt_FLAIR and Opt_FLAIR_c
hf = h5py.File(name=os.path.join(scan_dir, 'recon_denoiserTrue_phasefittingTrue_kwFalse_rh26.h5'), mode='r')
mag = hf[f'/{recon_type}_mag'][()]
phase = hf[f'/{recon_type}_ph'][()]

# for ADRC CUBE
# hf = h5py.File(name=os.path.join(scan_dir, 'recon_l1w_z216_44ch.h5'), mode='r')
# mag = hf[f'/Images_mag'][()]
# phase = hf[f'/Images_ph'][()]

im = mag * np.exp(1j * phase)

imN = []
for rep in range(Nrep):
    filename = f'recon_denoiserTrue_phasefittingTrue_kwFalse_rep{rep}_level{level}.h5'
    # filename = f'recon_l1w_z209_44ch_rep{rep}.h5'
    hf = h5py.File(name=os.path.join(scan_dir, filename), mode='r')
    imN.append(hf[f'/rep{rep}'][()])

imN = np.array(imN)
std_mag =  np.std(np.abs(imN), axis=0)
std_ph =  np.std(np.angle(imN), axis=0)
SNR  = np.abs(im) / std_mag

with h5py.File(os.path.join(scan_dir, f'std_{recon_type}_level{level}'), 'a') as hf:
    hf.create_dataset(f"std_rep{Nrep}_mag", data=std_mag)
    hf.create_dataset(f"std_rep{Nrep}_ph", data=std_ph)
    hf.create_dataset(f"SNR", data=SNR)


