""" This script reconstruct prospective Optimal_3DFLAIR and Optimal_3DFLAIR_c scans with MoDL or l1w.
    Good for MoDLs trained later, e.g.1847, 6267. For older models compatibility, refer to WaveCAIPI repo.
"""

import time
import torch
import numpy as np
import cupy as cp
import math
import sigpy as sp
import sigpy.mri as mri
from sigpy.pytorch import to_pytorch, from_pytorch
from sigpy.fourier import _scale_coord, _apodize, _get_oversamp_shape
import h5py
import os
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re, sys
import logging

logger = logging.getLogger(__name__)

from utils import get_outer, run_n4
from model import ResBlock, BlockWiseCNN



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pseudo_replica = False

scan_dirs = sys.argv[1:] or [rf'S:\Opt3dfse_ADRC_add_on\CKD_09583_2025-03-31\09583_00012_Optimal_3DFLAIR\raw_data']

##################### Trained MoDL releated params ###########################
# starting 7/1 for ADRC add-on, w/coords is 1847_60, w/o coords is 6267_60
Ntrial = 6267
Nepoch = 60
num_unroll = 10
train_dir = rf'D:\SSD\Data\run_3dfse\{Ntrial}'
with open(os.path.join(train_dir, f'3dfse_{Ntrial}.log'), 'r') as _f:
    _log_text = _f.read()
_match = re.search(
    rf'Epoch {Nepoch} lossMSE.*\nINFO:root:scale_global=\[([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\]',
    _log_text
)
scale_global = torch.tensor([float(_match.group(1))]).cuda()
scaleks = np.ones(num_unroll)       # For 1847 and 6267, didn not use scaleks.


recon_type = 'l1w'
kw_enabled = False
if_denoiser = True
masked = False  # if theres a brain mask
phase_fitting = True
true_complex = False
bias_correction = True
num_views = 2720
fov = 250
xres = 256
yres = 256
zres = 128
sliceres = 1.6
zres_recon = int(zres * sliceres / (fov/xres) ) # recon to 0.976 iso resolution because imageJ can only display isotropic voxels
zres_zp = 256
etl = 170  # prescribed etl
disdaqs = 0 # as of 10/16, manually add in text file to have more disdaqs
num_trains = int(np.ceil(num_views/etl))

if pseudo_replica:
    Nrep = 100
    scale_noise = 5
else:
    Nrep = 1

# ── N4 parameters to tune ──────────────────────────────────────────────────────
n4_params = dict(
    num_histogram_bins    = 200,      # try 100–400
    fitting_levels        = 4,        # try 3–6; more levels = captures larger-scale bias
    max_iter_per_level    = 50,       # iterations at each resolution level
    convergence_threshold = 0.001,
    shrink_factor         = 4,        # shrink before masking (speeds up Otsu)
    n_passes              = 1,        # set to 2 to run N4 twice (iterative)
)

for (fi, scan_dir) in enumerate(scan_dirs):

    case_dir = os.path.dirname(os.path.dirname(scan_dir))
    truth_matches = glob.glob(os.path.join(case_dir, '*_Sag_T2_FLAIR_CUBE', 'raw_data'))
    truth_folder = truth_matches[0] if truth_matches else None

    if recon_type == 'train':
        fname = f'recon_denoiser{if_denoiser}_phasefitting{phase_fitting}_kw{kw_enabled}_pseudorep{pseudo_replica}_rh26.h5'
    else:
        fname = f'recon_{recon_type}_pseudorep{pseudo_replica}.h5'

    if fname in os.listdir(scan_dir):
        continue
    else:
        reconiterfile = os.path.join(scan_dir, fname)

        print(f'reconstructing {scan_dir}')
        print(f'clinical scan in {truth_folder}')


        # get rhusers from header.txt, which came from pfile_info
        with open(f'{scan_dir}/header.txt', 'r') as file:
            text = file.read()
        rhuser24 = float(re.search(r'rhuser24\s*=\s*([-+]?\d*\.?\d+)', text).group(1))
        rhuser25 = float(re.search(r'rhuser25\s*=\s*([-+]?\d*\.?\d+)', text).group(1))
        rhuser26 = float(re.search(r'rhuser26\s*=\s*([-+]?\d*\.?\d+)', text).group(1))
        logger.info(f'rhuser24 = {rhuser24}')
        logger.info(f'rhuser25 = {rhuser25}')
        logger.info(f'rhuser26 = {rhuser26}')

        oc_xshift = -rhuser24
        oc_yshift = -rhuser25
        oc_zshift = rhuser26   # Note it is rhuser26. Not -rhuser26.
        deltax = 2 * math.pi * oc_xshift / fov
        deltay = 2 * math.pi * oc_yshift / fov
        deltaz = 2 * math.pi * oc_zshift / fov

        # get the actual kyzc executed by the scanner.
        kwfile = glob.glob(os.path.join(scan_dir,  'kw_sort*.txt'))
        centersfile = glob.glob(os.path.join(scan_dir,  'kzc*.txt'))
        kyzc_raw = np.loadtxt(centersfile[0], delimiter=',')  # kzc, kyc

        indices = np.where(kyzc_raw[:, 0] == -129)[0]
        kyzc = np.delete(kyzc_raw, indices, axis=0)
        start_idx = etl * disdaqs
        kyzc = kyzc[start_idx:start_idx+num_views]         # after the view=optview+1 fix
        centers = np.zeros((num_views, 3), dtype=kyzc.dtype)
        centers[:, 1:] = kyzc  # x is readout
        # NOTE:
        # Special case Exam1024 and Exam1025, ky in the kzc_*.txt file was indexed wrong. See recon_3dfse in WaveCaipi for handling those.
        # After 2023-11-09, echo train sorting and echoes2skip are done in C code. no need to do the skip here.

        # # visualization
        # echo_indices = np.tile(np.arange(etl), num_trains) # after  the view=optview+1 fix (10/16/2023)
        # echo_indices = echo_indices[:num_views]
        # fig = plt.figure(figsize=(6,3))
        # plt.scatter(kyzc[:, 0], kyzc[:, 1],s=3, alpha=0.7,c=echo_indices)
        # plt.colorbar()
        # plt.show()

        # # # fig = plt.figure(figsize=(6,6))
        # # etl_i = 10
        # # plt.scatter(kyzc[etl*(etl_i-1):etl*etl_i, 0], kyzc[etl*(etl_i-1):etl*etl_i, 1],s=5, alpha=0.7,c=np.arange(etl))
        # # plt.plot(kyzc[etl*(etl_i-1):etl*etl_i, 0], kyzc[etl*(etl_i-1):etl*etl_i, 1], 'k--',linewidth=0.5)
        # # # plt.colorbar()
        # # plt.show()

        # fig = plt.figure(figsize=(6,3))
        # plt.scatter(kyzc[:, 0], kyzc[:, 1],s=5, alpha=0.7,c=echo_indices);plt.gca().set_aspect('equal')
        # plt.xlim(-30, 30)
        # plt.ylim(-30, 30)
        # plt.show()


        radius_scale = np.ones((num_views,),dtype=np.float32)
        theta = np.zeros((num_views,),dtype=np.float32)

        ##################### Load smaps ##################################
        # separate smaps scan
        # hfsmaps = h5py.File(name=os.path.join(scan_root, f'lowres_fov{fov}_sag_{zres_zp}_cc{coil_compression}.h5'), mode='r')
        # smaps = hfsmaps['lowres_coil'][()]

        # smaps from ADRC -> JSENSE
        hfsmaps = h5py.File(name=os.path.join(truth_folder, f'smaps_jsense_z{zres_zp}.h5'), mode='r')
        smaps = hfsmaps['Maps_real'][()] + 1j* hfsmaps['Maps_imag'][()]

        if masked:
            mask = hfsmaps['mask'][()]
            smaps *= mask

        # if smaps is from T2flair_preprocessing.py
        # so the orientation is consistent with training.
        # if recon_type == 'train':
        smaps = np.rot90(smaps, k=1, axes=(1,2))
        smaps = smaps[:,:,:,::-1]
        nchannels_s = smaps.shape[0]
        ####################################################################


        # get full coords. Borrowed code from wave
        helix = np.zeros((xres, 3), dtype=np.float32)
        helix[:,0] = np.linspace(-xres/2, xres/2, xres)

        helix_interp = helix
        coords = np.zeros(helix_interp.shape + (num_views,), dtype=np.float32)
        for v in range(num_views):
            rot = np.stack((1., 0., 0.,
                            0., np.cos(theta[v]), -np.sin(theta[v]),
                            0., np.sin(theta[v]), np.cos(theta[v])))
            rot = np.reshape(rot, (3, 3))
            scaler = np.stack((1., radius_scale[v], radius_scale[v]))
            helix_tmp = helix_interp * scaler
            coords[..., v] = np.matmul(helix_tmp, rot) + np.expand_dims(centers[v], axis=0)
        coords = np.transpose(coords, (0, -1, 1))
        coord3D = coords.copy()
        coord = np.reshape(coord3D, (-1, 3))
        coord = sp.to_device(coord, sp.Device(0))

        # load raw data. They came from running the matlab code
        hf = h5py.File(name=os.path.join(scan_dir, f'raw.h5'), mode='r')
        kdata_raw = hf['kdata_r'][()].astype(np.complex64) + 1j * hf['kdata_i'][()].astype(np.complex64)
        nchannels_k = kdata_raw.shape[0]
        kdata = kdata_raw[:, :, start_idx:start_idx+num_views]
        kdata_cc_shift = kdata * np.exp(1j * coord3D[:, :, 0] * deltax)
        kdata_cc_shift = kdata_cc_shift * np.exp(1j * coord3D[:, :, 1] * deltay)
        kdata_cc_shift = kdata_cc_shift * np.exp(1j * coord3D[:, :, 2] * deltaz)

        # # show raw prewhitened(in matlab) kdata
        # plot_xz(coord3D, kdata_cc_shift[np.random.randint(0, nchannels), ...], y_target=0, tolerance=0.5)
        # plot_xy(coord3D, kdata_cc_shift[np.random.randint(0, nchannels), ...], z_target=0, tolerance=0.5)
        # plot_PEs(coord3D, kdata_cc_shift[np.random.randint(0, nchannels), ...], readout=127)

        k = np.reshape(kdata_cc_shift, (nchannels_k, -1))

        nchannels = min(nchannels_s, nchannels_k)
        k = k[:nchannels, :]
        smaps = smaps[:nchannels, ...]


        if pseudo_replica:
            noise_only = []
            for cc in range(nchannels):
                noise_only.append(get_outer(coord, k[cc], percent=5))
            noise_only = np.asarray(noise_only)
            noise_only = noise_only - np.mean(noise_only, axis=1, keepdims=True)
            cov0 = (noise_only @ noise_only.conj().T) / (noise_only.shape[1] - 1)
            # plt.imshow(np.abs(cov0));
            # plt.show()

            # add noise
            diag_var = np.diag(cov0)
            std_dev = np.sqrt(diag_var / 2).reshape(nchannels, 1) * scale_noise
        else:
            std_dev = 0.

        for rep in range(Nrep):

            real_noise = np.astype(np.random.randn(nchannels, k.shape[1]), 'float32') * std_dev
            imag_noise = np.astype(np.random.randn(nchannels, k.shape[1]), 'float32') * std_dev

            k =  np.real(k) + real_noise + 1j * (np.imag(k) + imag_noise)

            k_gpu = sp.to_device(k, sp.Device(0))
            if recon_type == 'gridding':

                oversamp = 1.25
                width = 4
                beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
                oshape = [256, 256, 256]
                ndim = 3
                os_shape = _get_oversamp_shape(oshape, ndim, oversamp)
                coord = _scale_coord(coord, oshape, oversamp)

                imCoil = np.zeros((nchannels, 256, 256, 256), dtype=cp.complex64)
                # kcart = np.zeros((num_coils,)+tuple(os_shape), dtype=cp.complex64)

                for cc in range(nchannels):
                    tt = time.time()
                    input = np.reshape(kdata_cc_shift[cc], (-1,))
                    input = sp.to_device(input, sp.Device(0))

                    output = sp.interp.gridding(input, coord, os_shape,
                                                kernel='kaiser_bessel', width=width, param=beta)
                    output /= width ** ndim
                    # kcart[cc] = output.get()

                    output = sp.fourier.ifft(output, axes=range(-ndim, 0), norm=None)
                    output = sp.util.resize(output, oshape)
                    output *= sp.util.prod(os_shape[-ndim:]) / sp.util.prod(oshape[-ndim:]) ** 0.5

                    # Apodize
                    _apodize(output, ndim, oversamp, width, beta)
                    imCoil[cc] = output.get()
                    # plt.figure();
                    # plt.imshow(np.abs(imCoil[cc, :,158,:]),cmap='gray');plt.colorbar()
                    # plt.show()
                    #
                    # plt.figure();
                    # plt.imshow(np.abs(im[cc, 128,:,:]),cmap='gray');plt.colorbar()
                    # plt.show()
                    #
                    # plt.figure();
                    # plt.imshow(np.abs(im[cc, :,:,128]),cmap='gray')
                    # plt.show()

                im = np.sum(imCoil * np.conj(smaps), axis=0)

                try:
                    os.remove(reconiterfile)
                except OSError:
                    pass
                with h5py.File(reconiterfile, 'a') as hf:
                    hf.create_dataset(f"im_SOS", data=np.abs(im))
                    hf.create_dataset(f"im_all", data=np.abs(imCoil))

            if recon_type == 'sense':
                im = sp.mri.app.SenseRecon(k_gpu, mps=smaps, weights=None, coord=coord,
                                            device=sp.Device(0), lamda=1e-4, max_iter=15,
                                            coil_batch_size=1, save_objective_values=False).run()
                im = np.squeeze(im.get().astype(np.complex64))
            elif recon_type == 'l1w':
                im = sp.mri.app.L1WaveletRecon(k_gpu, mps=smaps, weights=None, coord=coord,
                                                device=sp.Device(0), lamda=1e-4, max_iter=15,
                                                coil_batch_size=1, save_objective_values=False).run()
                im = np.squeeze(im.get().astype(np.complex64))
            elif recon_type == 'pils':
                im = sp.mri.app.SenseRecon(k_gpu, mps=smaps, weights=None, coord=coord,
                                            device=sp.Device(0), lamda=1e-4, max_iter=1,
                                            coil_batch_size=1, save_objective_values=False).run()
                im = np.squeeze(im.get().astype(np.complex64))
            elif recon_type == 'train':
                if kw_enabled:
                    kw_raw = np.loadtxt(kwfile[0], delimiter=',')  # kzc, kyc
                    indices = np.where(kw_raw == -1)
                    kw = np.delete(kw_raw, indices)
                else:
                    kw = np.ones((num_views,))
                kw = kw.astype(np.float32)
                kw = np.tile(kw, (xres, 1))  # (xres, num_views)
                kw = kw.reshape((-1,))
                kw_gpu = sp.to_device(kw, sp.Device(0))

                if if_denoiser:
                    denoiser_file = os.path.join(train_dir, f'denoiser_{Ntrial}_{Nepoch}.pt')
                    state = torch.load(denoiser_file)
                    denoiser = ResBlock(1, 1, true_complex=true_complex)
                    patch_size = [128, 128, 128]
                    overlap = [20, 20, 20]
                    denoiser_bw = BlockWiseCNN(denoiser, patch_size, overlap).to(device)
                    denoiser_bw.load_state_dict(state['state_dict'], strict=True)
                    denoiser_bw.cuda();
                    denoiser_bw.eval()


                k_gpu_weighted = k_gpu * kw_gpu
                A = sp.mri.linop.Sense(smaps, coord=coord, weights=None, tseg=None,
                                        coil_batch_size=1, comm=None,
                                        transp_nufft=False)
                im = torch.zeros([xres, yres, zres_zp], dtype=torch.complex64, device=device)
                im.requires_grad = False

                for u in range(num_unroll):
                    Ex = A * from_pytorch(im)
                    scalek = np.astype(scaleks[u], 'float32')
                    Ex = Ex / scalek
                    Exd = Ex - k_gpu  # Ex-d
                    Exd = Exd * kw_gpu

                    " Data consistency step "
                    im = im - to_pytorch(A.H * Exd) * scale_global
                    im = im.detach()
                    torch.cuda.empty_cache()

                    if if_denoiser:
                        if u==0:
                            X, Y, Z = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing='ij')
                            X_torch = torch.tensor(X, dtype=torch.cfloat, device='cuda')
                            Y_torch = torch.tensor(Y, dtype=torch.cfloat, device='cuda')
                            Z_torch = torch.tensor(Z, dtype=torch.cfloat, device='cuda')
                            # phase fitting
                            if phase_fitting:
                                a = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
                                b = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
                                c = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
                                phi = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)

                                optimizer_fit = torch.optim.Adam([a, b, c, phi], lr=1e-3)   # use higher if more phase
                                losses = []
                                for i in range(5000):
                                    optimizer_fit.zero_grad()
                                    im_est = torch.abs(im) * torch.exp(1j * (a * X_torch + b * Y_torch + c * Z_torch + phi))
                                    loss = (im_est - im).abs().pow(2).sum().pow(0.5) / im.abs().pow(2).sum().pow(0.5)
                                    loss.backward()
                                    optimizer_fit.step()
                                    print(f'Iter {i}, Loss {loss.item()}')
                                    losses.append(loss.item())
                                losses = np.array(losses)

                                plt.figure()
                                plt.plot(losses)
                                plt.show()
                            else:
                                a = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=False)
                                b = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=False)
                                c = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=False)
                                phi = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=False)

                        im *= torch.exp(-1j * (a * X_torch + b * Y_torch + c * Z_torch + phi))
                        im = denoiser_bw(im[None, None])

                        im = im.squeeze()

                        # put phase back
                        im *= torch.exp(1j * (a * X_torch + b * Y_torch + c * Z_torch + phi))

                    torch.cuda.empty_cache()

                # also save the phase removed images.
                if if_denoiser:
                    im2 = im * torch.exp(-1j * (a * X_torch + b * Y_torch + c * Z_torch + phi))


            # crop to (256,256,zres_recon)
            if recon_type != 'train':
                im = torch.from_numpy(im.copy())
                im = im.cuda()
            ksp_zp = torch.fft.ifftshift(im, dim=(-3, -2, -1))
            ksp_zp = torch.fft.fftn(ksp_zp, dim=(-3, -2, -1))
            ksp_zp = torch.fft.fftshift(ksp_zp, dim=(-3, -2, -1))
            ksp = ksp_zp[:, :, (zres_zp - zres_recon) // 2:(zres_zp - zres_recon) // 2 + zres_recon]

            im = torch.fft.ifftshift(ksp, dim=(-3, -2, -1))
            im = torch.fft.ifftn(im, dim=(-3, -2, -1))
            im = torch.fft.fftshift(im, dim=(-3, -2, -1))

            im = im.detach().cpu().numpy()

            im = np.rot90(im, k=3, axes=(0, 1))
            im = np.flip(im, axis=0)
            im = np.flip(im, axis=1)
            if recon_type=='train' and phase_fitting:
                ksp_zp2 = torch.fft.ifftshift(im2, dim=(-3, -2, -1))
                ksp_zp2 = torch.fft.fftn(ksp_zp2, dim=(-3, -2, -1))
                ksp_zp2 = torch.fft.fftshift(ksp_zp2, dim=(-3, -2, -1))
                ksp2 = ksp_zp2[:, :, (zres_zp - zres_recon) // 2:(zres_zp - zres_recon) // 2 + zres_recon]

                im2 = torch.fft.ifftshift(ksp2, dim=(-3, -2, -1))
                im2 = torch.fft.ifftn(im2, dim=(-3, -2, -1))
                im2 = torch.fft.fftshift(im2, dim=(-3, -2, -1))

                im2 = im2.detach().cpu().numpy()

                im2 = np.rot90(im2, k=3, axes=(0, 1))
                im2 = np.flip(im2, axis=0)
                im2 = np.flip(im2, axis=1)

            # plt.figure();
            # plt.imshow(np.angle(im[85, :, :].detach().cpu().numpy()), cmap='gray');
            # plt.show()

            # plt.figure();
            # plt.imshow(np.angle(im2[85, ...].detach().cpu().numpy()), cmap='gray');
            # plt.show()


            if bias_correction:
                corrected_image, bias = run_n4(im, n4_params)



        with h5py.File(reconiterfile, 'a') as hf:
            if pseudo_replica:
                hf.create_dataset(f"rep{rep}", data=im)

            else:
                if bias_correction:
                    hf.create_dataset(f"{recon_type}_mag_biasCorr", data=corrected_image)
                hf.create_dataset(f"{recon_type}_mag", data=np.abs(im))
                hf.create_dataset(f"{recon_type}_ph", data=np.angle(im))
                if recon_type == 'train' and phase_fitting:
                    hf.create_dataset(f"{recon_type}_ph_removed", data=np.angle(im2))


