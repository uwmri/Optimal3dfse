"""
    This file reconstruct Optimal_T2FLAIR and Optimal_T2FLAIR_c
"""

import torch
import numpy as np
import math
import h5py
import os
import glob
import matplotlib.pyplot as plt
import sigpy as sp
import sigpy.mri as mri
from sigpy.pytorch import to_pytorch, from_pytorch
from sigpy.fourier import _scale_coord, _apodize, _get_oversamp_shape
from utils import pca_coil_compression

from model import ResBlock, BlockWiseCNN


coil_compression = False
recon_type = 'train'
kw_enabled = False          # Did not end up using kw and scalek in training.
self_calib = False          # Using smaps from Sag_T2_FLAIR_CUBE scan
phase_fitting = True
blockwise = True
true_complex = False        # 3dfse not trained with fully complex conv
bias_correction = True
remove_phase_lowres = False # remove phase caused by ksp not centered

Ntrial = 6267               # starting 7/1/24 for ADRC add-on, w/coords is 1847_60, w/o coords is 6267_60
Nepoch = 60
num_unroll = 10
num_views = 2720
fov = 250                   # [mm]
xres = 256
yres = 256
zres = 128
sliceres = 1.6
zres_act = int(zres * sliceres)                     # ZFOV
zres_recon = int(zres * sliceres / (fov/xres) )     # recon to 0.976 iso resolution because imageJ can only display isotropic voxels
zres_zp = 256
disdaqs = 0                 # number of TRs before start acq
etl = 170                   # prescribed etl
num_trains = int(np.ceil(num_views/etl))
echoes2skip = 4


# scan shifts info from header
rhuser24 = 0.0
rhuser25 =  -19.38
rhuser26 = -1.25
oc_xshift = -rhuser24
oc_yshift = -rhuser25
oc_zshift = rhuser26   # Note it is rhuser26. Not -rhuser26.
deltax = 2 * math.pi * oc_xshift / fov
deltay = 2 * math.pi * oc_yshift / fov
deltaz = 2 * math.pi * oc_zshift / fov

scan_dir = rf'I:\Data\Scans_i\Opt3dfse_ADRC_add_on\AEA_08953_2024-10-15\08953_00013_Optimal_3DFLAIR\raw_data'
truth_folder = r'I:\Data\Scans_i\Opt3dfse_ADRC_add_on\AEA_08953_2024-10-15\08953_00011_Sag_T2_FLAIR_CUBE\raw_data'
nchannels = 48              # number of channels in the reference scan

centersfile = glob.glob(os.path.join(scan_dir,  'kzc*.txt'))
kwfile = glob.glob(os.path.join(scan_dir,  'kw_sort*.txt'))     # no longer used.
kyzc_raw = np.loadtxt(centersfile[0], delimiter=',')
indices = np.where(kyzc_raw[:, 0] == -129)[0]
kyzc = np.delete(kyzc_raw, indices, axis=0)
start_idx = etl * disdaqs
kyzc = kyzc[start_idx:start_idx+num_views]   
centers = np.zeros((num_views, 3), dtype=kyzc.dtype)
centers[:, 1:] = kyzc           # x is readout

# visualization
echo_indices = np.tile(np.arange(etl), num_trains)
echo_indices = echo_indices[:num_views]
fig = plt.figure(figsize=(6,3))
plt.scatter(kyzc[:, 0], kyzc[:, 1],s=3, alpha=0.7,c=echo_indices)
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(6,3))
plt.scatter(kyzc[:, 0], kyzc[:, 1],s=5, alpha=0.7,c=echo_indices);plt.gca().set_aspect('equal')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()

if not self_calib:
    # separate smaps scan
    # hfsmaps = h5py.File(name=os.path.join(scan_root, f'lowres_fov{fov}_sag_{zres_zp}_cc{coil_compression}.h5'), mode='r')
    # smaps = hfsmaps['lowres_coil'][()]
    
    # smaps from ADRC -> JSENSE generated from get_truth_smaps.py
    hfsmaps = h5py.File(name=os.path.join(truth_folder, f'smaps_jsense_z{zres_zp}.h5'), mode='r')
    smaps = hfsmaps['Maps_real'][()] + 1j* hfsmaps['Maps_imag'][()]
    smaps = np.rot90(smaps, k=1, axes=(1,2))
    smaps = smaps[:,:,:,::-1]

# get full coordinates. readout is x.
traj = np.zeros((xres, 3), dtype=np.float32)
traj[:,0] = np.linspace(-xres/2, xres/2, xres)
coords = np.zeros(traj.shape + (num_views,), dtype=np.float32)
for v in range(num_views):
    coords[..., v] = traj + np.expand_dims(centers[v], axis=0)
coords = np.transpose(coords, (0, -1, 1))
coord3D = coords.copy()
coord = np.reshape(coord3D, (-1, 3))
coord = sp.to_device(coord, sp.Device(0))

# raw kspace data
hf = h5py.File(name=os.path.join(scan_dir, f'raw.h5'), mode='r')
kdata_raw = hf['kdata_r'][()].astype(np.complex64) + 1j * hf['kdata_i'][()].astype(np.complex64)
kdata = kdata_raw[:nchannels, :, start_idx:start_idx+num_views]
kdata_cc_shift = kdata * np.exp(1j * coord3D[:, :, 0] * deltax)
kdata_cc_shift = kdata_cc_shift * np.exp(1j * coord3D[:, :, 1] * deltay)
kdata_cc_shift = kdata_cc_shift * np.exp(1j * coord3D[:, :, 2] * deltaz)
k = np.reshape(kdata_cc_shift, (nchannels, -1))
k_gpu = sp.to_device(k, sp.Device(0))

if self_calib:
    smaps = sp.mri.app.JsenseRecon(k_gpu, coord=coord, img_shape=[xres, yres, zres_zp], ksp_calib_width=32, mps_ker_width=16, lamda=0.001,
                                    max_iter=30, max_inner_iter=10,
                                    device=sp.Device(0), show_pbar=True).run()

###################################################################################################
# The kspace might not be perfectly centered. We can estimate this shift from the lowres images
# Get lowres images by first gridding to cartesian, then mask kcart
if remove_phase_lowres:
    # hann window mask
    def hann(x, width):
        xp = sp.backend.get_array_module(x)
        return .5 * (1 - xp.cos(2*math.pi*(x-width/2)/width))

    acs=32
    [kxx, kyy, kzz] = np.meshgrid(np.linspace(-acs//2, acs//2, acs), np.linspace(-acs//2, acs//2, acs), np.linspace(-acs//2, acs//2, acs), indexing='ij')
    kr = (kxx**2 + kyy**2 + kzz**2) ** .5
    fullwidth = np.sqrt(acs**2 *2)
    window = hann(kr, fullwidth)
    mask = np.zeros((xres,yres,zres_zp), dtype=np.float32)
    idxL = (xres-window.shape[0])//2
    idxS = (yres-window.shape[1])//2
    idxA = (zres_zp-window.shape[2])//2
    mask[idxL:idxL+acs, idxS:idxS+acs,idxA:idxA+acs] = window
    mask = sp.to_device(mask, sp.Device(0))
    xp = sp.backend.get_array_module(mask)

    target_coil = 24
    k_cc = pca_coil_compression(kdata=kdata_cc_shift, axis=0, target_channels=target_coil)
    kcartU_all = np.zeros((target_coil, xres, yres, zres_zp), dtype=np.complex64)

    ndim = 3
    width = 4
    oversamp = 1.25
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5

    oshape = [xres, yres, zres]
    oshape2 = [xres, yres, zres_zp]
    os_shape = _get_oversamp_shape(oshape2, ndim, oversamp)
    coord_scaled = _scale_coord(coord, oshape2, oversamp)

    for cc in range(target_coil):
        input = np.reshape(k_cc[cc], (-1,))
        input = sp.to_device(input, sp.Device(0))

        output = sp.interp.gridding(input, coord_scaled, os_shape,
                                    kernel='kaiser_bessel', width=width, param=beta)
        output /= width ** ndim

        output = sp.fourier.ifft(output, axes=range(-ndim, 0), norm=None)
        output = sp.util.resize(output, oshape2)
        output *= sp.util.prod(os_shape[-ndim:]) / sp.util.prod(oshape[-ndim:]) ** 0.5

        # Apodize
        _apodize(output, ndim, oversamp, width, beta)

        kcart = xp.fft.ifftshift(output, axes=(-3, -2, -1))
        kcart = xp.fft.fftn(kcart, axes=(-3, -2, -1))
        kcart = xp.fft.fftshift(kcart, axes=(-3, -2, -1))
    #
        kcartU = kcart * mask
        kcartU_all[cc] = kcartU.get()

    im_lowres = sp.mri.app.SenseRecon(kcartU_all, mps=smaps, weights=None,
                               device=sp.Device(0), lamda=1e-4, max_iter=1,
                               coil_batch_size=1, save_objective_values=False).run()
    im_lowres = np.squeeze(im_lowres.get().astype(np.complex64))
    flat = np.conj(im_lowres) / np.abs(im_lowres)
    # flat = im_lowres / np.abs(im_lowres)
    smaps *= flat

if coil_compression:
    kdata_cc_shift = pca_coil_compression(kdata=kdata_cc_shift, axis=0, target_channels=12)
    nchannels = kdata_cc_shift.shape[0]
if smaps.shape[0] != nchannels:
    smaps = smaps[:nchannels, ...]  # occasionally the ref/undersampled scan have 44/48 nchannels


# Do Recon
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

    # Load denoiser
    denoiser_file = os.path.join(rf'I:\Data\run_3dfse\{Ntrial}', f'denoiser_{Ntrial}_{Nepoch}.pt')
    state = torch.load(denoiser_file)
    denoiser = ResBlock(1, 1, true_complex=true_complex)
    patch_size = [128, 128, 128]
    overlap = [20, 20, 20]
    denoiser_bw = BlockWiseCNN(denoiser, patch_size, overlap).to(device)
    denoiser_bw.load_state_dict(state['state_dict'], strict=True)
    denoiser_bw.cuda();
    denoiser_bw.eval()

    # from training log
    if Ntrial == 1847:
        scale_global = torch.tensor([1.6189358]).cuda()
        scaleks = np.ones(num_unroll)
    elif Ntrial == 6267:
        scale_global = torch.tensor([1.7539611]).cuda()
        scaleks = np.ones(num_unroll)

    k_gpu_weighted = k_gpu * kw_gpu
    A = sp.mri.linop.Sense(smaps, coord=coord, weights=None, tseg=None,
                           coil_batch_size=1, comm=None,
                           transp_nufft=False)

    # 1847 and 6267 trained with init zero
    im = torch.zeros([xres, yres, zres_zp], dtype=torch.complex64, device=device)
    im.requires_grad = False

    for u in range(num_unroll):
        Ex = A * from_pytorch(im)
        Exd = Ex - k_gpu  # Ex-d
        Exd = Exd * kw_gpu

        " Data consistency step "
        im = im - to_pytorch(A.H * Exd) * scale_global
        im = im.detach()
        torch.cuda.empty_cache()

        # fitting to remove linear BG phase
        if u == 0:
            X, Y, Z = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing='ij')
            X_torch = torch.tensor(X, dtype=torch.cfloat, device='cuda')
            Y_torch = torch.tensor(Y, dtype=torch.cfloat, device='cuda')
            Z_torch = torch.tensor(Z, dtype=torch.cfloat, device='cuda')

            if phase_fitting:
                a = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
                b = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
                c = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
                phi = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)

                optimizer_fit = torch.optim.Adam([a, b, c, phi], lr=1e-3)  # use higher if more phase
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

    im2 = im * torch.exp(-1j * (a * X_torch + b * Y_torch + c * Z_torch + phi))


# save
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
if phase_fitting:
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

if bias_correction:
    import SimpleITK as sitk
    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(np.abs(im))

    # Perform N4ITK bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_sitk_image = corrector.Execute(sitk_image)

    # Convert corrected SimpleITK image back to numpy array
    corrected_image = sitk.GetArrayFromImage(corrected_sitk_image)


if recon_type == 'train':
    reconiterfile = os.path.join(scan_dir, f'recon_denoiser{if_denoiser}_phasefitting{phase_fitting}_kw{kw_enabled}_rh26.h5')
else:
    reconiterfile = os.path.join(scan_dir, f'recon_{recon_type}.h5')
with h5py.File(reconiterfile, 'a') as hf:
    if bias_correction:
        hf.create_dataset(f"{recon_type}_mag_biasCorr", data=corrected_image)
    hf.create_dataset(f"{recon_type}_mag", data=np.abs(im))
    hf.create_dataset(f"{recon_type}_ph", data=np.angle(im))
    if phase_fitting:
        hf.create_dataset(f"{recon_type}_ph_removed", data=np.angle(im2))
