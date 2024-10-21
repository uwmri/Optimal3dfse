# this script gives the truth images for training from raw KSPACE.h5 from createRaw.m for the ADRC cohort
# it can also be used to get the truth from the ADRC scan for volunteer scans
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import h5py
import os
import glob
import sigpy as sp
import sigpy.mri as mri
from utils import pca_coil_compression

id = 1
data_roots = [rf'I:\Data\Scans_i\Opt3dfse_ADRC_add_on\AEA_08953_2024-10-15\08953_00011_Sag_T2_FLAIR_CUBE\raw_data',]
recon_root = rf'I:\Data\Scans_i\Opt3dfse_ADRC_add_on\AEA_08953_2024-10-15\08953_00011_Sag_T2_FLAIR_CUBE\raw_data'
# scan_root =  rf'I:\Data\T2FLAIRTEST_V1_01261_2023-11-30'
do_recon = True
coil_compression = False
bias_correction = True
recon_type = 'l1w'

num_cases = len(data_roots)
xfov = 250
xres = 256
yres = 256
zres = 132                                  # min zres in all cases. Crop bigger ones to this size
zres_mm = 1.6
zres_zp = int(zres*zres_mm/(xfov/xres))     # this is for imagej, as it can only display isotropic voxels
# zres_zp = 256                             # for generating 256^3 data for easy training
zres_opt = 209

# when matrix is smaller than 256*256*211, always recon to this size (zero pad the kspace)
xres_raw = 256
yres_raw = 256
if coil_compression:
    target_coil = 12
else:
    target_coil = 48

for data_root in data_roots:
    data_files = glob.glob(os.path.join(data_root, 'KSPACE*.h5'))

    for file in data_files:
        hf = h5py.File(name=file, mode='r')

        kdata = hf['real'][()].astype(np.float32) + 1j * hf['imag'][()].astype(np.float32)

        # The ADRC scan has no_slab_wrap=1.03, sliceres=50%, FE=256, so actual PE used for recon is 128.
        # opslquant is also 128, not 132, so I'm guessing those extra lines are not real data
        # the collected PE=132, the extra lines are for excitation profile considerations and not used for online recon
        # so let's discard also, to keep the FOV consistent with online recon, 250*250*204.8mm
        kdata = kdata[:,66:197,...]
        num_coils = kdata.shape[0]
        zres_case = kdata.shape[1]
        # if need to crop kdata
        # idxL = int((zres_case-zres+1)/2)-1
        # idxR = idxL+zres
        # kdata = kdata[:,idxL:idxR,...]

        kdata_zp = np.zeros((num_coils, zres_zp, yres, xres), dtype=np.complex64)
        idxZ_zp = int((zres_zp-zres+1)/2)
        if yres != yres_raw:
            idxY_zp = int((yres-yres_raw+1)/2)
            idxX_zp = int((xres-xres_raw+1)/2)
            kdata_zp[:,idxZ_zp:idxZ_zp+zres,idxY_zp:idxY_zp+yres_raw:, idxX_zp:idxX_zp+xres_raw] = kdata

        else:
            kdata_zp[:,idxZ_zp:idxZ_zp+zres-1,...] = kdata

        if coil_compression:
            kdata_cc = pca_coil_compression(kdata=kdata_zp, axis=0, target_channels=target_coil)
        else:
            kdata_cc = kdata_zp

        im = cp.zeros_like(kdata_cc)
        k = cp.zeros_like(kdata_cc)
        for c in range(kdata_cc.shape[0]):
            kdata_cc_gpu = sp.to_device(kdata_cc[c], sp.Device(0))
            im[c] = cp.fft.ifftshift(kdata_cc_gpu, axes=(-3, -2, -1))
            im[c] = cp.fft.ifftn(im[c], axes=(-3, -2, -1))

            k[c] = cp.fft.ifftshift(im[c], axes=(-3, -2, -1))
            k[c] = cp.fft.fftn(k[c], axes=(-3, -2, -1))
            k[c] = cp.fft.fftshift(k[c], axes=(-3, -2, -1))



        smapsJ = sp.mri.app.JsenseRecon(k.get(), ksp_calib_width=32, mps_ker_width=16, lamda=0.001,
                                     max_iter=30, max_inner_iter=10,
                                     device=sp.Device(0), show_pbar=True).run()
        im = sp.mri.app.SenseRecon(k, mps=smapsJ, device=sp.Device(0), lamda=1e-4, max_iter=1,
                                   coil_batch_size=1, save_objective_values=False).run()

        if do_recon:
            if recon_type == 'l1w':
                # it has been lamda=1e-4, max_iter=15 for generating all ground truths
                im = sp.mri.app.L1WaveletRecon(k, mps=smapsJ, device=sp.Device(0), lamda=1e-4, max_iter=15,
                                               coil_batch_size=1, save_objective_values=False).run()
            if recon_type == 'tv':
                # it has been lamda=1e-4, max_iter=15 for generating all ground truths
                im = sp.mri.app.TotalVariationRecon(k, mps=smapsJ, device=sp.Device(0), lamda=1e-3, max_iter=15,
                                               coil_batch_size=1, save_objective_values=False).run()
            elif recon_type == 'pils':
                im = sp.mri.app.SenseRecon(k, mps=smapsJ, device=sp.Device(0), lamda=1e-4, max_iter=1,
                                               coil_batch_size=1, save_objective_values=False).run()

        im = np.squeeze(im.get().astype(np.complex64))

        # move z(slice dir) to the last dim, so it is in the same orientation as denoiser training
        im = np.rot90(im, k=1, axes=(0, 1))
        im = np.rot90(im, k=1, axes=(1, 2))
        scale_truth = 1.0 / np.max(np.abs(im))
        im *= scale_truth

        if do_recon:
            im_cube = im[::-1,::-1,::-1]
            idxZ_opt = int((zres_zp-zres_opt)/2)
            im_cube = im_cube[:,:,idxZ_opt:zres_opt+idxZ_opt]
        else:
            im_cube = im

        if bias_correction:
            import SimpleITK as sitk

            # Convert numpy array to SimpleITK image
            sitk_image = sitk.GetImageFromArray(np.abs(im_cube))

            # Perform N4ITK bias field correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected_sitk_image = corrector.Execute(sitk_image)

            # Convert corrected SimpleITK image back to numpy array
            corrected_image = sitk.GetArrayFromImage(corrected_sitk_image)

        # move z(slice dir) to the last dim, so it is in the same orientation as denoiser training
        smaps = np.rot90(smapsJ, k=1, axes=(1, 2))
        smaps = np.rot90(smaps, k=1, axes=(2, 3))
        smaps_cube = smaps

        if do_recon:
            reconfile = os.path.join(data_root, f'recon_{recon_type}_z{zres_zp}_{target_coil}ch.h5')
            # reconfile = os.path.join(recon_root, f'KSPACE_0{id}_24chan.h5')

        else:
            if coil_compression:
                reconfile = os.path.join(recon_root, f'smaps_jsense_z{zres_zp}_{target_coil}ch.h5')
            else:
                reconfile = os.path.join(recon_root, f'smaps_jsense_z{zres_zp}.h5')

        try:
            os.remove(reconfile)
        except OSError:
            pass
        with h5py.File(reconfile, 'a') as hf:
            if do_recon:
                hf.create_dataset(f"Images_real", data=np.real(im_cube))
                hf.create_dataset(f"Images_imag", data=np.imag(im_cube))
                hf.create_dataset(f"Images_mag", data=np.abs(im_cube))
                if bias_correction:
                    hf.create_dataset(f"Images_bc", data=corrected_image)
                hf.create_dataset(f"Images_ph", data=np.angle(im_cube))
            if zres_zp==256:
                hf.create_dataset(f"Maps_real", data=np.real(smaps_cube))
                hf.create_dataset(f"Maps_imag", data=np.imag(smaps_cube))
                hf.create_dataset(f"im_pils", data=np.abs(im_cube))

            # hf.create_dataset(f"Maps_mag", data=np.abs(smaps_cube))
            # hf.create_dataset(f"Maps_ph", data=np.angle(smaps_cube))
        id+=1