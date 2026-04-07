"""
    Simulation on test case.
"""
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import os
import re
from random import randrange
import argparse
import h5py
import numpy as np
import cupy as cp
import sigpy as sp
import csv
from sigpy import mri
# import nufftbindings.kbnufft as nufft
from nufft_sigpy import NUFFT, NUFFTadjoint
from model import ResBlock, BlockWiseCNN
from utils_DL import RunningAverage
import logging
import glob
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nufft_forward = NUFFT.apply
nufft_adjoint = NUFFTadjoint.apply

def _data_range(a, b):
    a_min = float(np.min(a))
    a_max = float(np.max(a))
    b_min = float(np.min(b))
    b_max = float(np.max(b))
    dr = max(a_max, b_max) - min(a_min, b_min)
    if dr == 0.0:
        dr = 1.0
    return dr

fov = 250
cube_size = 256
xres = 256
yres = 256
zres = 128
zres_zp = 256
slthick = 1.6
zres_recon = int(zres * slthick / (fov/xres) )
scale = 1.
recon_type='l1w'
# Ntrials = [1847, 6267]
Ntrials = [6267]
longitudinal = False
num_views = 2720
add_gaussian = False
gaussian_level = 1
NDIM = 3
bias_correction = False

# make an elliptical cylinder mask for sigma estimation when adding gaussian noise
a_acq = xres//2
b_acq = zres//2
h_acq = yres
a_edge = 100
b_edge = 52
h_edge = 200
x, y, z = np.ogrid[:xres, :xres, :xres]
large_cylinder = ((x - (cube_size // 2))**2 / (a_acq**2) + (z - (cube_size // 2))**2 / (b_acq**2)) <= 1
large_cylinder = large_cylinder & (y >= (cube_size // 2 - h_acq // 2)) & (y <= (cube_size // 2 + h_acq // 2))
small_cylinder = ((x - (cube_size // 2))**2 / (a_edge**2) + (z - (cube_size // 2))**2 / (b_edge**2)) <= 1
small_cylinder = small_cylinder & (y >= (cube_size // 2 - h_edge // 2)) & (y <= (cube_size // 2 + h_edge // 2))
small_cylinder = 1 - small_cylinder
mask = large_cylinder & small_cylinder
mask = sp.to_device(mask, sp.Device(0))

if longitudinal:
    # Nepoch = np.linspace(0,990,100, dtype=np.int)
    Nepochs = np.linspace(0,100,11, dtype=np.int)
    # Nepoch = [80]
    data_folder = [r'I:\99_OPT_FLAIR_INBOX\KLCEVER2757_01887_2024-05-22\01887_00006_Sag_T2_FLAIR_CUBE\raw_data']
else:
    Nepochs = [60, 60]
    data_root = [r'S:\Opt3dfse_ADRC_add_on']
    data_folder = []
    for root in data_root:
        data_folder.extend(sorted(
            glob.glob(os.path.join(root, "*_????-??-??", "*Sag_T2_FLAIR_CUBE", "raw_data"))
        ))

scantxt_dir = r'S:\code\Optimal3dfse\txt'


NMSEs = np.zeros((len(Nepochs), len(data_folder), len(Ntrials)))
SSIMs = np.zeros((len(Nepochs), len(data_folder), len(Ntrials)))
PSNRs = np.zeros((len(Nepochs), len(data_folder), len(Ntrials)))

# for c in range(len(data_folder)):
for c in range(7):

    for n, Ntrial in enumerate(Ntrials):
        true_complex = False
        kw_enabled = False
        if not longitudinal:
            Nepoch = [Nepochs[n],]
        else:
            Nepoch = Nepochs
        if Ntrial == 5487:
            num_unroll = 4
        elif Ntrial == 8102 or Ntrial == 3161 or Ntrial == 9485 or Ntrial == 8994:
            num_unroll = 10
            true_complex = True
            kw_enabled = False
        else:
            num_unroll = 10

        log_dir = rf'D:\SSD\Data\run_3dfse\{Ntrial}'
        evalimagesfile = os.path.join(data_folder[c], f'inference_{Ntrial}_longitudinal{longitudinal}_{recon_type}.h5')
        if os.path.exists(evalimagesfile):
            os.remove(evalimagesfile)
        # if os.path.exists(evalimagesfile):
        #     print(f'Skipping case {c} ({data_folder[c]}): {evalimagesfile} already exists.')
        #     continue

        scale_globals = []
        scaleks_all = []

        # Open the file for reading
        pattern = re.compile(r"INFO:root:scalek=\[(.*?)\]")
        with open(os.path.join(log_dir, f'3dfse_{Ntrial}.log'), "r") as file:
            for line in file:
                match = pattern.search(line)

                if "INFO:root:scale_global" in line:
                    # Extract the value between the square brackets
                    start_index = line.find("[")
                    end_index = line.find("]")
                    if start_index != -1 and end_index != -1:
                        value = float(line[start_index + 1:end_index])
                        scale_globals.append(value)
                if match:
                    scalek_str = match.group(1)
                    scalek_array = np.array([float(x) for x in scalek_str.split()])
                    # Append the numpy array to the list
                    scaleks_all.append(scalek_array)

        if not longitudinal:
            scale_globals = [scale_globals[-1]]
            if Ntrial == 5487:
                scaleks_all = [[0.66,0.68,0.80,0.89]]
            elif Ntrial == 6400:
                scaleks_all = [[0.67,0.67, 0.76,0.82,0.90]]
            elif Ntrial == 2830:
                scaleks_all = [[0.67,0.66, 0.76,0.81,0.90]]
            else:
                scaleks_all = [scaleks_all[-1]]
            print(scale_globals)
            print(scaleks_all)


        for e in range(len(Nepoch)):
            epoch = Nepoch[e]
            print(f'epoch {epoch}')
            if Ntrial != 2830:

                if not longitudinal:
                    # load the actual re-ordered coords from the scanner
                    centersfile = glob.glob(os.path.join(scantxt_dir, f'kzc_{Ntrial}_actual.txt'))
                    kyzc_raw = np.loadtxt(centersfile[0], delimiter=',')  # kzc, kyc

                else:
                    kyzc_raw = np.loadtxt(os.path.join(log_dir, f'centers_fse_{num_views}_{Ntrial}_{epoch}.txt'), delimiter=',')  # kzc, kyc
                indices = np.where(kyzc_raw[:, 0] == -129)[0]
                kyzc = np.delete(kyzc_raw, indices, axis=0)
                kyc0 = kyzc[:,0]
                kzc0 = kyzc[:,1]
            else:
                kyc0, kzc0 = np.loadtxt(os.path.join(log_dir,f'centers_poisson_256_130.txt'), delimiter=',',
                                        unpack=True)  # kzc, kyc
            kyc = torch.from_numpy(kyc0).cuda().flatten()
            kzc = torch.from_numpy(kzc0).cuda().flatten()
            kxc = torch.zeros_like(kyc)
            helix_xread = np.zeros((xres,3),dtype=np.float32)
            helix_xread[:,0] = np.linspace(-xres/2, xres/2, xres)

            scale_global = torch.tensor([scale_globals[e]], requires_grad=False, device='cuda')    #1.5612073 for #6400 epoch 990, 1.4971551 for #5487 epoch 999
            scaleks = torch.tensor(scaleks_all[e], requires_grad=False, device='cuda')

            if kw_enabled:
                if not longitudinal:
                    kwfile = glob.glob(os.path.join(scantxt_dir, 'kw_sort*.txt'))
                    kw_raw = np.loadtxt(kwfile[0], delimiter=',')  # kzc, kyc
                    # kw_raw = np.loadtxt(os.path.join(log_dir, f'kw_{Ntrial}_{epoch}.txt'), delimiter=',')

                else:
                    kw_raw = np.loadtxt(os.path.join(log_dir, f'kw_{Ntrial}_{epoch}.txt'), delimiter=',')
            else:
                kw_raw = np.ones((num_views,))
            indices = np.where(kw_raw == -1)
            kw = np.delete(kw_raw, indices)
            kw = torch.from_numpy(kw.astype(np.float32))
            kw2d = torch.tile(kw, (xres, 1))  # (xres, num_views)
            kw2d = kw2d.reshape((-1,))
            kw2d = kw2d.cuda()

            denoiser_file = os.path.join(log_dir, f'denoiser_{Ntrial}_{epoch}.pt')
            state = torch.load(denoiser_file)
            if Ntrial == 5487 or Ntrial == 6400 or Ntrial == 2830:
                denoiser = ResBlock(1, 1)
                denoiser.load_state_dict(state['state_dict'])
                denoiser.cuda()
                denoiser.eval()
            else:
                denoiser = ResBlock(1, 1)
                patch_size = [128, 128, 128]
                overlap = [20, 20, 20]
                denoiser_bw = BlockWiseCNN(denoiser, patch_size, overlap).to(device)
                denoiser_bw.load_state_dict(state['state_dict'])
                denoiser_bw.eval()



            kyzc = torch.stack((kyc, kzc), axis=1)
            coords_centers = torch.stack((kxc, kyc, kzc), axis=1)
            helix_torch = torch.from_numpy(helix_xread.copy()).cuda()
            coord = torch.zeros(helix_xread.shape + (num_views,), dtype=torch.float32).cuda()
            for v in range(num_views):
                coord[..., v] = helix_torch + coords_centers[v].unsqueeze(0)
            coord = torch.permute(coord, (0, -1, 1))
            coord = torch.reshape(coord, (-1, NDIM))



            recon_files = sorted(glob.glob(os.path.join(data_folder[c], f'recon_l1w_z{zres_zp}*.h5')))
            recon_file = recon_files[0]
            with h5py.File(recon_file, 'r') as hf:
                truth = hf['Images_real'][()] + 1j * hf['Images_imag'][()]
            with h5py.File(os.path.join(data_folder[c], f'smaps_jsense_z{zres_zp}.h5'), 'r') as hf:
                smaps = hf['Maps_real'][()] + 1j * hf['Maps_imag'][()]


            num_coils = smaps.shape[0]
            truth_tensor = torch.tensor(truth, dtype=torch.complex64)
            truth_tensor = truth_tensor.cuda()
            truth_tensor.requires_grad = False
            truth_tensor = torch.rot90(truth_tensor, k=1, dims=(0,1))

            smaps_tensor = torch.from_numpy(smaps).type(torch.complex64)
            smaps_tensor.requires_grad = False
            smaps_tensor = torch.rot90(smaps_tensor, k=1, dims=(1,2))

            if add_gaussian:
                truth_gpu = sp.to_device(truth, sp.Device(0))
                ktruth = cp.fft.ifftshift(truth_gpu, axes=(-3, -2, -1))
                ktruth = cp.fft.fftn(ktruth, axes=(-3, -2, -1))
                ktruth = cp.fft.fftshift(ktruth, axes=(-3, -2, -1))
                kedge = ktruth * mask
                kedge = kedge.reshape((-1,))
                kedge = kedge[np.nonzero(kedge)]
                sigma_est_real = np.median(np.abs(np.real(kedge)))
                sigma_est_imag = np.median(np.abs(np.imag(kedge)))
                sigma_real = gaussian_level * sigma_est_real
                sigma_imag = gaussian_level * sigma_est_imag
                print(f'sigma_real = {sigma_real}, sigma_imag = {sigma_imag}')

                gauss_real = cp.random.normal(0.0, sigma_real, (xres, xres, xres))
                gauss_imag = cp.random.normal(0.0, sigma_imag, (xres, xres, xres))
                knoisy_real = cp.real(ktruth) + gauss_real
                knoisy_imag = cp.imag(ktruth) + gauss_imag
                knoisy = knoisy_real + 1j * knoisy_imag
                truth_noisy = cp.fft.ifftshift(knoisy, axes=(-3, -2, -1))
                truth_noisy = cp.fft.ifftn(truth_noisy, axes=(-3, -2, -1))
                truth_noisy = cp.fft.fftshift(truth_noisy, axes=(-3, -2, -1))

                truth_tensor2 = torch.tensor(truth_noisy.get(), dtype=torch.complex64)
                truth_tensor2 = truth_tensor2.cuda()
                truth_tensor2.requires_grad = False
                truth_tensor2 = torch.rot90(truth_tensor2, k=1, dims=(0,1))

            cp._default_memory_pool.free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

            with torch.no_grad():
                if add_gaussian:
                    y = nufft_forward(coord, truth_tensor2, smaps_tensor)
                else:
                    y = nufft_forward(coord, truth_tensor, smaps_tensor)

                y_weighted = y * kw2d

                if recon_type == 'l1w':
                    smaps = np.rot90(smaps, k=1, axes=(1, 2))
                    im = sp.mri.app.L1WaveletRecon(sp.to_device(y_weighted.detach().cpu().numpy(), sp.Device(0)), mps=smaps, weights=None, coord=coord,
                                                   device=sp.Device(0), lamda=1e-4, max_iter=15,
                                                   coil_batch_size=1, save_objective_values=False).run()
                    im = np.squeeze(im.get().astype(np.complex64))
                    im = torch.from_numpy(im).cuda()
                    im_mag = np.abs(im.detach().cpu().numpy())

                else:
                    if Ntrial == 5487 or Ntrial == 6400 or Ntrial == 2830:
                        im = nufft_adjoint(coord, y_weighted, smaps_tensor)
                    else:
                        im = nufft_adjoint(coord, y, smaps_tensor)
                        im = torch.zeros_like(im)
                    for u in range(num_unroll):
                        Ex = nufft_forward(coord, im, smaps_tensor)
                        # if Ntrial == 5487 or Ntrial == 6400 or Ntrial == 2830:
                        #     scalek = torch.linalg.norm(Ex) / torch.linalg.norm(y)
                        # else:
                        scalek = scaleks[u]
                        # print(f'unroll {u}, scalek = {scalek}')
                        Ex = Ex / scalek

                        Exd = Ex - y  # Ex-d
                        if kw_enabled:
                            Exd = Exd * kw2d

                        # Data consistency
                        im = im - nufft_adjoint(coord, Exd, smaps_tensor) * scale_global

                        # " denoiser "
                        if Ntrial == 5487 or Ntrial == 6400 or Ntrial == 2830:
                            im = torch.rot90(im, k=3, dims=(0,1))
                            im = denoiser(im[None, None])
                            im = im.squeeze()
                            im = torch.rot90(im, k=1, dims=(0,1))
                        else:
                            im = denoiser_bw(im[None, None])
                        im = im.squeeze()

                        torch.cuda.empty_cache()


                    im_mag = np.abs(im.detach().cpu().numpy())
                    im_mag = np.rot90(im_mag, k=3, axes=(0, 1))
                truth_mag = np.abs(truth)

                # match historgrams
                from skimage.exposure import match_histograms
                im_mag = match_histograms(im_mag, truth_mag)

                # scaling
                scale = np.sum(im_mag * truth_mag) / np.sum(im_mag * im_mag)
                im_mag *= scale

                log_file = os.path.join(data_root[0], f'inference_{Ntrial}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
                with open(log_file, 'a') as f:
                    f.write(f"{'='*60}\n")
                    f.write(f"Epoch: {epoch}, Case: {c}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{'Image (im_mag) Statistics:':<30}\n")
                    f.write(f"  {'Mean:':<10} {np.mean(im_mag):.6f}\n")
                    f.write(f"  {'Std:':<10} {np.std(im_mag):.6f}\n")
                    f.write(f"  {'Min:':<10} {np.min(im_mag):.6f}\n")
                    f.write(f"  {'Max:':<10} {np.max(im_mag):.6f}\n")
                    f.write(f"\n{'Truth (truth_mag) Statistics:':<30}\n")
                    f.write(f"  {'Mean:':<10} {np.mean(truth_mag):.6f}\n")
                    f.write(f"  {'Std:':<10} {np.std(truth_mag):.6f}\n")
                    f.write(f"  {'Min:':<10} {np.min(truth_mag):.6f}\n")
                    f.write(f"  {'Max:':<10} {np.max(truth_mag):.6f}\n")
                    f.write(f"\n")

                # NMSE on mag images to match metrics calc on coregistered actual acquisitions
                NMSE =  float(np.sqrt(np.sum((im_mag - truth_mag) ** 2)) / (np.sqrt(np.sum(truth_mag ** 2)) + 1e-12))

                drange = _data_range(im_mag, truth_mag)
                ssim = structural_similarity(im_mag, truth_mag, data_range=drange)
                psnr = peak_signal_noise_ratio(truth_mag, im_mag, data_range=drange)

                print(f'range={drange}')
                print(f'{Ntrial},epoch {epoch}, NMSE = {NMSE}, SSIM = {ssim}, PSNR = {psnr}')

                NMSEs[e, c, n] = NMSE
                SSIMs[e, c, n] = ssim
                PSNRs[e, c, n] = psnr

                # convert back to (256,256,209)
                ksp_zp = torch.fft.ifftshift(im, dim=(-3, -2, -1))
                ksp_zp = torch.fft.fftn(ksp_zp, dim=(-3, -2, -1))
                ksp_zp = torch.fft.fftshift(ksp_zp, dim=(-3, -2, -1))
                ksp = ksp_zp[:, :, (zres_zp-zres_recon) //2:(zres_zp-zres_recon) //2 + zres_recon]

                im = torch.fft.ifftshift(ksp, dim=(-3, -2, -1))
                im = torch.fft.ifftn(im, dim=(-3, -2, -1))
                im = torch.fft.fftshift(im, dim=(-3, -2, -1))

            im_np = im.detach().cpu().numpy()
            if bias_correction:
                import SimpleITK as sitk
                # Convert numpy array to SimpleITK image
                sitk_image = sitk.GetImageFromArray(np.abs(im_np))

                # Perform N4ITK bias field correction
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrected_sitk_image = corrector.Execute(sitk_image)

                # Convert corrected SimpleITK image back to numpy array
                corrected_image = sitk.GetArrayFromImage(corrected_sitk_image)

                with h5py.File(evalimagesfile, 'a') as hf:
                    hf.create_dataset(f"epoch{epoch}_mag_case{c}_biasCorr", data=np.squeeze(np.abs(corrected_image)))
                    hf.create_dataset(f"epoch{epoch}_ph_case{c}", data=np.squeeze(np.angle(im_np)))
                    # hf.create_dataset(f"epoch{epoch}", data=np.squeeze(im.detach().cpu().numpy()))
            else:
                with h5py.File(evalimagesfile, 'a') as hf:
                    hf.create_dataset(f"epoch{epoch}_mag_case{c}", data=np.squeeze(np.abs(im_np)))
                    hf.create_dataset(f"epoch{epoch}_ph_case{c}", data=np.squeeze(np.angle(im_np)))

        # plot MSE and SSIM curve vs epochs
        if longitudinal:
            # load metrics files
            fig = plt.figure(figsize=(6,4))
            # plt.plot(Nepoch, NMSEs[:, 0, 0], linewidth=3, label='Trained with added Gaussian noise')
            plt.plot(Nepoch, NMSEs[:, 0, 0], linewidth=3, label='Learnanle sampling')
            plt.plot(Nepoch, NMSEs[:, 0, 1], linewidth=3, label='Fixed sampling')
            # plt.ylim(0.1, 0.25)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('NMSE', fontsize=14)
            plt.legend(fontsize=14)
            plt.show()
            plt.savefig(os.path.join(data_folder[0], f'inference_{Ntrial}_NMSE.png'), bbox_inches='tight')

            fig = plt.figure(figsize=(6,4))
            # plt.plot(Nepoch, SSIMs[:,0,0], linewidth=3, label='Trained with added Gaussian noise')
            plt.plot(Nepoch, SSIMs[:,0,0], linewidth=3, label='Learnanle sampling')
            plt.plot(Nepoch, SSIMs[:,0,1], linewidth=3, label='Fixed sampling')
            # plt.ylim(0.92,1)
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel('SSIM', fontsize=14)
            plt.legend(fontsize=14)
            plt.savefig(os.path.join(data_folder[0], f'inference_{Ntrial}_SSIM.png'), bbox_inches='tight')

if not longitudinal:
    NMSEs_list = [NMSEs[0,:,i].tolist() for i in range(len(Ntrials))]
    nmse_rows = list(zip(*NMSEs_list))
    nmse_csv = os.path.join(data_root[0], "inference_nmse_l1w.csv")
    with open(nmse_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["data_folder"] + [f"Ntrial_{n}" for n in Ntrials])
        for folder, row in zip(data_folder, nmse_rows):
            writer.writerow([folder] + list(row))

    SSIMs_list = [SSIMs[0,:,i].tolist()  for i in range(len(Ntrials))]
    ssim_rows = list(zip(*SSIMs_list))
    ssim_csv = os.path.join(data_root[0], "inference_ssim_l1w.csv")
    with open(ssim_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["data_folder"] + [f"Ntrial_{n}" for n in Ntrials])
        for folder, row in zip(data_folder, ssim_rows):
            writer.writerow([folder] + list(row))

    PSNRs_list = [PSNRs[0,:,i].tolist() for i in range(len(Ntrials))]
    psnr_rows = list(zip(*PSNRs_list))
    psnr_csv = os.path.join(data_root[0], "inference_psnr_l1w.csv")
    with open(psnr_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["data_folder"] + [f"Ntrial_{n}" for n in Ntrials])
        for folder, row in zip(data_folder, psnr_rows):
            writer.writerow([folder] + list(row))

