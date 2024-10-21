import matplotlib.pyplot as plt
import time
import os
import random
from random import randrange
import argparse
import h5py
import numpy as np
import cupy as cp
import sigpy as sp
from sigpy import mri
from nufft_sigpy import NUFFT, NUFFTadjoint
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import glob

from model import ResBlock, BlockWiseCNN
from utils_DL import RunningAverage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nufft_forward = NUFFT.apply
nufft_adjoint = NUFFTadjoint.apply

NDIM = 3

init_type = 'poisson'
Ntrial = randrange(10000)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_folder', type=str,
                    default=r'/mnt/PURENFS/cxt004/2Dsampling/T2FLAIR_CS_cubenorm/train',
                    help='Data path')

parser.add_argument('--data_folder_val', type=str,
                    default=r'/mnt/PURENFS/cxt004/2Dsampling/T2FLAIR_CS_cubenorm/val',
                    help='Data path')
parser.add_argument('--log_dir', type=str,
                    default=r'/mnt/PURENFS/cxt004/2Dsampling/run_3dfse',
                    help='Directory to log files')

parser.add_argument('--noise_dir', type=str,
                    default=r'/mnt/PURENFS/cxt004/2Dsampling/gaussian_noise_files',
                    help='Directory to log files')
parser.add_argument('--wave_dir', type=str,
                    default=r'/mnt/PURENFS/cxt004/2Dsampling/wave/create_wave_ro360',
                    help='Directory to log files')
parser.add_argument('--Nepochs', type=int, default=2000)
parser.add_argument('--r', type=int, default=6)
parser.add_argument('--cycles', type=int, default=8)
parser.add_argument('--nro', type=int, default=256)
parser.add_argument('--num_views', type=int, default=2720)
parser.add_argument('--acc', type=float, default=6)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_coils', type=int, default=12)
parser.add_argument('--pname', type=str, default=f'fse_{Ntrial}')
parser.add_argument('--resume_train', action='store_true', default=True)
parser.add_argument('--Ntrial_old', type=int, default=2208)
parser.add_argument('--Nepoch_old', type=int, default=50)
parser.add_argument('--scale_global', type=float, default=1.7540)
parser.add_argument('--unroll', type=int, default=5)
parser.add_argument('--check_scale', action='store_true', default=False)
parser.add_argument('--save_train_images', action='store_true', default=True)
parser.add_argument('--dgx', action='store_true', default=False)
parser.add_argument('--cn1', action='store_true', default=False)
parser.add_argument('--first_run', action='store_true', default=False)
parser.add_argument('--init_zero', action='store_true', default=True)

args = parser.parse_args()

writer_train = SummaryWriter(os.path.join(args.log_dir, f'train_{Ntrial}'))
writer_val = SummaryWriter(os.path.join(args.log_dir, f'val_{Ntrial}'))
logging.basicConfig(filename=os.path.join(args.log_dir, f'3dfse_{Ntrial}.log'),
                    filemode='w', level=logging.INFO)
logging.info(f'device={device}')
logging.info(f'{torch.cuda.is_available()}')

try:
    import setproctitle

    setproctitle.setproctitle(args.pname)
    print(f'Setting program name to {args.pname}')
except:
    print('setproctitle not installled,unavailable, or failed')


xres = 256
yres = 256
zres = 132

# create a mask if it's the first run. Both acc and num_views(npe) need to be specified as npe needs to be integer*ETL.
if args.first_run:
    for _ in range(100):
        mask = mri.poisson((yres, zres), accel=args.acc, calib=(0, 0), crop_corner=True, return_density=False,
                           dtype='float32', seed=None)
        kyc0, kzc0 = np.nonzero(mask)
        num_views = np.count_nonzero(mask)
        print(num_views)
        if num_views == args.npe:
            kyc0 = kyc0.astype(np.float32)
            kzc0 = kzc0.astype(np.float32)
            kyc0 -= kyc0.max() / 2
            kzc0 -= kzc0.max() / 2

            coords0 = np.stack((kyc0, kzc0), axis=1)
            centers0 = os.path.join(args.log_dir, f'centers_poisson_{xres}_{zres}_{Ntrial}_{args.num_views}.txt')
            try:
                os.remove(centers0)
            except OSError:
                pass
            np.savetxt(centers0, coords0, fmt='%f', delimiter=",")

            break
else:
    if not args.resume_train:
        kyc0, kzc0 = np.loadtxt(os.path.join(args.log_dir, f'centers_poisson_{xres}_{zres}_{args.num_views}.txt'), delimiter=',',
                                unpack=True)

        denoiser = ResBlock(1, 1)
        patch_size = [128, 128, 128]
        overlap = [20, 20, 20]
        denoiser_bw = BlockWiseCNN(denoiser, patch_size, overlap).to(device)

    else:
        logging.info(f'resuming {args.Ntrial_old} epoch {args.Nepoch_old}')
        kyc0, kzc0 = np.loadtxt(os.path.join(args.log_dir, f'centers_fse_{args.num_views}_{args.Ntrial_old}_{args.Nepoch_old}.txt'), delimiter=',',
                                unpack=True)

        denoiser = ResBlock(1, 1)
        patch_size = [128, 128, 128]
        overlap = [20, 20, 20]
        denoiser_bw = BlockWiseCNN(denoiser, patch_size, overlap).to(device)

        state = torch.load(os.path.join(args.log_dir, f'denoiser_{args.Ntrial_old}_{args.Nepoch_old}.pt'))
        denoiser_bw.load_state_dict(state['state_dict'], strict=True)

scale_global = torch.tensor([args.scale_global], requires_grad=True, device='cuda')


kyc = torch.from_numpy(kyc0).cuda().flatten()
kzc = torch.from_numpy(kzc0).cuda().flatten()
kxc = torch.zeros_like(kyc)
kyc.requires_grad = True
kzc.requires_grad = True
traj_xread = np.zeros((xres, 3), dtype=np.float32)
traj_xread[:, 0] = np.linspace(-xres / 2, xres / 2, xres)

optimizer = torch.optim.Adam([{'params': denoiser_bw.parameters()},
                              {'params': scale_global},
                             {'params': kyc},
                             {'params': kzc},
                              ], lr=1e-3, weight_decay=1e-4)
if args.resume_train:
    optimizer.load_state_dict(state['optimizer'])
denoiser_bw.cuda()

# save some eval images
evalimagesfile = os.path.join(args.log_dir, f'eval_{Ntrial}.h5')
try:
    os.remove(evalimagesfile)
except OSError:
    pass

# hard cutoff: the coordinates cannot exceed opres
activationY = torch.nn.Hardtanh(min_val=- yres / 2 , max_val=yres / 2, inplace=False)
activationZ = torch.nn.Hardtanh(min_val=- zres / 2, max_val=zres / 2 , inplace=False)

for epoch in range(args.Nepochs):

    gaussian_level = random.uniform(0,2)


    # average loss for num_cases
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    data_files = glob.glob(os.path.join(args.data_folder, '*.h5'))
    num_cases = len(data_files)
    logging.info(f'epoch {epoch}, {num_cases} cases')

    denoiser_bw.train()

    idx = np.random.permutation(np.arange(num_cases))
    for cs in range(num_cases):

        optimizer.zero_grad()

        kyc_clamp = activationY(kyc)
        kzc_clamp = activationZ(kzc)

        i = int(idx[cs])
        file = data_files[i]
        with h5py.File(file, 'r') as hf:
            truth = hf['Images_real'][()] + 1j* hf['Images_imag'][()]
            smaps = hf['Maps_real'][()] + 1j* hf['Maps_imag'][()]
        num_coils = smaps.shape[0]
        truth_gpu = sp.to_device(truth, sp.Device(0))
        ktruth = cp.fft.ifftshift(truth_gpu, axes=(-3, -2, -1))
        ktruth = cp.fft.fftn(ktruth, axes=(-3, -2, -1))
        ktruth = cp.fft.fftshift(ktruth, axes=(-3, -2, -1))

        # estimate noise sigma from the shell of outer kspace
        sigma_real = gaussian_level * sigma_est_real
        sigma_imag = gaussian_level * sigma_est_imag

        gauss_real = cp.random.normal(0.0, sigma_real, (xres, xres, xres))
        gauss_imag = cp.random.normal(0.0, sigma_imag, (xres, xres, xres))
        knoisy_real = cp.real(ktruth) + gauss_real
        knoisy_imag = cp.imag(ktruth) + gauss_imag
        knoisy = knoisy_real + 1j * knoisy_imag
        truth_noisy = cp.fft.ifftshift(knoisy, axes=(-3, -2, -1))
        truth_noisy = cp.fft.ifftn(truth_noisy, axes=(-3, -2, -1))
        truth_noisy = cp.fft.fftshift(truth_noisy, axes=(-3, -2, -1))
        
        truth_tensor = torch.tensor(truth, dtype=torch.complex64)
        truth_tensor = truth_tensor.cuda()
        truth_tensor.requires_grad = False
        truth_tensor = torch.rot90(truth_tensor, k=1, dims=(0,1))

        truth_tensor2 = torch.tensor(truth_noisy.get(), dtype=torch.complex64)
        truth_tensor2 = truth_tensor2.cuda()
        truth_tensor2.requires_grad = False
        truth_tensor2 = torch.rot90(truth_tensor2, k=1, dims=(0,1))

        smaps_tensor = torch.from_numpy(smaps).type(torch.complex64)
        smaps_tensor = torch.rot90(smaps_tensor, k=1, dims=(1,2))
        smaps_tensor.requires_grad = False

        # generate full coordinates
        kyzc = torch.stack((kyc_clamp, kzc_clamp), axis=1)
        coords_centers = torch.stack((kxc, kyc_clamp, kzc_clamp), axis=1)
        traj_torch = torch.from_numpy(traj_xread.copy()).cuda()
        coord = torch.zeros(traj_xread.shape + (num_views,), dtype=torch.float32).cuda()
        for v in range(num_views):
            coord[..., v] = traj_torch + coords_centers[v].unsqueeze(0)
        coord = torch.permute(coord, (0, -1, 1))
        coord = torch.reshape(coord, (-1, NDIM))

        cp._default_memory_pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

        y = nufft_forward(coord, truth_tensor2, smaps_tensor)
        # # if learn density compensation
        # y_weighted = y * kw2d
        # im = nufft_adjoint(coord, y_weighted, smaps_tensor)
        im = nufft_adjoint(coord, y, smaps_tensor)
        if args.init_zero:
            im = torch.zeros_like(im)

        for u in range(args.unroll):
            Ex = nufft_forward(coord, im, smaps_tensor)

            Exy = Ex - y
            # Exy = Exy * kw2d

            " Data consistency step "
            im = im - nufft_adjoint(coord, Exy, smaps_tensor) * scale_global
            im = denoiser_bw(im[None, None])
            im = im.squeeze()
            torch.cuda.empty_cache()

        lossMSE = (im - truth_tensor).abs().pow(2).sum().pow(0.5) / truth_tensor.abs().pow(2).sum().pow(0.5)
        lossMSE.backward(retain_graph=True)
        optimizer.step()

        train_avg.update(lossMSE.detach().item())
    
    kyc_clamp = activationY(kyc)
    kzc_clamp = activationZ(kzc)

    writer_train.add_scalar('LossMSE', train_avg.avg(), epoch)
    writer_train.add_scalar('scale_global', scale_global[0].detach().cpu().numpy(), epoch)
    
    ########################################### plot and save #################################################
    
    
    with torch.no_grad():
        denoiser_bw.eval()
        
        data_files_val = glob.glob(os.path.join(args.data_folder_val, '*.h5'))
        num_cases_val = len(data_files_val)

        idx = np.arange(num_cases_val)
        for cs in range(num_cases_val):
            i = int(idx[cs])
            file_val = data_files_val[i]
            logging.info(f'val file {file_val}')
            with h5py.File(file_val, 'r') as hf:
                truth = hf['Images_real'][()] + 1j * hf['Images_imag'][()]
                smaps = hf['Maps_real'][()] + 1j * hf['Maps_imag'][()]
            num_coils = smaps.shape[0]
    
            truth_tensor = torch.tensor(truth, dtype=torch.complex64)
            truth_tensor = truth_tensor.cuda()
            truth_tensor.requires_grad = False
            truth_tensor = torch.rot90(truth_tensor, k=1, dims=(0, 1))
            
            smaps_tensor = torch.from_numpy(smaps).type(torch.complex64)
            smaps_tensor.requires_grad = False
            smaps_tensor = torch.rot90(smaps_tensor, k=1, dims=(1, 2))
    
            kyzc = torch.stack((kyc_clamp, kzc_clamp), axis=1)
            coords_centers = torch.stack((kxc, kyc_clamp, kzc_clamp), axis=1)
            traj_torch = torch.from_numpy(traj_xread.copy()).cuda()
    
            coord = torch.zeros(traj_xread.shape + (num_views,), dtype=torch.float32).cuda()
            for v in range(num_views):
                coord[..., v] = traj_torch + coords_centers[v].unsqueeze(0)
            coord = torch.permute(coord, (0, -1, 1))
            coord = torch.reshape(coord, (-1, NDIM))
    
            y = nufft_forward(coord, truth_tensor, smaps_tensor)
            im = nufft_adjoint(coord, y, smaps_tensor)
            if args.init_zero:
                im = torch.zeros_like(im)
    
            for u in range(args.unroll):
                Ex = nufft_forward(coord, im, smaps_tensor)
    
                Exy = Ex - y  # Ex-d
    
                " Data consistency step "
                im = im - nufft_adjoint(coord, Exy, smaps_tensor) * scale_global
    
                " denoiser "
                im = denoiser_bw(im[None, None])
                im = im.squeeze()
                torch.cuda.empty_cache()
            
            lossMSE_val = (im - truth_tensor).abs().pow(2).sum().pow(0.5) / truth_tensor.abs().pow(2).sum().pow(0.5)
    
            ########################################### plot and save #################################################
            if epoch % 10 == 0 and cs == 0:
                with h5py.File(evalimagesfile, 'a') as hf:
                    if epoch == 0:
                        hf.create_dataset(f"eval_truth_mag", data=np.squeeze(np.abs(truth_tensor.cpu().numpy())))
                    hf.create_dataset(f"eval_im_epoch{epoch}_mag", data=np.squeeze(np.abs(im.detach().cpu().numpy())))
            ########################################### plot and save #################################################
    
            del im, y
            torch.cuda.empty_cache()
    
        eval_avg.update(lossMSE_val.detach().item())
        writer_val.add_scalar('LossMSE', eval_avg.avg(), epoch)
