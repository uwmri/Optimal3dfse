"""
Estimate rigid motion between scans
code from https://github.com/uwmri/flow_recon/blob/master/rigid_correction.py
"""
from pathlib import Path
import numpy as np
import h5py
import sigpy as sp
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os

class Rotation3D(nn.Module):
    def __init__(self, number=None):
        super(Rotation3D, self).__init__()

        # Number of output of the localization network (Expected image is frames, number of features)
        self.phi = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.theta = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.psi = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))

    def forward(self):
        rot = torch.zeros(3, 3, device=self.phi.device)
        rot[0, 0] = torch.cos(self.theta) * torch.cos(self.psi)
        rot[0, 1] = -torch.cos(self.phi) * torch.sin(self.psi) + torch.sin(self.phi) * torch.sin(self.theta) * torch.cos(self.psi)
        rot[0, 2] =  torch.sin(self.phi) * torch.sin(self.psi) + torch.cos(self.phi) * torch.sin( self.theta) * torch.cos(self.psi)

        rot[1, 0] = torch.cos(self.theta)*torch.sin(self.psi)
        rot[1, 1] = torch.cos(self.phi) * torch.cos(self.psi) + torch.sin(self.phi) * torch.sin( self.theta) * torch.sin(self.psi)
        rot[1, 2] = -torch.sin(self.phi) * torch.cos(self.psi) + torch.cos(self.phi) * torch.sin(self.theta) * torch.sin( self.psi)

        rot[2, 0] = -torch.sin(self.theta)
        rot[2, 1] = torch.sin( self.phi) * torch.cos( self.theta)
        rot[2, 2] = torch.cos( self.phi ) * torch.cos( self.theta)

        return rot


class RigidRegistration(nn.Module):
    def __init__(self):
        super(RigidRegistration, self).__init__()

        # Number of output of the localization network (Expected image is frames, number of features)
        self.phi = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.theta = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.psi = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.tx = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.ty = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.tz = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))

    def forward(self, images):

        # R_x = torch.Tensor(
        #     [[1, 0, 0], [0, torch.cos(self.phi_x), -torch.sin(self.phi_x)], [0, torch.sin(self.phi_x), torch.cos(self.phi_x)]])
        # R_y = torch.Tensor(
        #     [[torch.cos(self.phi_y), 0, torch.sin(self.phi_y)], [0, 1, 0], [-torch.sin(self.phi_y), 0, torch.cos(self.phi_y)]])
        # R_z = torch.Tensor(
        #     [[torch.cos(self.phi_z), -torch.sin(self.phi_z), 0], [torch.sin(self.phi_z), torch.cos(self.phi_z), 0], [0, 0, 1]])
        #
        # matrix = torch.mm(torch.mm(R_z, R_y), R_x)
        rot = torch.zeros(3, 4, dtype=images.dtype, device=images.device)
        rot[0, 0] = torch.cos(self.theta) * torch.cos(self.psi)
        rot[0, 1] = -torch.cos(self.phi) * torch.sin(self.psi) + torch.sin(self.phi) * torch.sin(self.theta) * torch.cos(self.psi)
        rot[0, 2] =  torch.sin(self.phi) * torch.sin(self.psi) + torch.cos(self.phi) * torch.sin( self.theta) * torch.cos(self.psi)

        rot[1, 0] = torch.cos(self.theta)*torch.sin(self.psi)
        rot[1, 1] = torch.cos(self.phi) * torch.cos(self.psi) + torch.sin(self.phi) * torch.sin( self.theta) * torch.sin(self.psi)
        rot[1, 2] = -torch.sin(self.phi) * torch.cos(self.psi) + torch.cos(self.phi) * torch.sin(self.theta) * torch.sin( self.psi)

        rot[2, 0] = -torch.sin(self.theta)
        rot[2, 1] = torch.sin( self.phi) * torch.cos( self.theta)
        rot[2, 2] = torch.cos( self.phi ) * torch.cos( self.theta)

        #print(rot)

        rot[0, 3] = self.tx
        rot[1, 3] = self.ty
        rot[2, 3] = self.tz

        # reshape into (Nbatch*Nframes)x2x3 affine matrix
        theta = rot.view(-1, 3, 4)
        #print(theta)

        images = images.view(-1,1,images.shape[-3], images.shape[-2], images.shape[-1])

        # Create affine grid from affine transform
        # affine grid uses matrices from -1 to 1 along each dimension
        grid = F.affine_grid(theta, images.size(), align_corners=False)

        # Sample the data on the grid
        registered = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)



         # print(raw_theta)
        return registered


def weighted_mse_loss(input, target, weight):
    # Use weighted loss for masks
    return torch.sum(weight * (input - target) ** 2)


def estimate_mask( images):
    r"""Estimates a mask with the zprofile of exciation

    Args:
        images (array): a 4D array [Nt x Nz x Ny x Nz ] to
    Returns:
        mask (array): a mask broad castable to the image size for weighted mean square error
    """


    # Average images
    avg = np.mean(images, axis=tuple(range(len(images.shape)-3)))
    avg /= np.max(avg)

    # Create a mask for the zprofile
    zprofile = np.max( avg, axis=(-2,-1))
    zprofile_idx = np.nonzero(np.squeeze(zprofile > 0.1))
    start_idx = zprofile_idx[0][0] + 10
    stop_idx = zprofile_idx[0][-1] - 10
    print(f'Start idx = {start_idx} , stop idx {stop_idx}')
    mask = np.zeros_like(avg)
    mask[start_idx:stop_idx,:,:] = 1.0
    return mask

def average_rotation( r=None):
    r"""Finds a rotation to a common space

    Args:
        r (array): a 3D array [Nt x 3 x 3] containing rotation matrices over time
    Returns:
        rp (array): a 3D array [Nt x 3 x 3] containing corrected rotation matrices
    """

    # This just defines a rotation matrix using euler angles
    RotModel = Rotation3D()

    # Optimize using Adam
    optimizer = torch.optim.Adam(RotModel.parameters(), lr=1e-4)

    # Need identify to get loss
    id = torch.eye(3)

    # Get the rotation matrix
    R = torch.tensor(r)

    for epoch in range(100000):

        optimizer.zero_grad()

        # B is the correction to R
        B = RotModel()

        # Calculate the corrected matrix
        y = torch.matmul(B, R)

        # Loss is average difference from identity
        loss = torch.sum(torch.abs(y - id) ** 2)

        if epoch % 1000 == 0:
            print(f'Epoch {epoch} Loss = {loss.item()}')

        loss.backward()
        optimizer.step()

    # Apply rotation to R
    B = RotModel()

    print(f'Inverse Common Rotation')
    print(f'  Psi={RotModel.psi[0,0]} ')
    print(f'  Theta={RotModel.theta[0, 0]} ')
    print(f'  Phi={RotModel.phi[0, 0]} ')
    print(f'  B = {B}')

    return(B.detach().cpu().numpy())

def register_images(images, mask, logdir=None, out_reg_images=False, plot_loss=True, normalize_out_images=False):
    r"""Registers a series of 3D images collected over time using pytorch affine and masked mean square error

    Args:
        images (array): a 4D array [Nt x Nz x Ny x Nz ] to
        mask (array): a mask broad castable to the image size for weighted mean square error
        logdir (path): a folder location to save the data
    Returns:
        tuple containing  tx, ty, tz, phi, psi, theta rigid transforms at each timepoint
    """

    # Ensure mask is a tensor on gpu
    mask = torch.tensor(mask).to('cuda')

    # The model is declared once so that we only need to estimate differences from frame to frame
    model = RigidRegistration()
    model.cuda()

    fixed_image = torch.tensor(images[0]).to('cuda')
    fixed_image = fixed_image.view(-1, 1, fixed_image.shape[-3], fixed_image.shape[-2], fixed_image.shape[-1])
    print(f'Fixed image max {torch.max(fixed_image)}')
    fixed_image /= torch.max(fixed_image)
    print(f'Fixed image shape {fixed_image.shape}')

    # Pad to be square
    max_size = torch.max(torch.tensor(fixed_image.shape))

    pad_amount1 = (max_size - fixed_image.shape[-1]) // 2
    pad_amount2 = (max_size - fixed_image.shape[-2]) // 2
    pad_amount3 = (max_size - fixed_image.shape[-3]) // 2

    pad_f = (pad_amount1, pad_amount1, pad_amount2, pad_amount2, pad_amount3, pad_amount3)
    fixed_image = nn.functional.pad(fixed_image, pad_f)

    mask = nn.functional.pad(mask, pad_f)

    # Register the images
    all_tx = []
    all_ty = []
    all_tz = []
    all_phi = []
    all_psi = []
    all_theta = []
    all_images = [fixed_image.detach().cpu().numpy(),]
    all_moving = [fixed_image.detach().cpu().numpy(),]
    all_loss = []
    all_tx.append(model.tx.detach().cpu().numpy())
    all_ty.append(model.ty.detach().cpu().numpy())
    all_tz.append(model.tz.detach().cpu().numpy())
    all_phi.append(model.phi.detach().cpu().numpy())
    all_psi.append(model.psi.detach().cpu().numpy())
    all_theta.append(model.theta.detach().cpu().numpy())

    for idx in range(1, images.shape[0]):

        print(f'Image {idx} of {images.shape[0]}')

        moving_image = torch.tensor( images[idx] ).to('cuda')
        #print(f'Max moving = {torch.max( moving_image)}')
        moving_image /= torch.max( moving_image)
        moving_image = moving_image.view(-1, 1, moving_image.shape[-3], moving_image.shape[-2], moving_image.shape[-1])

        moving_image = nn.functional.pad(moving_image, pad_f)

        # Get optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        #loss_func = torch.nn.MSELoss()
        loss_func = weighted_mse_loss

        loss_monitor = []
        loss_thresh = 1e-12
        #loss_thresh = 1e-1 # Temp

        loss_window = 20
        #loss_window = 2 # temp

        # Grab the images
        model.train()
        for epoch in range(0, 3000):
            optimizer.zero_grad()

            registered = model(moving_image)
            loss = loss_func( registered, fixed_image, mask)

            # compute gradients and update parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            if epoch ==0:
                loss0 = loss.item()
            loss_monitor.append(loss.item()/(loss0+1e-9))
            if epoch > loss_window:
                dloss = loss_monitor[-loss_window] - loss_monitor[-1]
                if dloss < loss_thresh:
                    break

        print(f'Stop reg : epoch = {epoch} loss = {loss_monitor[-1]}')

        all_images.append( registered.detach().cpu().numpy())
        all_moving.append( moving_image.detach().cpu().numpy())
        all_tx.append(model.tx.detach().cpu().numpy())
        all_ty.append(model.ty.detach().cpu().numpy())
        all_tz.append(model.tz.detach().cpu().numpy())
        all_phi.append(model.phi.detach().cpu().numpy())
        all_psi.append(model.psi.detach().cpu().numpy())
        all_theta.append(model.theta.detach().cpu().numpy())
        all_loss.append(loss_monitor)

    # Export to file
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        out_name = os.path.join(logdir, 'RegisteredImages.h5')
        print('Saving images to ' + out_name)
        try:
            os.remove(out_name)
        except OSError:
            pass
        if out_reg_images and normalize_out_images:
            # Shared min/max across all volumes so window/level matches
            all_concat = np.concatenate([np.ravel(v) for v in (all_images + all_moving)])
            v_min = np.min(all_concat)
            v_max = np.max(all_concat)
            denom = v_max - v_min

            def _normalize_volumes(vols):
                if denom == 0:
                    return vols
                return [(v - v_min) / denom for v in vols]

            all_images_out = _normalize_volumes(all_images)
            all_moving_out = _normalize_volumes(all_moving)
        else:
            all_images_out = all_images
            all_moving_out = all_moving

        with h5py.File(out_name, 'w') as hf:
            if out_reg_images:
                hf.create_dataset("REGISTERED", data=np.squeeze(np.stack(all_images_out)))
                hf.create_dataset("MOVING", data=np.squeeze(np.stack(all_moving_out)))
            hf.create_dataset("phi", data=180.0/math.pi*np.squeeze(np.array(all_phi)))
            hf.create_dataset("psi", data=180.0 / math.pi * np.squeeze(np.array(all_psi)))
            hf.create_dataset("theta", data=180.0 / math.pi * np.squeeze(np.array(all_theta)))
            hf.create_dataset("tx", data=np.squeeze(np.array(all_tx)) * images.shape[-1]/2.0)
            hf.create_dataset("ty", data=np.squeeze(np.array(all_ty)) * images.shape[-2]/2.0)
            hf.create_dataset("tz", data=np.squeeze(np.array(all_tz)) * images.shape[-3]/2.0)

        if plot_loss and all_loss:
            for i, loss_curve in enumerate(all_loss, start=1):
                if not loss_curve:
                    continue
                plt.figure()
                plt.plot(loss_curve)
                plt.xlabel('Epoch')
                plt.ylabel('Normalized Loss')
                plt.title(f'Registration Loss (Image {i})')
                out_plot = os.path.join(logdir, f'RegistrationLoss_image{i}_new.png')
                plt.tight_layout()
                plt.savefig(out_plot)
                plt.close()

    # Cast to np array instead of list
    all_tx = np.squeeze(np.array(all_tx))
    all_ty = np.squeeze(np.array(all_ty))
    all_tz = np.squeeze(np.array(all_tz))
    all_phi = np.squeeze(np.array(all_phi))
    all_psi = np.squeeze(np.array(all_psi))
    all_theta = np.squeeze(np.array(all_theta))

    return all_tx, all_ty, all_tz, all_phi, all_psi, all_theta



if __name__ == "__main__":
    scan_ref = r'I:\Data\Scans_i\Opt3dfse_ADRC_add_on\EDF_09080_2024-11-13\09080_00003_Sag_T2_FLAIR_CUBE\raw_data\recon_l1w_z216_48ch.h5'
    reconfilename = 'recon_denoiserTrue_phasefittingTrue_kwFalse_rh26.h5'
    scandir1 = r'I:\Data\Scans_i\Opt3dfse_ADRC_add_on\EDF_09080_2024-11-13\09080_00011_Optimal_3DFLAIR_c\raw_data'
    scandir2 = r'I:\Data\Scans_i\Opt3dfse_ADRC_add_on\EDF_09080_2024-11-13\09080_00012_Optimal_3DFLAIR\raw_data'
    scanroot = Path(scandir1).parents[1]
    MAG = []

    # reference is product ADRC Sag_T2_FLAIR_CUBE and is stationary
    with h5py.File(scan_ref, 'r') as hf:
        temp = hf['Images_mag'][()].squeeze()
        scale = 1 / np.max(np.abs(temp))
        temp *= scale
        MAG.append(temp)
    with h5py.File(os.path.join(scandir1, reconfilename), 'r') as hf:
        temp = hf['train_mag'][()].squeeze()
        scale = 1 / np.max(np.abs(temp))
        temp *= scale
        MAG.append(temp)
    with h5py.File(os.path.join(scandir2, reconfilename), 'r') as hf:
        temp = hf['train_mag'][()].squeeze()
        scale = 1 / np.max(np.abs(temp))
        temp *= scale
        MAG.append(temp)

    MAG = np.stack(MAG, axis=0)
    mask = estimate_mask(MAG)
    tx, ty, tz, phi, psi, theta, loss = register_images(MAG, mask, logdir=scanroot, out_reg_images=True)
