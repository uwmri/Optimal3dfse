import os
import logging
from math import ceil

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import h5py
from sigpy import backend
from sigpy.fourier import _scale_coord, _apodize, _get_oversamp_shape
import SimpleITK as sitk

def pca_coil_compression(kdata=None, axis=0, target_channels=None):
    logger = logging.getLogger('PCA_CoilCompression')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')
        kdata_cc = kdata[0]
    else:
        kdata_cc = kdata

    logger.info(f'Compressing to {target_channels} channels, along axis {axis}')
    logger.info(f'Initial  size = {kdata_cc.shape} ')

    # Put channel to first axis
    kdata_cc = np.moveaxis(kdata_cc, axis, -1)
    old_channels = kdata_cc.shape[-1]
    logger.info(f'Old channels =  {old_channels} ')

    # Subsample to reduce memory for SVD
    mask_shape = np.array(kdata_cc.shape)
    mask = np.random.choice([True, False], size=mask_shape[:-1], p=[0.05, 1 - 0.05])

    # Create a subsampled array
    kcc = np.zeros((old_channels, np.sum(mask)), dtype=kdata_cc.dtype)
    logger.info(f'Kcc Shape = {kcc.shape} ')
    for c in range(old_channels):
        ktemp = kdata_cc[..., c]
        kcc[c, :] = ktemp[mask]

    kdata_cc = np.moveaxis(kdata_cc, -1, axis)

    #  SVD decomposition
    logger.info(f'Working on SVD of {kcc.shape}')
    u, s, vh = np.linalg.svd(kcc, full_matrices=False)

    logger.info(f'S = {s}')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

        for e in range(len(kdata)):
            kdata[e] = np.moveaxis(kdata[e], axis, -1)
            kdata[e] = np.expand_dims(kdata[e], -1)
            logger.info(f'Shape = {kdata[e].shape}')
            kdata[e] = np.matmul(u, kdata[e])
            kdata[e] = np.squeeze(kdata[e], axis=-1)
            kdata[e] = kdata[e][..., :target_channels]
            kdata[e] = np.moveaxis(kdata[e], -1, axis)

        for ksp in kdata:
            logger.info(f'Final Shape {ksp.shape}')
    else:
        # Now iterate over and multiply by u
        kdata = np.moveaxis(kdata, axis, -1)
        kdata = np.expand_dims(kdata, -1)
        kdata = np.matmul(u, kdata)
        logger.info(f'Shape = {kdata.shape}')

        # Crop to target channels
        kdata = np.squeeze(kdata, axis=-1)
        kdata = kdata[..., :target_channels]

        # Put back
        kdata = np.moveaxis(kdata, -1, axis)
        logger.info(f'Final shape = {kdata.shape}')

    return kdata


def get_outer(coord, values_flat, percent=1, plot=False):
    """ get the outer N percent of kdata based on normalized distance to the center
        coords: (Readout * # PE points, 3)
        values_flat: single coil kdata (Readout * # PE points,)
        y_thresh: float, vacinity w
    """
    coords_flat = coord.get()
    # Estimate ellipsoid center and semi-axis lengths
    center = coords_flat.mean(axis=0)
    ranges = coords_flat.max(axis=0) - coords_flat.min(axis=0)
    semi_axes = ranges / 2

    normalized_coords = (coords_flat - center) / semi_axes
    r_squared = np.sum(normalized_coords ** 2, axis=1)

    # Find threshold for outermost 5%
    threshold = np.percentile(r_squared, 100-percent)

    # Mask for outermost shell
    outer_mask = r_squared >= threshold
    outer_values = values_flat[outer_mask]
    outer_coords = coords_flat[outer_mask]
    if plot:
        outer_colors = np.log(np.abs(outer_values) + 1e-10)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(outer_coords[:, 0], outer_coords[:, 1], outer_coords[:, 2],
                        c=outer_colors, cmap='viridis', s=1)
        plt.colorbar(sc, label='log(abs(value))')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Outermost 5% Shell of Ellipsoidal Point Cloud')
        plt.tight_layout()
        plt.show()

    return  outer_values


def run_n4(image_np: np.ndarray, params: dict):
    """Run N4 bias correction; returns (corrected, bias_field) as numpy arrays."""
    sitk_image = sitk.GetImageFromArray(image_np.astype(np.float32))

    # Build Otsu mask on a shrunken copy for speed
    sf = params['shrink_factor']
    shrunk = sitk.Shrink(sitk_image, [sf] * sitk_image.GetDimension())
    mask   = sitk.OtsuThreshold(shrunk, 0, 1, 200)
    mask   = sitk.Resample(mask, sitk_image,
                           interpolator=sitk.sitkNearestNeighbor)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfHistogramBins(params['num_histogram_bins'])
    corrector.SetMaximumNumberOfIterations(
        [params['max_iter_per_level']] * params['fitting_levels'])  # list length = number of levels
    corrector.SetConvergenceThreshold(params['convergence_threshold'])

    corrected = sitk_image 
    for _ in range(params['n_passes']):
        corrected = corrector.Execute(corrected, mask)

    log_bias   = corrector.GetLogBiasFieldAsImage(sitk_image)
    bias_field = sitk.GetArrayFromImage(sitk.Exp(log_bias))

    return sitk.GetArrayFromImage(corrected), bias_field