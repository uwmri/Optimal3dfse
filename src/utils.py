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

def pca_coil_compression(kdata=None, axis=0, target_channels=None, sample_frac=0.05, seed=0):

    """Compress coils to their ``target_channels`` principal components.

    Args:
        kdata: array with coils along ``axis``, or a list of such arrays (one
            per encoding). For a list, the compression matrix is estimated from
            entry 0 and applied to every entry, so all encodings share virtual
            coils.
        target_channels: channels to keep (all, reordered by energy, if None).
        sample_frac: fraction of samples used to estimate the coil covariance.

    Returns:
        The same structure with ``target_channels`` along ``axis``.
    """

    logger = logging.getLogger('PCA_CoilCompression')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

    ref = kdata[0] if isinstance(kdata, list) else kdata    # ref (C, ...)
    ref = np.moveaxis(ref, axis, -1)                        # ref (..., C)
    num_coils = ref.shape[-1]

    # Subsample to reduce memory for SVD
    rng = np.random.default_rng(seed)
    mask = rng.random(ref.shape[:-1]) < sample_frac

    kcc = ref[mask].T  # (C, # of true values, flattened)
    logger.info(f"PCA coil compression {num_coils} -> {target_channels} "
                f"from {kcc.shape[1]} samples")
    u, s, _ = np.linalg.svd(kcc, full_matrices=False)       # u (C, C), s (C,), Vh (C, n_samples)
    logger.info(f"singular values: {s}")
    proj = u.conj().T[:target_channels]  # (target, C)

    def apply(k):
        k = np.moveaxis(k, axis, -1)                        # k (..., C)
        k = k @ proj.T                                      # k (..., target_channels)
        return np.moveaxis(k, -1, axis)                     # k with target_channels along original axis

    if isinstance(kdata, list):
        return [apply(k) for k in kdata]
    return apply(kdata)


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