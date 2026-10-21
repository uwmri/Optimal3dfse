import logging

import numpy as np


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
