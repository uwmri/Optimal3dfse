"""
Experiment with N4ITK bias field correction on train_mag_biasCorr from an h5 file.
Tune parameters and compare results interactively.
"""

import numpy as np
import h5py
import SimpleITK as sitk
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from src.utils import run_n4

# ── Config ─────────────────────────────────────────────────────────────────────
h5_file = r'S:\Opt3dfse_ADRC_add_on\DAS_09413_2025-02-19\09413_00011_Optimal_3DFLAIR_c\raw_data\recon_denoiserTrue_phasefittingTrue_kwFalse_rh26.h5'
dataset  = 'train_mag_biasCorr'      # input dataset in the h5

# ── N4 parameters to tune ──────────────────────────────────────────────────────
n4_params = dict(
    num_histogram_bins    = 200,      # try 100–400
    fitting_levels        = 5,        # try 3–6; more levels = captures larger-scale bias
    max_iter_per_level    = 50,       # iterations at each resolution level
    convergence_threshold = 0.001,
    shrink_factor         = 4,        # shrink before masking (speeds up Otsu)
    n_passes              = 1,        # set to 2 to run N4 twice (iterative)
)

# ── Visualization ──────────────────────────────────────────────────────────────
# Pick slice indices to display along the chosen axis
display_slices = [60, 80, 100]       # adjust to your data range
axis = 0                             # 0=first dim, 1=second dim, 2=third dim
clim = (0, None)                     # (vmin, vmax); None = auto 99th percentile

# ──────────────────────────────────────────────────────────────────────────────





def show_comparison(original, corrected, bias_field, slices, ax):
    n = len(slices)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    fig.suptitle('Top: original  |  Middle: corrected  |  Bottom: bias field', fontsize=12)

    vmax = np.percentile(original, 99) if clim[1] is None else clim[1]

    for col, sl in enumerate(slices):
        orig_sl = np.take(original,    sl, axis=ax)
        corr_sl = np.take(corrected,   sl, axis=ax)
        bf_sl   = np.take(bias_field,  sl, axis=ax)

        axes[0, col].imshow(orig_sl, cmap='gray', vmin=clim[0], vmax=vmax)
        axes[0, col].set_title(f'slice {sl}')
        axes[0, col].axis('off')

        axes[1, col].imshow(corr_sl, cmap='gray', vmin=clim[0], vmax=vmax)
        axes[1, col].axis('off')

        im = axes[2, col].imshow(bf_sl, cmap='hot')
        axes[2, col].set_title(f'bias {bf_sl.min():.2f}–{bf_sl.max():.2f}')
        axes[2, col].axis('off')
        plt.colorbar(im, ax=axes[2, col], fraction=0.046)

    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
print(f'Loading "{dataset}" from h5...')
with h5py.File(h5_file, 'r') as hf:
    print('Datasets in file:', list(hf.keys()))
    original = hf[dataset][()].astype(np.float32)

print(f'Volume shape: {original.shape}')
print('Running N4 with params:', n4_params)

corrected, bias_field = run_n4(original, n4_params)

print(f'Bias field range: {bias_field.min():.3f} – {bias_field.max():.3f}')
print(f'Correction ratio: {bias_field.max() / bias_field.min():.2f}x')

show_comparison(original, corrected, bias_field, display_slices, axis)

# ── Save result ────────────────────────────────────────────────────────────────
with h5py.File(h5_file, 'a') as hf:
    if 'train_mag_biasCorr2' in hf:
        del hf['train_mag_biasCorr2']
    hf.create_dataset('train_mag_biasCorr2', data=corrected)
print('Saved train_mag_biasCorr2 to', h5_file)
