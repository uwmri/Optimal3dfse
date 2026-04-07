# Convert recon-ed h5 files to NIfTI with N4 bias correction, reorientation, and scaling.
# Supports multiple item types; use --item to select which to process.

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk


CASE_RE = re.compile(r".+_\d{5}_\d{4}-\d{2}-\d{2}$")

# N4 params used for LST items (opt3dflair_c, opt3dflair, cube).
# Tuned for MODL/denoiser recons; do not use for l1w.
N4_PARAMS_LST = dict(
    num_histogram_bins    = 200,
    fitting_levels        = 4,
    max_iter_per_level    = 50,
    convergence_threshold = 0.001,
    shrink_factor         = 4,
    n_passes              = 1,
)

# SimpleITK N4BiasFieldCorrectionImageFilter default values.
# Used for l1w recon where LST params are too aggressive.
N4_PARAMS_DEFAULT = dict(
    num_histogram_bins    = 200,
    fitting_levels        = 4,
    max_iter_per_level    = 50,
    convergence_threshold = 0.001,
    shrink_factor         = 4,
    n_passes              = 1,
)

# Item registry: key -> processing configuration
ITEM_CONFIGS: dict[str, dict] = {
    "opt3dflair_c": dict(
        out_name      = "opt3DFLAIR_c_scaled.nii",
        subdir_suffix = "_3DFLAIR_c",
        h5_pattern    = "recon_denoiserTrue_phasefittingTrue_kwFalse*rh26.h5",
        dataset       = "/train_mag_biasCorr",
        n4_params     = N4_PARAMS_LST,
    ),
    "opt3dflair": dict(
        out_name      = "opt3DFLAIR_scaled.nii",
        subdir_suffix = "_3DFLAIR",
        h5_pattern    = "recon_denoiserTrue_phasefittingTrue_kwFalse*rh26.h5",
        dataset       = "/train_mag_biasCorr",
        n4_params     = N4_PARAMS_LST,
    ),
    "cube": dict(
        out_name      = "CUBE_scaled.nii",
        subdir_suffix = "Sag_T2_FLAIR_CUBE",
        h5_pattern    = "recon_l1w_z216*.h5",
        dataset       = "/Images_bc",
        n4_params     = N4_PARAMS_LST,
    ),
    "l1w": dict(
        out_name      = "opt3DFLAIR_l1w_scaled.nii",
        subdir_suffix = "_3DFLAIR",
        h5_pattern    = "recon_l1w_pseudorepFalse.h5",
        dataset       = "/l1w_mag",
        n4_params     = N4_PARAMS_DEFAULT,
    ),
}

ALL_ITEMS = list(ITEM_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_case_dirs(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and CASE_RE.match(p.name)],
        key=lambda p: p.name,
    )


def _find_subdir(case_dir: Path, suffix: str) -> Path | None:
    for child in case_dir.iterdir():
        if child.is_dir() and child.name.endswith(suffix):
            return child
    return None


def _find_h5(h5_dir: Path, pattern: str) -> Path | None:
    matches = sorted(h5_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        logging.warning("Multiple matches for %s in %s; using %s", pattern, h5_dir, matches[0].name)
    return matches[0]


def _load_h5_dataset(h5_path: Path, dataset_name: str) -> np.ndarray | None:
    try:
        with h5py.File(h5_path, "r") as f:
            if dataset_name not in f:
                logging.error("Missing dataset %s in %s", dataset_name, h5_path)
                return None
            data = f[dataset_name][()]
    except OSError as exc:
        logging.error("Failed to read %s: %s", h5_path, exc)
        return None
    return np.asarray(data)


def _apply_n4_bias_correction(data: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply N4 bias field correction to remove intensity inhomogeneity."""
    sitk_image = sitk.GetImageFromArray(data.astype(np.float32))

    sf = params['shrink_factor']
    shrunk = sitk.Shrink(sitk_image, [sf] * sitk_image.GetDimension())
    mask   = sitk.OtsuThreshold(shrunk, 0, 1, 200)
    mask   = sitk.Resample(mask, sitk_image, interpolator=sitk.sitkNearestNeighbor)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfHistogramBins(params['num_histogram_bins'])
    corrector.SetMaximumNumberOfIterations(
        [params['max_iter_per_level']] * params['fitting_levels'])
    corrector.SetConvergenceThreshold(params['convergence_threshold'])

    corrected = sitk_image
    for _ in range(params['n_passes']):
        corrected = corrector.Execute(corrected, mask)

    log_bias   = corrector.GetLogBiasFieldAsImage(sitk_image)
    bias_field = sitk.GetArrayFromImage(sitk.Exp(log_bias))

    return sitk.GetArrayFromImage(corrected), bias_field


def _reorient_and_scale(
    data: np.ndarray,
    params: dict,
    ref_img: nib.Nifti1Image,
) -> tuple[np.ndarray, nib.Nifti1Header]:
    """N4 correct, reorient to axial, flip z, and intensity-scale to reference."""
    logging.info("Applying N4 bias field correction...")
    data, _ = _apply_n4_bias_correction(data, params)

    # (0,1,2) -> (0,2,1): move axial axis to position 2
    data = np.transpose(data, (0, 2, 1))
    # Rotate 90° clockwise in axial plane
    data = np.rot90(data, k=-1, axes=(0, 1))
    # Flip z-axis
    data = data[:, :, ::-1]

    # Intensity scale to match reference
    ref_data = ref_img.get_fdata()
    idxL     = (data.shape[0] - ref_data.shape[0]) // 2
    data_crop = data[idxL:idxL + ref_data.shape[0]]
    scale     = np.sum(data_crop * ref_data) / np.sum(data_crop * data_crop)
    data     *= scale
    logging.info("Applied scale factor: %.4f", scale)

    return data, ref_img.header


def _save_nifti(
    data: np.ndarray,
    out_path: Path,
    header: nib.Nifti1Header | None = None,
    affine: np.ndarray | None = None,
) -> bool:
    if header is None:
        header = nib.Nifti1Header()
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, out_path)
    logging.info("Wrote %s", out_path)
    return True


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(
    case_dir: Path,
    item_keys: list[str],
    ref_img: nib.Nifti1Image,
    overwrite: bool,
    dry_run: bool,
) -> None:
    for key in item_keys:
        cfg      = ITEM_CONFIGS[key]
        out_path = case_dir / cfg["out_name"]

        if out_path.exists() and not overwrite:
            logging.info("Skipping %s (already exists)", out_path)
            continue

        subdir = _find_subdir(case_dir, cfg["subdir_suffix"])
        if subdir is None:
            logging.warning("Missing subdir (*%s) for item '%s' in %s", cfg["subdir_suffix"], key, case_dir)
            continue

        h5_root = subdir / "raw_data" if (subdir / "raw_data").is_dir() else subdir
        h5_path = _find_h5(h5_root, cfg["h5_pattern"])
        if h5_path is None:
            logging.warning("Missing H5 (%s) for item '%s' in %s", cfg["h5_pattern"], key, h5_root)
            continue

        if dry_run:
            logging.info("Dry run: would read %s:%s -> %s", h5_path, cfg["dataset"], out_path)
            continue

        data = _load_h5_dataset(h5_path, cfg["dataset"])
        if data is None:
            continue

        data, header = _reorient_and_scale(data.astype(np.float32), cfg["n4_params"], ref_img)
        _save_nifti(data, out_path, header=header, affine=ref_img.affine)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Opt3dfse ADRC add-on H5 arrays to NIfTI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available items: {', '.join(ALL_ITEMS)}",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"S:\Opt3dfse_ADRC_add_on"),
        help="Root folder containing case directories. ",
    )
    parser.add_argument(
        "--case",
        type=str,
        nargs="+",
        default=None,
        help="Process only these cases (e.g., --case VLP_09705_2025-04-24 DAS_09413_2025-02-19)",
    )
    parser.add_argument(
        "--item",
        type=str,
        nargs="+",
        choices=ALL_ITEMS,
        default=ALL_ITEMS,
        metavar="ITEM",
        help=f"Which item(s) to process. Choices: {', '.join(ALL_ITEMS)}. Default: all.",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path(r"S:\code\2Dsampling\4_t2-flair_adni.nii"),
        help="Reference NIfTI for header and scaling.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing NIfTI files.")
    parser.add_argument("--dry-run", action="store_true", help="List conversions without writing files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = args.root
    if not root.exists():
        logging.error("Root not found: %s", root)
        return 1

    ref_path = args.ref
    if not ref_path.exists():
        logging.error("Reference not found: %s", ref_path)
        return 1

    ref_img = nib.load(ref_path)
    logging.info("Loaded reference: %s", ref_path)
    logging.info("Processing items: %s", args.item)

    cases = _find_case_dirs(root)
    if args.case:
        wanted = set(args.case)
        cases  = [c for c in cases if c.name in wanted]
        if not cases:
            logging.error("No cases found for: %s", args.case)
            return 1

    if not cases:
        logging.warning("No case directories found in %s", root)
        return 0

    for case_dir in cases:
        process_case(case_dir, args.item, ref_img, overwrite=args.overwrite, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
