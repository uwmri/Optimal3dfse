import argparse
import csv
import fnmatch
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F


from estimate_motion import estimate_mask, register_images


def move_images(image, phi, theta, psi, tx, ty, tz):
    """
    Moving image according to registration params
    Args:
        image: [Nz, Ny, Nx]
    """
    rot = torch.zeros(3, 4, dtype=image.dtype, device=image.device)
    rot[0, 0] = torch.cos(theta) * torch.cos(psi)
    rot[0, 1] = -torch.cos(phi) * torch.sin(psi) + torch.sin(phi) * torch.sin(theta) * torch.cos(psi)
    rot[0, 2] = torch.sin(phi) * torch.sin(psi) + torch.cos(phi) * torch.sin(theta) * torch.cos(psi)

    rot[1, 0] = torch.cos(theta) * torch.sin(psi)
    rot[1, 1] = torch.cos(phi) * torch.cos(psi) + torch.sin(phi) * torch.sin(theta) * torch.sin(psi)
    rot[1, 2] = -torch.sin(phi) * torch.cos(psi) + torch.cos(phi) * torch.sin(theta) * torch.sin(psi)

    rot[2, 0] = -torch.sin(theta)
    rot[2, 1] = torch.sin(phi) * torch.cos(theta)
    rot[2, 2] = torch.cos(phi) * torch.cos(theta)

    rot[0, 3] = tx
    rot[1, 3] = ty
    rot[2, 3] = tz

    theta = rot.view(-1, 3, 4)
    image = image.view(-1, 1, image.shape[-3], image.shape[-2], image.shape[-1])

    # affine grid uses matrices from -1 to 1 along each dimension
    grid = F.affine_grid(theta, image.size(), align_corners=False)
    registered = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return registered


def find_case_dirs(root_dir: str) -> List[Tuple[str, str]]:
    results = []
    if not os.path.isdir(root_dir):
        return results
    for name in sorted(os.listdir(root_dir)):
        case_dir = os.path.join(root_dir, name)
        if not os.path.isdir(case_dir):
            continue
        results.append((name, case_dir))
    return results


def find_subdir(parent_dir: str, pattern: str) -> Optional[str]:
    if not parent_dir or not os.path.isdir(parent_dir):
        return None
    for name in sorted(os.listdir(parent_dir)):
        if fnmatch.fnmatch(name, pattern) and os.path.isdir(os.path.join(parent_dir, name)):
            return os.path.join(parent_dir, name)
    return None


def find_h5_file(folder: str, pattern: str) -> Optional[str]:
    if not folder or not os.path.isdir(folder):
        return None
    matches = sorted([os.path.join(folder, f) for f in os.listdir(folder) if fnmatch.fnmatch(f, pattern)])
    return matches[0] if matches else None


def load_h5_dataset(h5_path: str, dataset_name: str) -> np.ndarray:
    print(h5_path)
    with h5py.File(h5_path, "r") as hf:
        data = hf[dataset_name][()]
    return np.array(data, dtype=np.float32).squeeze()

def _data_range(a, b):
    a_min = float(np.min(a))
    a_max = float(np.max(a))
    b_min = float(np.min(b))
    b_max = float(np.max(b))
    dr = max(a_max, b_max) - min(a_min, b_min)
    if dr == 0.0:
        dr = 1.0
    return dr

def compute_psnr(fixed: np.ndarray, moving: np.ndarray) -> float:
    data_range = _data_range(fixed, moving)
    mse = float(np.mean((fixed - moving) ** 2))
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(data_range / np.sqrt(mse)))


def compute_ssim_volume(fixed: np.ndarray, moving: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:
        raise RuntimeError("scikit-image is required for SSIM. Install with `pip install scikit-image`.") from exc

    if fixed.shape != moving.shape:
        raise ValueError("Fixed and moving volumes must have the same shape for SSIM.")

    data_range = _data_range(fixed, moving)
    print(f'ssim calc: data_range={data_range}')
    if data_range == 0:
        return 1.0 if np.allclose(fixed, moving) else 0.0

    # Compute SSIM over the full 3D volume (N-D SSIM)
    return float(structural_similarity(fixed, moving, data_range=data_range))


def register_and_metrics(moving: np.ndarray, fixed: np.ndarray, logdir: str) -> Tuple[float, float]:
    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)

    # scaling
    scale = np.sum(moving * fixed) / np.sum(moving * moving)
    moving *= scale

    # Histogram match moving -> fixed to align intensity distributions
    try:
        from skimage.exposure import match_histograms
    except Exception as exc:
        raise RuntimeError("scikit-image is required for histogram matching. Install with `pip install scikit-image`.") from exc
    moving = match_histograms(moving, fixed)

    fixed_max = np.max(np.abs(fixed)) + 1e-12
    moving_max = np.max(np.abs(moving)) + 1e-12

    # fixed_reg = fixed / fixed_max
    # moving_reg = moving / moving_max

    images = np.stack([fixed, moving], axis=0)
    mask = estimate_mask(images)
    tx, ty, tz, phi, psi, theta = register_images(
        images,
        mask,
        logdir=logdir,
        out_reg_images=True,
        plot_loss=True,
        normalize_out_images=True,
    )

    tx = torch.from_numpy(tx).cuda()
    ty = torch.from_numpy(ty).cuda()
    tz = torch.from_numpy(tz).cuda()
    phi = torch.from_numpy(phi).cuda()
    psi = torch.from_numpy(psi).cuda()
    theta = torch.from_numpy(theta).cuda()

    # fixed_norm = fixed / fixed_max
    # moving_norm = moving / fixed_max

    moving_tensor = torch.from_numpy(moving).cuda()
    moving_registered = move_images(
        moving_tensor, phi[1], theta[1], psi[1], tx[1], ty[1], tz[1]
    ).squeeze().cpu().numpy()

    diff = moving_registered - fixed
    nmse = float(np.sqrt(np.sum(diff ** 2)) / (np.sqrt(np.sum(fixed ** 2)) + 1e-12))
    ssim = compute_ssim_volume(fixed, moving_registered)
    psnr = compute_psnr(fixed, moving_registered)
    return nmse, ssim, psnr


def main():
    parser = argparse.ArgumentParser(description="Register 3DFLAIR to Sag T2 FLAIR CUBE and compute MSE/SSIM.")
    parser.add_argument(
        "--root",
        default=r"S:\Opt3dfse_ADRC_add_on",
        help="Root folder containing case folders with raw_data",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Run only a single case (folder name under root). If omitted, runs all cases.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for registration (register_images uses GPU).")

    cases = find_case_dirs(args.root)
    if not cases:
        print(f"No case folders found under {args.root}")
        return
    if args.case:
        cases = [c for c in cases if c[0] == args.case]
        if not cases:
            print(f"Case {args.case} not found under {args.root}")
            return

    rows_c = []
    rows_no_c = []

    for case_name, case_dir in cases:
        flair_c_dir = find_subdir(case_dir, "*_3DFLAIR_c")
        flair_dir = find_subdir(case_dir, "*_3DFLAIR")
        cube_dir = find_subdir(case_dir, "*Sag_T2_FLAIR_CUBE")

        flair_c_raw = os.path.join(flair_c_dir, "raw_data") if flair_c_dir else None
        flair_raw = os.path.join(flair_dir, "raw_data") if flair_dir else None
        cube_raw = os.path.join(cube_dir, "raw_data") if cube_dir else None

        cube_h5 = find_h5_file(cube_raw, "recon_l1w_z216*.h5") if cube_raw else None

        if cube_h5:
            cube_mag = load_h5_dataset(cube_h5, "/Images_bc")
        else:
            cube_mag = None

        if cube_mag is None:
            print(f"[{case_name}] Missing CUBE data, skipping case.")
            continue

        # 3DFLAIR_c -> CUBE
        if flair_c_dir:
            flair_c_h5 = find_h5_file(flair_c_raw, "recon_denoiserTrue_phasefittingTrue_kwFalse*rh26.h5") if flair_c_raw else None
            if flair_c_h5:
                flair_c_mag = load_h5_dataset(flair_c_h5, "/train_mag_biasCorr")
                nmse, ssim, psnr = register_and_metrics(flair_c_mag, cube_mag, logdir=flair_c_dir)
                rows_c.append(
                    {
                        "case": case_name,
                        "nmse": nmse,
                        "ssim": ssim,
                        "psnr": psnr,
                    }
                )
                print(f'loaded {flair_c_h5}')
            else:
                print(f"[{case_name}] Missing recon_denoiserTrue_phasefittingTrue_kwFalse*.h5 in 3DFLAIR_c.")
        else:
            print(f"[{case_name}] Missing *_3DFLAIR_c folder.")

        # 3DFLAIR (no c) -> CUBE
        if flair_dir:
            flair_h5 = find_h5_file(flair_raw, "recon_denoiserTrue_phasefittingTrue_kwFalse*rh26.h5") if flair_raw else None
            if flair_h5:
                flair_mag = load_h5_dataset(flair_h5, "/train_mag_biasCorr")
                nmse, ssim, psnr = register_and_metrics(flair_mag, cube_mag, logdir=flair_dir)
                rows_no_c.append(
                    {
                        "case": case_name,
                        "nmse": nmse,
                        "ssim": ssim,
                        "psnr": psnr,
                    }
                )
                print(f'loaded {flair_h5}')

            else:
                print(f"[{case_name}] Missing recon_denoiserTrue_phasefittingTrue_kwFalse*.h5 in 3DFLAIR.")
        else:
            print(f"[{case_name}] Missing *_3DFLAIR folder.")

    # Write CSVs
    out_c = os.path.join(args.root, "pyregistered_metrics_CUBE-c.csv")
    out_no_c = os.path.join(args.root, "pyregistered_metrics_CUBE-no-c.csv")

    if rows_c:
        with open(out_c, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "nmse", "ssim", "psnr"])
            writer.writeheader()
            writer.writerows(rows_c)
        print(f"Wrote {len(rows_c)} rows to {out_c}")
    else:
        print("No CUBE-c rows to write.")

    if rows_no_c:
        with open(out_no_c, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "nmse", "ssim", "psnr"])
            writer.writeheader()
            writer.writerows(rows_no_c)
        print(f"Wrote {len(rows_no_c)} rows to {out_no_c}")
    else:
        print("No CUBE-no-c rows to write.")


if __name__ == "__main__":
    main()
