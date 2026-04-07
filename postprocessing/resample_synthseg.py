"""
Resample all *_synthseg.nii files back to their corresponding input image space.

SynthSeg outputs 1mm isotropic images; this restores the original shape/affine.

Usage:
    python resample_synthseg.py            # skip already-resampled files
    python resample_synthseg.py --force    # overwrite existing resampled files
"""

import re
import sys
from pathlib import Path

import nibabel as nib
from nilearn.image import resample_img

DATA_DIRS = [
    Path(r"D:\SSD\Data\Scans_i\Opt3dfse_ADRC_add_on"),
    Path(r"S:\Opt3dfse_ADRC_add_on"),
]

INPUT_FILES = [
    # "opt3DFLAIR_c_scaled.nii",
    # "opt3DFLAIR_scaled.nii",
    # "CUBE_scaled.nii",
    "opt3DFLAIR_l1w_scaled.nii",
]

CASE_PATTERN = re.compile(r"^[A-Z_]+_\d{5}_\d{4}-\d{2}-\d{2}$")


def find_case_dirs():
    case_dirs = []
    for base_dir in DATA_DIRS:
        if not base_dir.exists():
            print(f"Warning: {base_dir} does not exist, skipping.")
            continue
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and CASE_PATTERN.match(d.name):
                case_dirs.append(d)
    return case_dirs


def resample_seg(input_path: Path, seg_path: Path, out_path: Path, force: bool = False):
    if out_path.exists() and not force:
        print(f"    [skip] {out_path.name} already exists.")
        return True

    ref = nib.load(str(input_path))
    seg = nib.load(str(seg_path))

    seg_resampled = resample_img(
        seg,
        target_affine=ref.affine,
        target_shape=ref.shape[:3],
        interpolation="nearest",
    )

    nib.save(seg_resampled, str(out_path))
    print(f"    Resampled: {seg_path.name} -> {out_path.name}")
    return True


def main():
    force = "--force" in sys.argv
    case_dirs = find_case_dirs()
    print(f"Found {len(case_dirs)} case directories.\n")

    total, done = 0, 0
    skipped, failed = [], []

    for case_dir in case_dirs:
        print(f"Case: {case_dir.name}")
        for fname in INPUT_FILES:
            input_path = case_dir / fname
            stem = fname.replace(".nii.gz", "").replace(".nii", "")
            seg_path = case_dir / f"{stem}_synthseg.nii"
            out_path = case_dir / f"{stem}_synthseg_r.nii"

            if not seg_path.exists():
                print(f"    [skip] {seg_path.name} not found.")
                continue

            if not input_path.exists():
                print(f"    [skip] input {fname} not found, cannot resample.")
                skipped.append(str(seg_path))
                continue

            total += 1
            try:
                if resample_seg(input_path, seg_path, out_path, force=force):
                    done += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                failed.append(str(seg_path))

    print(f"\n{'='*60}")
    print(f"Completed: {done}/{total}")
    if skipped:
        print(f"\nSkipped (missing input, {len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
