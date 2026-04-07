"""
Run SynthSeg segmentation on opt3DFLAIR_c_scaled.nii, opt3DFLAIR_scaled.nii,
and CUBE_scaled.nii for all case folders in both data directories.

Case folder pattern: [CASE]_[5-digit-id]_YYYY-MM-DD

Usage:
    python run_synthseg.py            # skip already-segmented files
    python run_synthseg.py --force    # re-run even if output exists
"""

import re
import sys
import subprocess
from pathlib import Path

DATA_DIRS = [
    Path(r"D:\SSD\Data\Scans_i\Opt3dfse_ADRC_add_on"),
    Path(r"S:\Opt3dfse_ADRC_add_on"),
]

# INPUT_FILES = [
#     "opt3DFLAIR_c_scaled.nii",
#     "opt3DFLAIR_scaled.nii",
#     "CUBE_scaled.nii",
#     "opt3DFLAIR_l1w_scaled.nii",
# ]
INPUT_FILES = [
    "opt3DFLAIR_l1w_scaled.nii",
]

CASE_PATTERN = re.compile(r"^[A-Z_]+_\d{5}_\d{4}-\d{2}-\d{2}$")

SYNTHSEG_SCRIPT = Path(r"S:\code\SynthSeg\scripts\commands\SynthSeg_predict.py")


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


def run_synthseg(input_path: Path, output_path: Path, force: bool = False) -> bool:
    if output_path.exists() and not force:
        print(f"    [skip] {output_path.name} already exists.")
        return True

    cmd = [
        sys.executable,
        str(SYNTHSEG_SCRIPT),
        "--i", str(input_path),
        "--o", str(output_path),
        "--robust",
    ]

    print(f"    Running SynthSeg: {input_path.name} -> {output_path.name}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"    ERROR: SynthSeg failed for {input_path}")
        return False
    return True


def main():
    force = "--force" in sys.argv
    case_dirs = find_case_dirs()
    print(f"Found {len(case_dirs)} case directories.\n")

    total, done = 0, 0
    failed = []

    for case_dir in case_dirs:
        print(f"Case: {case_dir.name}")
        for fname in INPUT_FILES:
            input_path = case_dir / fname
            if not input_path.exists():
                print(f"    [skip] {fname} not found.")
                continue

            stem = fname.replace(".nii.gz", "").replace(".nii", "")
            output_path = case_dir / f"{stem}_synthseg.nii"

            total += 1
            if run_synthseg(input_path, output_path, force=force):
                done += 1
            else:
                failed.append(str(input_path))

    print(f"\n{'='*60}")
    print(f"Completed: {done}/{total}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
