#!/bin/bash
# Run recon.py on all *_3DFLAIR_c/raw_data and *_3DFLAIR/raw_data
# across S:\Opt3dfse_ADRC_add_on and D:\SSD\Data\Scans_i\Opt3dfse_ADRC_add_on

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BASE_DIRS=(
    "/s/Opt3dfse_ADRC_add_on"
    "/d/SSD/Data/Scans_i/Opt3dfse_ADRC_add_on"
)

# Collect all raw_data directories (both _3DFLAIR_c and _3DFLAIR)
SCAN_DIRS=()
for base_dir in "${BASE_DIRS[@]}"; do
    if [ ! -d "$base_dir" ]; then
        echo "Skipping missing directory: $base_dir"
        continue
    fi
    while IFS= read -r -d '' dir; do
        SCAN_DIRS+=("$(cygpath -w "$dir")")
    done < <(find "$base_dir" -mindepth 3 -maxdepth 3 -type d -name "raw_data" \
        \( -path "*_3DFLAIR_c/raw_data" -o -path "*_3DFLAIR/raw_data" \) -print0 | sort -z)
done

if [ ${#SCAN_DIRS[@]} -eq 0 ]; then
    echo "No scan directories found."
    exit 1
fi

echo "Found ${#SCAN_DIRS[@]} scan directories:"
printf '  %s\n' "${SCAN_DIRS[@]}"
echo ""

cd "$SCRIPT_DIR/src"
python recon.py "${SCAN_DIRS[@]}"
