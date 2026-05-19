#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "./run_dolfinx.sh" ]]; then
    echo "Error: run_dolfinx.sh not found. Run this script from the repo root." >&2
    exit 1
fi

mkdir -p results

cases=(
    "20   1   4"
    "50   1   8"
    "100  2   15"
    "200  2   20"
    "200  4   40"
    "300  4   60"
    "500  4   80"
    "500  8   160"
    "800  8   160"
    "1000 10  320"
)

echo "============================================================"
echo "  Point-charge mesh density sweep — 10 cases"
echo "  Results → results/sweep.csv"
echo "============================================================"

total=${#cases[@]}

for i in "${!cases[@]}"; do
    ID=$(( i + 1 ))
    read -r L H_FINE H_COARSE <<< "${cases[$i]}"

    echo "--- Case ${ID}/${total}: L=${L}nm h_fine=${H_FINE}nm h_coarse=${H_COARSE}nm ---"

    SECONDS=0

    ./run_dolfinx.sh src/verify/point_charge_sweep.py \
        --L_nm "$L" --h_fine_nm "$H_FINE" --h_coarse_nm "$H_COARSE" \
        --case_id "$ID" --out results/sweep.csv

    echo "Elapsed: ${SECONDS}s"
done

echo "============================================================"
echo "All cases done. Results in results/sweep.csv"
echo "To view: column -t -s, results/sweep.csv"
echo "============================================================"
