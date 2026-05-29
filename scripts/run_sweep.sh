#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "./run_dolfinx.sh" ]]; then
    echo "Error: run_dolfinx.sh not found. Run from the repo root." >&2
    exit 1
fi

mkdir -p results

# Fixed domain: 200nm cube, coarse mesh 40nm.
# Fine box: 20nm cube centred at the charge.
# We sweep h_fine from coarse → ultra-fine to see how runtime scales.
#
# Columns: case_id  L_nm  h_fine_nm  h_coarse_nm  fine_box_nm
cases=(
    "1   200  10.0   40  20"
    "2   200   5.0   40  20"
    "3   200   2.0   40  20"
    "4   200   1.0   40  20"
    "5   200   0.5   40  20"
    "6   200   0.25  40  20"
    "7   200   0.1   40  20"
    "8   200   0.5   40   8"
    "9   200   0.25  40   8"
    "10  200   0.1   40   8"
)

echo "================================================================"
echo "  h_fine sweep  |  domain=200nm  |  h_coarse=40nm"
echo "  Goal: see how runtime scales as mesh gets finer"
echo "  Results → results/sweep.csv"
echo "================================================================"
printf "%-6s %-10s %-10s %-10s %-12s\n" \
       "Case" "h_fine(nm)" "box(nm)" "cells" "total_time(s)"

# Clear old results
rm -f results/sweep.csv

total=${#cases[@]}

for entry in "${cases[@]}"; do
    read -r ID L H_FINE H_COARSE BOX_NM <<< "$entry"

    echo ""
    echo "--- Case ${ID}/${total}: h_fine=${H_FINE}nm  fine_box=${BOX_NM}nm ---"
    SECONDS=0

    ./run_dolfinx.sh src/verify/point_charge_sweep.py \
        --L_nm        "$L"       \
        --h_fine_nm   "$H_FINE"  \
        --h_coarse_nm "$H_COARSE" \
        --fine_box_nm "$BOX_NM"  \
        --case_id     "$ID"      \
        --out results/sweep.csv

    echo "Wall time: ${SECONDS}s"
done

echo ""
echo "================================================================"
echo "All cases done."
echo ""
echo "Results table:"
column -t -s, results/sweep.csv
echo "================================================================"
