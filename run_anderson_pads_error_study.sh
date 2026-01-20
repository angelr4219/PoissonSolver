#!/usr/bin/env bash
set -euo pipefail

IMAGE="dolfinx/dolfinx:stable"
SCRIPT="anderson_pads_error_study.py"

OUTDIR="Results/anderson_error_study"
BASENAME="anderson_pads"

docker run --rm -v "$PWD":/app -w /app "${IMAGE}" \
  python3 -u "${SCRIPT}" \
    --outdir "${OUTDIR}" \
    --basename "${BASENAME}" \
    --Lx 3.0e-7 --Ly 3.0e-7 --H 2.0e-7 \
    --a 3.5e-8 \
    --gate_xs "-7.0e-8 0.0 7.0e-8" \
    --gate_ys "0.0 0.0 0.0" \
    --gate_Vs "0.25 0.10 0.25" \
    --hs "1.5e-8 1.0e-8 7.5e-9" \
    --degs "1 2 3" \
    --eps_r 1.0 --rho 0.0 \
    --sample_y 0.0 --sample_z 3.5e-8 --sample_xmax 7.0e-8 --nsample 401 \
    --tau 1.0e-9

echo ""
echo "DONE. Look here:"
echo "  ${OUTDIR}/${BASENAME}_hp_table.csv"
echo "  ${OUTDIR}/${BASENAME}_lines/"
