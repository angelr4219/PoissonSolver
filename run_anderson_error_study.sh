#!/usr/bin/env bash
set -euo pipefail

IMAGE="dolfinx/dolfinx:stable"
SCRIPT="anderson_pads_error_study.py"

OUTDIR="Results/anderson_error_study"
BASENAME="three_gates"

# Use a small sweep first. You can tighten h further if runtime is ok.
HS="1.0e-8 7.5e-9 5.0e-9"
DEGS="1 2 3"

docker run --rm -v "$PWD":/app -w /app "${IMAGE}" \
  python3 -u "${SCRIPT}" \
    --outdir "${OUTDIR}" \
    --basename "${BASENAME}" \
    --Lx 3.0e-7 --Ly 3.0e-7 --H 2.0e-7 \
    --a 3.5e-8 \
    --gate_xs "-7.0e-8 0.0 7.0e-8" \
    --gate_ys "0.0 0.0 0.0" \
    --gate_Vs "0.25 0.10 0.25" \
    --hs "${HS}" \
    --degs "${DEGS}" \
    --rho 0.0 \
    --nsample 401

echo ""
echo "DONE."
echo "Summary CSV:"
echo "  ${OUTDIR}/${BASENAME}_pads_error_summary.csv"
echo ""
