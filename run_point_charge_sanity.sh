#!/usr/bin/env bash
# run_point_charge_sanity.sh
# Validation ladder Step 1: point-charge sanity benchmark.
# Verifies that the FEM solver correctly recovers phi ~ 1/r behavior.
#
# Pass:  log-log slope of phi(r) vs r ≈ -1.00 (within ±0.10)
# Fail:  slope is far from -1 → wrong epsilon, rho, or boundary treatment
#
# Usage:
#   ./run_point_charge_sanity.sh
#   ./run_point_charge_sanity.sh --eps-r 11.7 --sigma-nm 3

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
OUTDIR="outputs/point_charge_sanity"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

echo "========================================================"
echo "  Validation Ladder Step 1: Point-charge sanity"
echo "  Output dir: ${OUTDIR}"
echo "========================================================"

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app -w /app \
  "$IMAGE" \
  sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \
  analysis/point_charge_sanity.py \
    --outdir "${OUTDIR}" \
    "$@"

echo ""
echo "Done. Outputs in: ${OUTDIR}/"
echo "  Metrics : ${OUTDIR}/point_charge_sanity_metrics.json"
echo "  Plot    : ${OUTDIR}/point_charge_sanity.png"
