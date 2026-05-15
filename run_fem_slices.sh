#!/usr/bin/env bash
# run_fem_slices.sh — standalone FEM slice visualizer (no VTK needed)
#
# Reads the existing gate-active DOLFINx solve and produces smooth 2D
# heatmaps of phi at 0, 10, 20, 50, 100, 155 nm depth.
# Shows disk gate circle overlay on each plot.
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./run_fem_slices.sh

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
BASENAME="basepot_R10_exact_tet_h6_neg1_neg12"
XDMF="results/${BASENAME}/${BASENAME}.xdmf"
OUTDIR="outputs/fem_slices"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

if [[ ! -f "${XDMF}" ]]; then
  echo "ERROR: ${XDMF} not found."
  echo "  Run the gate-active solve first (see MASQUE-Comparison/run_trackB.sh for command)."
  exit 1
fi

echo "========================================================"
echo "  FEM slice visualizer"
echo "  XDMF  : ${XDMF}"
echo "  grid  : 301×301 per slice"
echo "  outdir: ${OUTDIR}/"
echo "========================================================"

mkdir -p "${OUTDIR}"

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app -w /app \
  "$IMAGE" \
  sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; \
          pip install -q scipy 2>/dev/null || true; \
          /dolfinx-env/bin/python3 -u "$@"' -- \
  analysis/visualize_fem_slices.py \
    --xdmf "${XDMF}" \
    --slices-nm 0 10 20 50 100 155 \
    --grid-n 301 \
    --outdir "${OUTDIR}"

echo ""
echo "Done. Open: open ${OUTDIR}/"
