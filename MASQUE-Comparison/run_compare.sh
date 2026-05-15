#!/usr/bin/env bash
# MASQUE-Comparison/run_compare.sh
# Run the clean VTK vs XDMF comparison inside Docker.
# Mounts both the project AND ~/Downloads so the VTK file is accessible.
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./MASQUE-Comparison/run_compare.sh
#   ./MASQUE-Comparison/run_compare.sh --grid-n 401   # finer grid (slower)
#   ./MASQUE-Comparison/run_compare.sh --cases basepot_R10_exact_tet_h6_neg1_neg12

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
VTK_PATH="/downloads/basePotential3d.vtk"   # mounted from ~/Downloads
OUTDIR="MASQUE-Comparison/outputs"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

echo "========================================================"
echo "  MASQUE Comparison — VTK vs DOLFINx FEM"
echo "  VTK    : ~/Downloads/basePotential3d.vtk"
echo "  XDMF   : results/basepot_R10_*/..."
echo "  Output : ${OUTDIR}/"
echo "========================================================"

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app \
  -v "$HOME/Downloads":/downloads \
  -w /app \
  "$IMAGE" \
  sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; \
          pip install -q scipy 2>/dev/null || true; \
          /dolfinx-env/bin/python3 -u "$@"' -- \
  MASQUE-Comparison/compare_masque.py \
    --vtk "${VTK_PATH}" \
    --results-root results \
    --outdir "${OUTDIR}" \
    "$@"

echo ""
echo "Done. See outputs in: ${OUTDIR}/"
echo ""
echo "Open results:"
echo "  open ${OUTDIR}/"
