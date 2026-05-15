#!/usr/bin/env bash
# run_validation.sh
# Run the full multi-case validation ladder inside Docker.
# Compares all available h=6 and h=3 cases against basePotential3d.vtk.
#
# Usage:
#   ./run_validation.sh                         # run all registered cases
#   ./run_validation.sh --cases basepot_R10_exact_tet_h6_neg1_neg12  # one case only
#
# Output goes to: outputs/disk10nm_comparison/

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
OUTDIR="outputs/disk10nm_comparison"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

echo "========================================================"
echo "  Validation Ladder — VTK vs XDMF comparison"
echo "  VTK reference : basePotential3d.vtk"
echo "  Output dir    : ${OUTDIR}"
echo "========================================================"

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app -w /app \
  "$IMAGE" \
  sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \
  analysis/run_validation_ladder.py \
    --vtk basePotential3d.vtk \
    --results-root results \
    --outdir "${OUTDIR}" \
    --grid-n 201 \
    --solver-scale 1e9 \
    "$@"

echo ""
echo "Done. Results in: ${OUTDIR}/"
echo "  Master summary : ${OUTDIR}/master_comparison_summary.csv"
echo "  Report         : ${OUTDIR}/validation_report.txt"
echo "  Bar chart      : ${OUTDIR}/z155_acceptance_bar.png"
echo "  Depth plot     : ${OUTDIR}/error_vs_depth_all_cases.png"
