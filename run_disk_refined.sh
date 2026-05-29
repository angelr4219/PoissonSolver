#!/usr/bin/env bash
# run_disk_refined.sh
#
# Solve the MaSQE device with local disk refinement using both
# Dirichlet and Periodic (x,y) boundary conditions.
#
# Outputs:
#   results/disk_refined_dirichlet/disk_refined_dirichlet.xdmf / .h5
#   results/disk_refined_periodic/disk_refined_periodic.xdmf   / .h5
#
# After it finishes, run the MaSQE comparison with:
#   ./MASQUE-Comparison/run_compare.sh \
#     --cases disk_refined_dirichlet disk_refined_periodic \
#     --qw-exclude-nm 50 55
#
# For a quick test at lower resolution, prepend:
#   H_FINE_NM=5 ./run_disk_refined.sh
# (not yet wired; just edit H_FINE in the script directly)

set -euo pipefail

IMAGE="ghcr.io/jorgensd/dolfinx_mpc:latest"
PROJECT="/Users/angelramirez/Desktop/poisson_solver"

TTY=""
if [ -t 0 ] && [ -t 1 ]; then TTY="-t"; fi

echo "========================================================"
echo "  disk_refined_periodic_masque.py"
echo "  h_fine = 2 nm  (edit script to change)"
echo "  Image : ${IMAGE}"
echo "  Mount : ${PROJECT}"
echo "========================================================"

docker run --rm -i ${TTY} \
  -v "${PROJECT}":/work \
  -w /work \
  "${IMAGE}" \
  bash -lc 'python3 -u poisson_solver/disk_refined_periodic_masque.py'

echo ""
echo "Done. Run comparison:"
echo "  ./MASQUE-Comparison/run_compare.sh \\"
echo "    --cases disk_refined_dirichlet disk_refined_periodic \\"
echo "    --qw-exclude-nm 50 55"
