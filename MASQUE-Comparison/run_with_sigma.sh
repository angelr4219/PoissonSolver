#!/usr/bin/env bash
# MASQUE-Comparison/run_with_sigma.sh  (Track A — background validation)
#
# Solves DOLFINx with NO gate Dirichlet BC + surface charge sigma=-2e11 e/cm^2.
# This matches exactly what basePotential3d.vtk represents (1D background only).
# Expected result: <0.1% error everywhere if our background physics is correct.
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./MASQUE-Comparison/run_with_sigma.sh

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
BASENAME="trackA_nogate_sigma2e11_tet_h6"
OUTCASE="results/${BASENAME}"
VTK_PATH="/downloads/basePotential3d.vtk"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

echo "========================================================"
echo "  Step: solve with MaSQE interface charge"
echo "  sigma_top = -2e11 e/cm^2 (from dotarray1disk2.xml)"
echo "  outdir: ${OUTCASE}"
echo "========================================================"

if [[ -f "${OUTCASE}/${BASENAME}.xdmf" ]]; then
  echo "  [SKIP] Already solved: ${OUTCASE}/${BASENAME}.xdmf"
  echo "  Delete the directory to re-solve."
else
  mkdir -p "${OUTCASE}"
  docker run --rm ${TTY_FLAGS} \
    -v "$PWD":/app -w /app \
    "$IMAGE" \
    sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \
    cif_xml_disk_poisson.py \
      --Lx_nm 500 --Ly_nm 500 \
      --disk_radius_nm 10 --disk_cx_nm 250 --disk_cy_nm 250 \
      --layer_materials "Si70Ge30,sSi,Si70Ge30" \
      --layer_heights_nm "50,5,200" \
      --layer_epsr "13.05,11.7,13.05" \
      --rho_layers_C_m3 "0,0,0" \
      --V_gate -1.0 --V_bottom -12.0 \
      --bottom_bc_type DIRICHLET \
      --side_bc_type natural \
      --celltype tet --h_nm 6 \
      --interface_bc_flag \
      --sigma_top_cm2=-2e11 \
      --skip_disk_bc \
      --outdir "${OUTCASE}" \
      --basename "${BASENAME}"
  echo "  Solved: ${OUTCASE}/${BASENAME}.xdmf"
fi

echo ""
echo "========================================================"
echo "  Now running comparison with sigma case included..."
echo "========================================================"

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app \
  -v "$HOME/Downloads":/downloads \
  -w /app \
  "$IMAGE" \
  sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; \
          pip install -q h5py scipy 2>/dev/null || true; \
          /dolfinx-env/bin/python3 -u "$@"' -- \
  MASQUE-Comparison/compare_masque.py \
    --vtk "${VTK_PATH}" \
    --results-root results \
    --outdir "MASQUE-Comparison/outputs_trackA" \
    --cases "${BASENAME}" \
    --slices-nm 0 10 20 40 50 51 52 55 57 155 255

echo ""
echo "Done. Track A outputs: MASQUE-Comparison/outputs_trackA/"
echo "Expect <0.1% error if background physics matches basePotential3d.vtk."
