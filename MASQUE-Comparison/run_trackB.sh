#!/usr/bin/env bash
# MASQUE-Comparison/run_trackB.sh  (Track B — gate-active validation)
#
# Solves DOLFINx with:
#   - disk gate Dirichlet = -1 V  (gate active)
#   - bottom Dirichlet    = -12 V
#   - surface sigma       = -2e11 e/cm²  (same Neumann as Track A)
#   - same epsilon stack and box geometry
#   - rho = 0
#
# Then compares DOLFINx phi_total vs MaSQE totalPotential3d.vtk on
# a shared bilinear-interpolated grid (401×401 per z-slice).
#
# PREREQUISITE: ~/Downloads/totalPotential3d.vtk  (from Leah)
#   — must contain: base background + active disk gate response
#   — caveat: if MaSQE totalPotential includes heterostructure band-edge
#     offsets, z=50-55 nm will show ~150 mV mismatch (not a Poisson bug).
#     Those slices are plotted but flagged "QW excluded" in the verdict.
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./MASQUE-Comparison/run_trackB.sh              # solve + compare
#   ./MASQUE-Comparison/run_trackB.sh --skip-solve # compare only (reuse existing solve)

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
BASENAME="trackB_gate_sigma2e11_tet_h6"
OUTCASE="results/${BASENAME}"
VTK_PATH="/downloads/totalPotential3d.vtk"
OUTDIR="MASQUE-Comparison/outputs_trackB"
GRID_N=401

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

SKIP_SOLVE=0
for arg in "$@"; do
  [[ "$arg" == "--skip-solve" ]] && SKIP_SOLVE=1
done

# ── Pre-flight: totalPotential3d.vtk ──────────────────────────────────────────

if [[ ! -f "$HOME/Downloads/totalPotential3d.vtk" ]]; then
  echo "ERROR: ~/Downloads/totalPotential3d.vtk not found."
  echo "  Request from Leah: gate-active total potential"
  echo "  (V_gate=-1V, V_bottom=-12V, sigma=-2e11 cm^-2, R=10nm disk, same box)"
  echo ""
  echo "  Caveat: if totalPotential3d.vtk includes band-edge offsets,"
  echo "  z=50-55 nm will show ~150 mV mismatch (flagged QW excluded, not a bug)."
  echo ""
  echo "  To visualize the FEM result standalone (no VTK needed), run:"
  echo "    ./run_fem_slices.sh"
  exit 1
fi

# ── Step 1: solve ─────────────────────────────────────────────────────────────

if [[ "${SKIP_SOLVE}" -eq 1 ]]; then
  echo "[--skip-solve] Reusing existing solve: ${OUTCASE}/${BASENAME}.xdmf"
elif [[ -f "${OUTCASE}/${BASENAME}.xdmf" ]]; then
  echo "  [SKIP] Already solved: ${OUTCASE}/${BASENAME}.xdmf"
  echo "  Delete the directory to re-solve, or pass --skip-solve to force comparison."
else
  mkdir -p "${OUTCASE}"
  echo "========================================================"
  echo "  Step 1: Solve (sigma + gate active)"
  echo "  sigma_top = -2e11 e/cm²"
  echo "  V_gate    = -1 V  (disk R=10nm active)"
  echo "  V_bottom  = -12 V"
  echo "  outdir    : ${OUTCASE}"
  echo "========================================================"
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
      --outdir "${OUTCASE}" \
      --basename "${BASENAME}"
  echo "  Solved: ${OUTCASE}/${BASENAME}.xdmf"
fi

if [[ ! -f "${OUTCASE}/${BASENAME}.xdmf" ]]; then
  echo "ERROR: solve output not found: ${OUTCASE}/${BASENAME}.xdmf"
  exit 1
fi

# ── Step 2: compare ───────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  Step 2: Compare vs totalPotential3d.vtk"
echo "  grid-n : ${GRID_N}"
echo "  QW excluded: z=50-55 nm (band-offset region)"
echo "  outdir : ${OUTDIR}/"
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
    --outdir "${OUTDIR}" \
    --cases "${BASENAME}" \
    --grid-n "${GRID_N}" \
    --slices-nm 0 10 20 40 50 51 52 55 57 100 155 255 \
    --linecutz 0 155 \
    --qw-exclude-nm 50 55

echo ""
echo "Done. Track B outputs: ${OUTDIR}/"
echo "Open: open ${OUTDIR}/${BASENAME}/"
