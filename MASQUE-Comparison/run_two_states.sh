#!/usr/bin/env bash
# MASQUE-Comparison/run_two_states.sh
#
# Full two-state validation:
#
#   State 0 — background (no gate):
#     DOLFINx: sigma=-2e11 cm⁻², V_bottom=-12 V, no disk gate
#     MaSQE:   ~/Downloads/basePotential3d.vtk   ← already validated (Track A)
#
#   State 1 — gate active:
#     DOLFINx: sigma=-2e11 cm⁻², V_bottom=-12 V, V_gate=-1 V (R=10nm disk)
#     MaSQE:   ~/Downloads/totalPotential3d.vtk  ← need from Leah
#
# Comparison:
#   Both states on the same 401×401 bilinear grid per z-slice.
#   Gate perturbation Δφ = total − background compared between codes.
#   QW band-offset region z=50-55 nm flagged but not counted in verdict.
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./MASQUE-Comparison/run_two_states.sh
#   ./MASQUE-Comparison/run_two_states.sh --skip-solve   # skip solves, compare only

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"

BASE_BG="trackA_nogate_sigma2e11_tet_h6"
BASE_GATE="trackB_gate_sigma2e11_tet_h6"

XDMF_BG="results/${BASE_BG}/${BASE_BG}.xdmf"
XDMF_GATE="results/${BASE_GATE}/${BASE_GATE}.xdmf"

VTK_BG="/downloads/basePotential3d.vtk"
VTK_GATE="/downloads/totalPotential3d.vtk"

OUTDIR="MASQUE-Comparison/outputs_two_states"
GRID_N=401

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS="-it"; fi

SKIP_SOLVE=0
for arg in "$@"; do [[ "$arg" == "--skip-solve" ]] && SKIP_SOLVE=1; done

# ── Pre-flight ────────────────────────────────────────────────────────────────

if [[ ! -f "$HOME/Downloads/basePotential3d.vtk" ]]; then
  echo "ERROR: ~/Downloads/basePotential3d.vtk not found."
  exit 1
fi
if [[ ! -f "$HOME/Downloads/totalPotential3d.vtk" ]]; then
  echo "ERROR: ~/Downloads/totalPotential3d.vtk not found."
  echo "  This is the gate-active total potential — request from Leah."
  exit 1
fi

echo "========================================================"
echo "  Two-state comparison: background vs gate-active"
echo "  VTK bg   : ~/Downloads/basePotential3d.vtk"
echo "  VTK gate : ~/Downloads/totalPotential3d.vtk"
echo "========================================================"

# ── Solve 0: background (sigma only, no gate) ─────────────────────────────────

if [[ "${SKIP_SOLVE}" -eq 1 ]]; then
  echo "[--skip-solve] Skipping solves."
else

  if [[ -f "${XDMF_BG}" ]]; then
    echo "  [SKIP] Background solve exists: ${XDMF_BG}"
  else
    mkdir -p "results/${BASE_BG}"
    echo ""
    echo "── Solve 0: background (sigma only, no gate) ──"
    docker run --rm ${TTY_FLAGS} \
      -v "$PWD":/app -w /app "$IMAGE" \
      sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \
      cif_xml_disk_poisson.py \
        --Lx_nm 500 --Ly_nm 500 \
        --disk_radius_nm 10 --disk_cx_nm 250 --disk_cy_nm 250 \
        --layer_materials "Si70Ge30,sSi,Si70Ge30" \
        --layer_heights_nm "50,5,200" \
        --layer_epsr "13.05,11.7,13.05" \
        --rho_layers_C_m3 "0,0,0" \
        --V_gate -1.0 --V_bottom -12.0 \
        --bottom_bc_type DIRICHLET --side_bc_type natural \
        --celltype tet --h_nm 6 \
        --interface_bc_flag --sigma_top_cm2=-2e11 \
        --skip_disk_bc \
        --outdir "results/${BASE_BG}" --basename "${BASE_BG}"
    echo "  Done: ${XDMF_BG}"
  fi

  # ── Solve 1: gate active (sigma + disk gate) ────────────────────────────────

  if [[ -f "${XDMF_GATE}" ]]; then
    echo "  [SKIP] Gate-active solve exists: ${XDMF_GATE}"
  else
    mkdir -p "results/${BASE_GATE}"
    echo ""
    echo "── Solve 1: gate active (sigma + V_gate=-1V) ──"
    docker run --rm ${TTY_FLAGS} \
      -v "$PWD":/app -w /app "$IMAGE" \
      sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \
      cif_xml_disk_poisson.py \
        --Lx_nm 500 --Ly_nm 500 \
        --disk_radius_nm 10 --disk_cx_nm 250 --disk_cy_nm 250 \
        --layer_materials "Si70Ge30,sSi,Si70Ge30" \
        --layer_heights_nm "50,5,200" \
        --layer_epsr "13.05,11.7,13.05" \
        --rho_layers_C_m3 "0,0,0" \
        --V_gate -1.0 --V_bottom -12.0 \
        --bottom_bc_type DIRICHLET --side_bc_type natural \
        --celltype tet --h_nm 6 \
        --interface_bc_flag --sigma_top_cm2=-2e11 \
        --outdir "results/${BASE_GATE}" --basename "${BASE_GATE}"
    echo "  Done: ${XDMF_GATE}"
  fi

fi  # SKIP_SOLVE

# ── Compare ───────────────────────────────────────────────────────────────────

echo ""
echo "── Compare: two-state side-by-side ──"
echo "  grid-n : ${GRID_N}"
echo "  outdir : ${OUTDIR}/"

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app \
  -v "$HOME/Downloads":/downloads \
  -w /app "$IMAGE" \
  sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; \
          pip install -q h5py scipy 2>/dev/null || true; \
          /dolfinx-env/bin/python3 -u "$@"' -- \
  MASQUE-Comparison/compare_two_states.py \
    --vtk-bg   "${VTK_BG}" \
    --vtk-gate "${VTK_GATE}" \
    --xdmf-bg  "${XDMF_BG}" \
    --xdmf-gate "${XDMF_GATE}" \
    --outdir "${OUTDIR}" \
    --grid-n "${GRID_N}" \
    --slices-nm 0 10 20 50 55 100 155 \
    --qw-exclude-nm 50 55 \
    --disk-cx-nm 250 --disk-cy-nm 250

echo ""
echo "Done. Outputs: ${OUTDIR}/"
echo "Open: open ${OUTDIR}/"
