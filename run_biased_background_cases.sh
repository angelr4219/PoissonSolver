#!/usr/bin/env bash
# run_biased_background_cases.sh — Family 2: BIASED BACKGROUND CASES
#
# Physics: V_bottom=-12V, no gate Dirichlet BC, no sigma, rho=0.
# This is the "empty electrostatics with bias" baseline — laterally uniform
# Laplace solution driven entirely by the bottom boundary condition.
#
# Note: Track A (trackA_nogate_sigma2e11_tet_h6) adds sigma=-2e11 e/cm² on top
# for MaSQE comparison purposes. These biased_bg cases deliberately omit sigma
# to give a clean Laplace-only background reference for delta calculations.
# If you want the sigma version for MaSQE comparison, use run_with_sigma.sh.
#
# Cases:
#   biased_bg_exact_tet_h6   — 500×500×255 nm
#   biased_bg_larger_tet_h6  — 800×800×255 nm
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./run_biased_background_cases.sh [exact|larger]

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
STACK="--layer_materials Si70Ge30,sSi,Si70Ge30
  --layer_heights_nm 50,5,200
  --layer_epsr 13.05,11.7,13.05
  --rho_layers_C_m3 0,0,0"
MESH="--celltype tet --h_nm 6 --mesh_mode uniform"
BCS="--V_bottom -12.0
  --bottom_bc_type DIRICHLET --side_bc_type natural
  --skip_disk_bc"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS="-it"; fi

RUN() {
  local LABEL="$1" LX="$2" LY="$3"
  if [[ -f "results/${LABEL}/${LABEL}.xdmf" ]]; then
    echo "  [SKIP] Already exists: results/${LABEL}/"
    return
  fi
  mkdir -p "results/${LABEL}"
  echo ""
  echo "── ${LABEL} ──────────────────────────────────────────────"
  docker run --rm ${TTY_FLAGS} \
    -v "$PWD":/app -w /app "$IMAGE" \
    sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \
    cif_xml_disk_poisson.py \
      --Lx_nm "$LX" --Ly_nm "$LY" \
      --disk_radius_nm 10 --disk_cx_nm $((LX/2)) --disk_cy_nm $((LY/2)) \
      ${STACK} ${MESH} ${BCS} \
      --outdir "results/${LABEL}" --basename "${LABEL}"
  echo "  → results/${LABEL}/"
}

CASE="${1:-both}"
[[ "$CASE" == "exact" || "$CASE" == "both" ]] && RUN "biased_bg_exact_tet_h6"  500 500
[[ "$CASE" == "larger" || "$CASE" == "both" ]] && RUN "biased_bg_larger_tet_h6" 800 800

echo ""
echo "Done. Biased background cases in results/biased_bg_*/"
