#!/usr/bin/env bash
# run_blank_cases.sh — Family 1: TRUE BLANK / NO-VOLTAGE CASES
#
# Physics: V_gate=0, V_bottom=0, no gate Dirichlet BC, rho=0.
# This is the trivial electrostatics baseline (phi=0 everywhere for pure Laplace).
# Useful as a null check and reference for delta calculations.
#
# Cases:
#   blank_exact_tet_h6   — 500×500×255 nm box
#   blank_larger_tet_h6  — 800×800×255 nm box
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./run_blank_cases.sh [exact|larger]   # default: both

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
STACK="--layer_materials Si70Ge30,sSi,Si70Ge30
  --layer_heights_nm 50,5,200
  --layer_epsr 13.05,11.7,13.05
  --rho_layers_C_m3 0,0,0"
MESH="--celltype tet --h_nm 6 --mesh_mode uniform"
BCS="--V_gate 0.0 --V_bottom 0.0
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
[[ "$CASE" == "exact" || "$CASE" == "both" ]] && RUN "blank_exact_tet_h6"  500 500
[[ "$CASE" == "larger" || "$CASE" == "both" ]] && RUN "blank_larger_tet_h6" 800 800

echo ""
echo "Done. Blank cases in results/blank_*/"
