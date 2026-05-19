#!/usr/bin/env bash
# run_pos1_cases.sh — Family 3: POSITIVE GATE-ACTIVE CASES
#
# Physics: V_gate=+1.0 V, V_bottom=+12.0 V, gate active, rho=0.
# Positive polarity — useful for symmetry checks and sign-convention tests.
#
# Delta pairing:
#   delta_pos = phi_pos1_p12 - phi_blank_exact
#   (we use the TRUE BLANK as background because there is no +12V-biased no-gate
#   case in the standard suite; see VALIDATION_LADDER_NEXT_STEPS.md)
#
# Cases:
#   pos1_p12_exact_tet_h6   — 500×500×255 nm
#   pos1_p12_larger_tet_h6  — 800×800×255 nm
#
# Usage:
#   cd ~/Desktop/poisson_solver
#   ./run_pos1_cases.sh [exact|larger]

set -euo pipefail

IMAGE="ghcr.io/fenics/dolfinx/dolfinx:stable"
STACK="--layer_materials Si70Ge30,sSi,Si70Ge30
  --layer_heights_nm 50,5,200
  --layer_epsr 13.05,11.7,13.05
  --rho_layers_C_m3 0,0,0"
MESH="--celltype tet --h_nm 6 --mesh_mode uniform"
BCS="--V_gate 1.0 --V_bottom 12.0
  --bottom_bc_type DIRICHLET --side_bc_type natural"

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
[[ "$CASE" == "exact" || "$CASE" == "both" ]] && RUN "pos1_p12_exact_tet_h6"  500 500
[[ "$CASE" == "larger" || "$CASE" == "both" ]] && RUN "pos1_p12_larger_tet_h6" 800 800

echo ""
echo "Done. Positive gate cases in results/pos1_*/"
