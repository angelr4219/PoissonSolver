#!/usr/bin/env bash
# run_local_gate_tests.sh
#
# Test commands for the two mesh modes:
#   a. uniform h=6  (current behavior, unchanged)
#   b. local_gate h_bulk=6, h_gate=1
#   c. local_gate h_bulk=6, h_gate=0.5
#
# Run from the project root:
#   cd ~/Desktop/poisson_solver
#   ./run_local_gate_tests.sh [a|b|c]   # or no arg to run all three

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
COMMON="--Lx_nm 500 --Ly_nm 500
  --disk_radius_nm 10 --disk_cx_nm 250 --disk_cy_nm 250
  --layer_materials Si70Ge30,sSi,Si70Ge30
  --layer_heights_nm 50,5,200
  --layer_epsr 13.05,11.7,13.05
  --rho_layers_C_m3 0,0,0
  --V_gate -1.0 --V_bottom -12.0
  --bottom_bc_type DIRICHLET --side_bc_type natural
  --interface_bc_flag --sigma_top_cm2=-2e11"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS="-it"; fi

RUN() {
  local LABEL="$1"; shift
  echo ""
  echo "========================================================"
  echo "  $LABEL"
  echo "========================================================"
  mkdir -p "results/${LABEL}"
  docker run --rm ${TTY_FLAGS} \
    -v "$PWD":/app -w /app "$IMAGE" \
    sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; \
            pip install -q gmsh 2>/dev/null || true; \
            /dolfinx-env/bin/python3 -u "$@"' -- \
    cif_xml_disk_poisson.py \
      ${COMMON} \
      "$@" \
      --outdir "results/${LABEL}" --basename "${LABEL}"
  echo "  Done → results/${LABEL}/"
}

CASE="${1:-all}"

# ── a. uniform h=6 ────────────────────────────────────────────────────────────
if [[ "$CASE" == "a" || "$CASE" == "all" ]]; then
  RUN "mesh_uniform_h6" \
    --celltype tet \
    --mesh_mode uniform \
    --h_nm 6
fi

# ── b. local_gate h_bulk=6 h_gate=1 ──────────────────────────────────────────
if [[ "$CASE" == "b" || "$CASE" == "all" ]]; then
  RUN "mesh_local_hbulk6_hgate1" \
    --mesh_mode local_gate \
    --h_bulk_nm 6 \
    --h_gate_nm 1 \
    --refine_radius_nm 30 \
    --refine_depth_nm 20
fi

# ── c. local_gate h_bulk=6 h_gate=0.5 ────────────────────────────────────────
if [[ "$CASE" == "c" || "$CASE" == "all" ]]; then
  RUN "mesh_local_hbulk6_hgate0p5" \
    --mesh_mode local_gate \
    --h_bulk_nm 6 \
    --h_gate_nm 0.5 \
    --refine_radius_nm 30 \
    --refine_depth_nm 20
fi

echo ""
echo "All done. Results in results/mesh_*/"
