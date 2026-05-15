#!/usr/bin/env bash
# run_h3_cases.sh
# Run Cases C and D: h=3 nm disk-gate benchmark (exact 500x500x255 box).
# These are the mesh-refinement cases to test whether the near-surface
# mismatch vs basePotential3d.vtk is caused by disk-edge under-resolution.
#
# Case C: exact tet, h=3 nm, R=10 nm, 500x500x255 nm box
# Case D: exact hex, h=3 nm, R=10 nm, 500x500x255 nm box
#
# Each case takes longer than h=6 (roughly 8x more cells).
# Expected wall time: 5-20 min per case depending on hardware.
#
# Usage:
#   ./run_h3_cases.sh
#   ./run_h3_cases.sh hex   # hex only
#   ./run_h3_cases.sh tet   # tet only
#
# After completion, run ./run_validation.sh to compare against VTK.

set -euo pipefail

IMAGE="dolfinx/dolfinx:nightly"
SCRIPT="cif_xml_disk_poisson.py"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

RUN_TET=true
RUN_HEX=true
if [[ $# -ge 1 ]]; then
  case "$1" in
    tet) RUN_HEX=false ;;
    hex) RUN_TET=false ;;
    *) echo "Usage: $0 [tet|hex]"; exit 1 ;;
  esac
fi

run_case() {
  local CELLTYPE="$1"
  local H_NM=3
  local BASENAME="basepot_R10_exact_${CELLTYPE}_h${H_NM}_neg1_neg12"
  local OUTDIR="results/${BASENAME}"

  echo ""
  echo "========================================================"
  echo "  Case: ${BASENAME}"
  echo "  celltype = ${CELLTYPE}"
  echo "  h_nm     = ${H_NM}"
  echo "  box      = 500 x 500 x 255 nm"
  echo "  R_disk   = 10 nm"
  echo "  V_gate   = -1.0 V"
  echo "  V_bottom = -12.0 V"
  echo "  outdir   = ${OUTDIR}"
  echo "========================================================"

  if [[ -f "${OUTDIR}/${BASENAME}.xdmf" ]]; then
    echo "  [SKIP] ${OUTDIR}/${BASENAME}.xdmf already exists. Delete to rerun."
    return 0
  fi

  mkdir -p "${OUTDIR}"

  docker run --rm ${TTY_FLAGS} \
    -v "$PWD":/app -w /app \
    "$IMAGE" \
    sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \
    "${SCRIPT}" \
      --Lx_nm 500 \
      --Ly_nm 500 \
      --disk_radius_nm 10 \
      --disk_cx_nm 250 \
      --disk_cy_nm 250 \
      --layer_materials "Si70Ge30,sSi,Si70Ge30" \
      --layer_heights_nm "50,5,200" \
      --layer_epsr "13.05,11.7,13.05" \
      --rho_layers_C_m3 "0,0,0" \
      --V_gate -1.0 \
      --V_bottom -12.0 \
      --bottom_bc_type DIRICHLET \
      --side_bc_type natural \
      --celltype "${CELLTYPE}" \
      --h_nm "${H_NM}" \
      --outdir "${OUTDIR}" \
      --basename "${BASENAME}"

  echo ""
  echo "  Finished: ${OUTDIR}/${BASENAME}.xdmf"
}

if $RUN_TET; then
  run_case tet
fi

if $RUN_HEX; then
  run_case hex
fi

echo ""
echo "All requested h=3 cases completed."
echo "Now run:  ./run_validation.sh"
