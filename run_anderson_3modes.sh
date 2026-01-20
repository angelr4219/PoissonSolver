#!/usr/bin/env bash
set -euo pipefail

# Runs:
#   1) pads
#   2) extruded (also solves pads on same mesh -> delta_phi)
#   3) cavity   (also solves pads on same mesh -> delta_phi)
#
# Output layout:
#   Results/anderson_like/three_gates/
#     pads/
#     extruded/
#     cavity/

SCRIPT="anderson_style_3gates_extrude_or_cavity.py"

ROOT_OUTDIR="Results/anderson_like/three_gates"
BASENAME="three_gates"

# Geometry / physics params (same across runs)
Lx="3.0e-7"
Ly="3.0e-7"
H="2.0e-7"
a="3.5e-8"

gate_xs="-7.0e-8 0.0 7.0e-8"
gate_ys="0.0 0.0 0.0"
gate_Vs="0.25 0.10 0.25"

# Refinement
h="7.5e-9"
deg="3"

# "Realistic" 3D geometry params
gate_thickness="2.0e-8"
cavity_depth="2.0e-8"

rho="0.0"

run_mode () {
  local mode="$1"
  local outdir="$2"
  shift 2

  echo ""
  echo "============================================================"
  echo "[RUN] mode=${mode}"
  echo "[OUT] ${outdir}"
  echo "============================================================"

  mkdir -p "${outdir}"

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    python3 -u "${SCRIPT}" \
      --mode "${mode}" \
      --outdir "${outdir}" \
      --basename "${BASENAME}_${mode}" \
      --Lx "${Lx}" --Ly "${Ly}" --H "${H}" \
      --a "${a}" \
      --gate_xs "${gate_xs}" \
      --gate_ys "${gate_ys}" \
      --gate_Vs "${gate_Vs}" \
      --h "${h}" --deg "${deg}" \
      --rho "${rho}" \
      "$@"
}

# 1) Pads (analytic reference included)
run_mode "pads"     "${ROOT_OUTDIR}/pads"

# 2) Extruded (also solve pads on same mesh -> delta_phi)
run_mode "extruded" "${ROOT_OUTDIR}/extruded" \
  --also_solve_pads \
  --gate_thickness "${gate_thickness}"

# 3) Cavity (also solve pads on same mesh -> delta_phi)
run_mode "cavity"   "${ROOT_OUTDIR}/cavity" \
  --also_solve_pads \
  --cavity_depth "${cavity_depth}"

echo ""
echo "✅ All runs finished."
echo "Outputs in: ${ROOT_OUTDIR}"
echo ""
echo "Key files to look for:"
echo "  pads:     ${ROOT_OUTDIR}/pads/${BASENAME}_pads_line_sample.csv"
echo "  extruded: ${ROOT_OUTDIR}/extruded/${BASENAME}_extruded_delta_phi.xdmf"
echo "  cavity:   ${ROOT_OUTDIR}/cavity/${BASENAME}_cavity_delta_phi.xdmf"
