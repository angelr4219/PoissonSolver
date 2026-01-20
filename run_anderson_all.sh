#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User knobs (edit these if you want)
# --------------------------------------------
SCRIPT="anderson_style_3gates_extrude_or_cavity.py"

OUTROOT="Results/anderson_like_all"
BASENAME="three_gates"

# Geometry
Lx="3.0e-7"
Ly="3.0e-7"
H="2.0e-7"
a="3.5e-8"

# IMPORTANT: pass these as ONE quoted string each, because argparse expects a single arg
gate_xs="-7.0e-8 0.0 7.0e-8"
gate_ys="0.0 0.0 0.0"
gate_Vs="0.25 0.10 0.25"

# Numerics
h="1.0e-8"
deg="2"
rho="0.0"

# Realistic gate geometry knobs
gate_thickness="2.0e-8"
cavity_depth="2.0e-8"

# Docker
IMAGE="dolfinx/dolfinx:stable"

# --------------------------------------------
# Helpers
# --------------------------------------------
run_case () {
  local mode="$1"
  local sub="$2"
  shift 2

  local outdir="${OUTROOT}/${sub}"
  mkdir -p "${outdir}"

  echo ""
  echo "============================================"
  echo "Running mode=${mode} -> ${outdir}"
  echo "============================================"

  docker run --rm -v "$PWD":/app -w /app "${IMAGE}" \
    python3 -u "${SCRIPT}" \
      --mode "${mode}" \
      --outdir "${outdir}" \
      --basename "${BASENAME}" \
      --Lx "${Lx}" --Ly "${Ly}" --H "${H}" \
      --a "${a}" \
      --gate_xs "${gate_xs}" \
      --gate_ys "${gate_ys}" \
      --gate_Vs "${gate_Vs}" \
      --h "${h}" --deg "${deg}" \
      --rho "${rho}" \
      "$@"
}

# --------------------------------------------
# Run all three
# --------------------------------------------

# 1) Pads (includes analytic reference + line_sample.csv)
run_case "pads" "01_pads"

# 2) Extruded (also solves pads reference on same mesh and writes delta_phi)
run_case "extruded" "02_extruded" \
  --also_solve_pads \
  --gate_thickness "${gate_thickness}"

# 3) Cavity (also solves pads reference on same mesh and writes delta_phi)
run_case "cavity" "03_cavity" \
  --also_solve_pads \
  --cavity_depth "${cavity_depth}"

echo ""
echo "DONE."
echo "Outputs are in: ${OUTROOT}"
echo ""
echo "Quick show-your-boss files:"
echo "  Pads phi:      ${OUTROOT}/01_pads/${BASENAME}_pads_phi.xdmf"
echo "  Extruded phi:  ${OUTROOT}/02_extruded/${BASENAME}_extruded_phi.xdmf"
echo "  Extruded dphi: ${OUTROOT}/02_extruded/${BASENAME}_extruded_delta_phi.xdmf"
echo "  Cavity phi:    ${OUTROOT}/03_cavity/${BASENAME}_cavity_phi.xdmf"
echo "  Cavity dphi:   ${OUTROOT}/03_cavity/${BASENAME}_cavity_delta_phi.xdmf"
echo ""
