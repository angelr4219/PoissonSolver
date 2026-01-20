#!/usr/bin/env bash
set -euo pipefail

SCRIPT="anderson_style_3gates_extrude_or_cavity.py"
IMAGE="dolfinx/dolfinx:stable"

OUTROOT="Results/anderson_like_all"
BASENAME="three_gates"

# Geometry
Lx="3.0e-7"
Ly="3.0e-7"
H="2.0e-7"
a="3.5e-8"
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

mkdir -p "${OUTROOT}"

# 1) Pads (analytic reference included)
run_case "pads" "01_pads"

# 2) Extruded (compare to pads on same mesh, writes delta_phi + delta_line_sample.csv)
run_case "extruded" "02_extruded" \
  --also_solve_pads \
  --gate_thickness "${gate_thickness}"

# 3) Cavity (compare to pads on same mesh, writes delta_phi + delta_line_sample.csv)
run_case "cavity" "03_cavity" \
  --also_solve_pads \
  --cavity_depth "${cavity_depth}"

echo ""
echo "============================================"
echo "Computing line-based errors (3 metrics each)"
echo "============================================"
./analyze_anderson_line_errors.py \
  --root "${OUTROOT}" \
  --basename "${BASENAME}" \
  --tau 1e-9 \
  --out_csv "${OUTROOT}/error_summary.csv"

echo ""
echo "DONE."
echo "Outputs: ${OUTROOT}"
echo "  Error summary: ${OUTROOT}/error_summary.csv"
