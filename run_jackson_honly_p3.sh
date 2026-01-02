#!/usr/bin/env bash
set -e

# ============================
# Fixed problem parameters
# ============================

OUTDIR="results/jackson_conv_p3_honly"
SCRIPT="jackson_point_charge_interface.py"
IMAGE="dolfinx/dolfinx:stable"

# Geometry
Lx="1.0e-7"
Ly="1.0e-7"
Lz="1.0e-7"

# Dielectrics
EPS1R="11.7"
EPS2R="3.9"

# Charge
Q="1.602176634e-19"
Z0="1.0e-8"
SIGMA="5.0e-9"

# ============================
# Refinement study
# ============================

P=3
H_LIST=(
  1.0e-8
  8.0e-9
  6.0e-9
  5.0e-9
  4.0e-9
  3.0e-9
  2.5e-9
)

# ============================
# Setup
# ============================

mkdir -p "${OUTDIR}"

echo "=========================================="
echo "Jackson h-refinement at fixed p = ${P}"
echo "Output directory: ${OUTDIR}"
echo "=========================================="

# ============================
# Main loop
# ============================

for h in "${H_LIST[@]}"; do

  BASENAME="jackson_L${Lx}_h${h}_p${P}"

  echo "------------------------------------------"
  echo "p = ${P}, h = ${h}"
  echo "basename = ${BASENAME}"
  echo "------------------------------------------"

  docker run --rm \
    -v "$PWD":/app \
    -w /app \
    "${IMAGE}" \
    /dolfinx-env/bin/python3 -u "${SCRIPT}" \
      --Lx "${Lx}" --Ly "${Ly}" --Lz "${Lz}" \
      --h "${h}" \
      --deg "${P}" \
      --eps1r "${EPS1R}" --eps2r "${EPS2R}" \
      --q "${Q}" --z0 "${Z0}" --sigma "${SIGMA}" \
      --outdir "${OUTDIR}" \
      --basename "${BASENAME}"

done

echo "=========================================="
echo "All p=3 h-refinement runs completed."
echo "CSV log:"
echo "  ${OUTDIR}/jackson_runs.csv"
echo "=========================================="

