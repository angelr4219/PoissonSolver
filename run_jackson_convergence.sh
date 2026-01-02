#!/usr/bin/env bash
set -euo pipefail

IMG="${IMG:-dolfinx/dolfinx:stable}"
SCRIPT="jackson_point_charge_interface.py"

OUTDIR="results/jackson_pointcharge/convergence"
CSV="${OUTDIR}/jackson_relerr_max.csv"
mkdir -p "${OUTDIR}"

# Problem params
eps1r=11.7
eps2r=3.9
q=1.602176634e-19
z0=1e-8
sigma=5e-9
Lx=1e-7
Ly=1e-7
Lz=1e-7

# Sweeps
hs=(1.6e-8 1.2e-8 8e-9 6e-9 4e-9)   # h-refinement
deg_fixed=1

h_fixed=8e-9                          # p-refinement
degs=(1 2 3)                           # (add 4 if you want)

run_one () {
  local study="$1"
  local h="$2"
  local deg="$3"
  local write_xdmf="${4:-0}"

  local base="jackson_${study}_h${h}_p${deg}"
  base="${base//./p}"

  local xdmf_flag="--no-xdmf"
  if [[ "${write_xdmf}" == "1" ]]; then
    xdmf_flag=""
  fi

  docker run --rm -v "$PWD":/app -w /app "${IMG}" \
    /dolfinx-env/bin/python3 -u "${SCRIPT}" \
      --h "${h}" --deg "${deg}" \
      --eps1r "${eps1r}" --eps2r "${eps2r}" \
      --q "${q}" --z0 "${z0}" --sigma "${sigma}" \
      --Lx "${Lx}" --Ly "${Ly}" --Lz "${Lz}" \
      --outdir "${OUTDIR}" --basename "${base}" \
      ${xdmf_flag} \
      --csv "${CSV}" --tag "${study}" \
    | grep -m1 '^RESULT ' || true

  if [[ "${write_xdmf}" == "1" ]]; then
    echo "[XDMF] ${OUTDIR}/${base}.xdmf"
  fi
}

# ---- h-study: write XDMF only for finest h ----
for i in "${!hs[@]}"; do
  h="${hs[$i]}"
  write=0
  [[ "$i" -eq "$((${#hs[@]}-1))" ]] && write=1
  run_one "h-study" "${h}" "${deg_fixed}" "${write}"
done

# ---- p-study: write XDMF only for highest p ----
for i in "${!degs[@]}"; do
  deg="${degs[$i]}"
  write=0
  [[ "$i" -eq "$((${#degs[@]}-1))" ]] && write=1
  run_one "p-study" "${h_fixed}" "${deg}" "${write}"
done

echo
echo "CSV: ${CSV}"
echo "XDMF:"
ls -lh "${OUTDIR}"/*.xdmf 2>/dev/null || echo "(none)"
