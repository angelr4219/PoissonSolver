#!/usr/bin/env bash
set -euo pipefail

# ---- Defaults (override with env) ----
DEG="${DEG:-3}"
A="${A:-3.5e-8}"
ZFILL="${ZFILL:-2.0e-7}"          # H
ZCONST="${ZCONST:-$A}"            # probe z (use a by default)
HINT_H="${HINT_H:-5e-9}"         # target element size

# Solver wants JSON arrays:
XS_JSON="${XS_JSON:-[-7.0e-8,0.0,7.0e-8]}"
VS_JSON="${VS_JSON:-[0.25,0.10,0.25]}"

# Verifier wants CSV (NO brackets):
XS_CSV="${XS_CSV:--7.0e-8,0.0,7.0e-8}"
VS_CSV="${VS_CSV:-0.25,0.10,0.25}"

PADS=(2.0 3.0 4.0 5.0)

STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="results/runs/${STAMP}/deg${DEG}"
mkdir -p "${ROOT}"
echo "==> Results root: ${ROOT}"

py() {
  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 "$@"
}

pymod() {
  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable /bin/bash -lc "
/dolfinx-env/bin/python3 -m pip -q install --no-cache-dir meshio h5py >/dev/null 2>&1 || true
exec /dolfinx-env/bin/python3 -m $*
"
}


for P in "${PADS[@]}"; do
  TAG="p$(printf '%.0f' "${P}")a"
  echo "---- solve: ${TAG}  deg=${DEG}"
  py rect_gate_benchmark.py \
    --deg "${DEG}" \
    --a   "${A}" \
    --zfill "${ZFILL}" \
    --xs  "${XS_JSON}" \
    --Vs  "${VS_JSON}" \
    --h   "${HINT_H}" \
    --pad "${P}"

  SOLVE_DIR="$(ls -td results/runs/* 2>/dev/null | head -1)"
  if [[ -z "${SOLVE_DIR:-}" || ! -d "${SOLVE_DIR}" ]]; then
    echo "ERROR: Could not find solver dir under results/runs/*" >&2
    exit 2
  fi

  SOLVE_XDMF="${SOLVE_DIR}/${TAG}_deg${DEG}.xdmf"
  SOLVE_LINE="${SOLVE_DIR}/${TAG}_deg${DEG}_line.csv"
  if [[ ! -f "${SOLVE_XDMF}" ]]; then
    echo "ERROR: Missing solver file: ${SOLVE_XDMF}" >&2
    ls -l "${SOLVE_DIR}" >&2 || true
    exit 3
  fi

  CASE_DIR="${ROOT}/${TAG}"
  mkdir -p "${CASE_DIR}"
  cp -f "${SOLVE_XDMF}" "${CASE_DIR}/"
  [[ -f "${SOLVE_LINE}" ]] && cp -f "${SOLVE_LINE}" "${CASE_DIR}/"

  echo "---- verify: ${TAG}"
  pymod verify.compare_rect_vs_analytic \
    --phi-xdmf="${CASE_DIR}/${TAG}_deg${DEG}.xdmf" \
    --a="${A}" \
    --xs=''"${XS_CSV}"'' \
    --Vs=''"${VS_CSV}"'' \
    --z="${ZCONST}" \
    --field-name=phi \
    --out-stem="${CASE_DIR}/${TAG}_deg${DEG}"


  for f in \
    results/${TAG}_err*.xdmf results/${TAG}_err*.h5 \
    results/${TAG}_grad*.xdmf results/${TAG}_grad*.h5 \
    results/${TAG}_rel_point*.xdmf results/${TAG}_rel_point*.h5 \
    results/*${TAG}*.csv
  do
    [[ -e "$f" ]] || continue
    mv -f "$f" "${CASE_DIR}/"
  done

  echo ".... done: ${TAG}"
done

if command -v tree >/dev/null 2>&1; then
  echo; tree -a "${ROOT}"; echo
else
  echo; find "${ROOT}" -print; echo
fi

echo "==> Finished. Outputs in: ${ROOT}"
