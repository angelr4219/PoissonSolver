#!/usr/bin/env bash
set -euo pipefail

# ---------- User params ----------
A="3.5e-8"            # a (m)  = 35 nm
ZBAR="${A}"           # probe height z = a
ZFILL="2.0e-7"        # H (m)  = 200 nm
XS='[-7.0e-8,0,7.0e-8]'
VS='[0.25,0.10,0.25]'
HINT_H="5.0e-9"       # target element size (m) ~ 5 nm

DEGREES=(1 2 3)       # CG1, CG2, CG3
PADS=(2 3 4 5)        # p = 2a..5a

# ---------- Stamp + output root ----------
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNROOT="results/runs/${STAMP}"
mkdir -p "${RUNROOT}"
echo "==> Results root: ${RUNROOT}"

# ---------- Helper: run solver once ----------
run_solver () {
  local pad="$1" deg="$2"
  echo "---- solve: p=${pad}a  deg=${deg}"
  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 rect_gate_benchmark.py \
      --deg "${deg}" \
      --a "${A}" --zfill "${ZFILL}" --h "${HINT_H}" \
      --xs "${XS}" --Vs "${VS}" \
      --pad "${pad}"
}

# ---------- Sweep ----------
for p in "${PADS[@]}"; do
  for deg in "${DEGREES[@]}"; do
    subdir="${RUNROOT}/p${p}a/deg${deg}"
    mkdir -p "${subdir}"

    run_solver "${p}" "${deg}"

    # Collect outputs from results/ into the subdir
    moved_any=0
    for f in results/phi_p${p}a*; do
      mv -f "$f" "${subdir}/"
      moved_any=1
    done
    if [[ "${moved_any}" -eq 0 ]]; then
      echo "WARNING: no outputs found for p${p}a deg${deg}"
    else
      echo "Collected into: ${subdir}"
    fi

    # --- Verify (compute errors vs analytic) ---
    # Assumes your verifier module supports these flags
    if [[ -f "${subdir}/phi_p${p}a.xdmf" ]]; then
      echo "---- verify: p=${p}a deg=${deg}"
      python3 -m verify.compare_rect_vs_analytic \
        --phi-xdmf "${subdir}/phi_p${p}a.xdmf" \
        --a "${A}" --zbar "${ZBAR}" \
        --xs "${XS}" --Vs "${VS}" \
        --outprefix "${subdir}/verify"
    else
      echo "SKIP verify: ${subdir}/phi_p${p}a.xdmf not found"
    fi
  done
done

echo
echo "==> Done. Tree:"
command -v tree >/dev/null 2>&1 && tree -a "${RUNROOT}" || find "${RUNROOT}" -print
echo
echo "Open results with ParaView, e.g.:"
echo "  paraview ${RUNROOT}/p2a/deg2/phi_p2a.xdmf"
