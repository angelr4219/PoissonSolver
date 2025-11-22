# Save me as: scripts/run_rect_series.sh
#!/usr/bin/env bash
# Robust, timestamped run harness for rect_gate_benchmark.py
# - Creates results/runs/<STAMP>/
# - Sweeps degrees and (optionally) nx hints, moving outputs into subfolders
# - Bash + nullglob so missing matches don't error

set -euo pipefail
shopt -s nullglob dotglob

# ------- Defaults -------
DEG_LIST=(1 2 3 4 5)
NX_LIST=(16 32 64 128)
A="3.5e-8"
ZFILL="3.5e-8"
XS="[-7.0e-8,0,7.0e-8]"
VS="[0.25,0.10,0.25]"
PADS=("2.0" "3.0" "4.0" "5.0")  # only used if your script needs per-pad runs

# ------- Args parsing -------
print_help () {
  cat <<EOF
Usage: $0 [--deg-list "1 2 3 4 5"] [--nx-list "16 32 64 128"]
       [--a 3.5e-8] [--zfill 3.5e-8] [--xs '[-7e-8,0,7e-8]'] [--vs '[0.25,0.10,0.25]']

Runs rect_gate_benchmark.py in a no-overwrite, timestamped folder:
  results/runs/<YYYYMMDD-HHMMSS>/

Auto-detects whether --nx_hint is supported by rect_gate_benchmark.py.
If not, nx sweep is skipped and only degree sweep is used.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deg-list) shift; IFS=' ' read -r -a DEG_LIST <<< "${1:-}"; shift || true;;
    --nx-list)  shift; IFS=' ' read -r -a NX_LIST  <<< "${1:-}"; shift || true;;
    --a)        shift; A="${1}"; shift || true;;
    --zfill)    shift; ZFILL="${1}"; shift || true;;
    --xs)       shift; XS="${1}"; shift || true;;
    --vs|--Vs)  shift; VS="${1}"; shift || true;;
    -h|--help)  print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 2;;
  esac
done

# ------- Stamp + base dir -------
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="results/runs/${STAMP}"
mkdir -p "${RUN_ROOT}"
echo "==> Run stamp: ${STAMP}"
echo "==> Output root: ${RUN_ROOT}"

# ------- Detect --nx_hint support -------
HAS_NX_HINT=0
if grep -q -- "--nx_hint" rect_gate_benchmark.py; then
  HAS_NX_HINT=1
fi
if [[ "${HAS_NX_HINT}" -eq 1 ]]; then
  echo "==> rect_gate_benchmark.py supports --nx_hint (nx sweep enabled)."
else
  echo "==> rect_gate_benchmark.py does NOT advertise --nx_hint (nx sweep skipped)."
fi

# ------- Helper: run once and relocate outputs -------
run_and_collect () {
  local deg="$1"
  local nx="$2"     # can be "NA" if nx not used
  local subdir

  if [[ "${HAS_NX_HINT}" -eq 1 ]]; then
    subdir="${RUN_ROOT}/deg${deg}_nx${nx}"
  else
    subdir="${RUN_ROOT}/deg${deg}"
  fi
  mkdir -p "${subdir}"

  echo "----"
  echo "Running: deg=${deg} ${HAS_NX_HINT:+nx_hint=${nx}}"
  echo "Temp outputs will first land in results/, then be moved to: ${subdir}"

  # If rect_gate_benchmark.py runs all pads at once, call once; otherwise per-pad.
  if [[ "${HAS_NX_HINT}" -eq 1 ]]; then
    docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
      /dolfinx-env/bin/python3 rect_gate_benchmark.py \
        --deg "${deg}" --nx_hint "${nx}" \
        --a "${A}" --zfill "${ZFILL}" \
        --xs "${XS}" --Vs "${VS}"
  else
    docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
      /dolfinx-env/bin/python3 rect_gate_benchmark.py \
        --deg "${deg}" \
        --a "${A}" --zfill "${ZFILL}" \
        --xs "${XS}" --Vs "${VS}"
  fi

  # Move whatever phi_p?a* artifacts were produced (xdmf/h5/csv/etc.)
  local moved_any=0
  for f in results/phi_p[2345]a*; do
    mv -f "$f" "${subdir}/"
    moved_any=1
  done

  if [[ "${moved_any}" -eq 0 ]]; then
    # If rect_gate_benchmark.py actually requires per-pad runs, try them:
    echo "No phi_p?a* files found. Trying per-pad invocations …"
    for pad in "${PADS[@]}"; do
      echo "  -> running --pad ${pad}"
      if [[ "${HAS_NX_HINT}" -eq 1 ]]; then
        docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
          /dolfinx-env/bin/python3 rect_gate_benchmark.py \
            --pad "${pad}" --deg "${deg}" --nx_hint "${nx}" \
            --a "${A}" --zfill "${ZFILL}" \
            --xs "${XS}" --Vs "${VS}"
      else
        docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
          /dolfinx-env/bin/python3 rect_gate_benchmark.py \
            --pad "${pad}" --deg "${deg}" \
            --a "${A}" --zfill "${ZFILL}" \
            --xs "${XS}" --Vs "${VS}"
      fi
    done

    for f in results/phi_p[2345]a*; do
      mv -f "$f" "${subdir}/" || true
      moved_any=1
    done
  fi

  if [[ "${moved_any}" -eq 0 ]]; then
    echo "WARNING: No outputs matching results/phi_p[2345]a* were produced."
  else
    echo "Collected outputs into: ${subdir}"
    ls -1 "${subdir}" || true
  fi
}

# ------- Sweep -------
if [[ "${HAS_NX_HINT}" -eq 1 ]]; then
  for deg in "${DEG_LIST[@]}"; do
    for nx in "${NX_LIST[@]}"; do
      run_and_collect "${deg}" "${nx}"
    done
  done
else
  for deg in "${DEG_LIST[@]}"; do
    run_and_collect "${deg}" "NA"
  done
fi

echo
echo "==> Finished. Run tree:"
command -v tree >/dev/null 2>&1 && tree -a "${RUN_ROOT}" || find "${RUN_ROOT}" -print
echo
echo "Outputs in: ${RUN_ROOT}"
