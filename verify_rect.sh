#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<USAGE
Usage:
  $0 --phi-xdmf results/phi_p2a.xdmf --a 3.5e-8 --zbar 3.5e-8 \
     --xs '[-7.0e-8,0,7.0e-8]' --Vs '[0.25,0.10,0.25]'
USAGE
  exit 1
fi

# choose entrypoint: package vs file path
if [[ -f verify/__init__.py ]]; then
  ENTRY=(-m verify.compare_rect_vs_analytic)
elif [[ -f verify/compare_rect_vs_analytic.py ]]; then
  ENTRY=(verify/compare_rect_vs_analytic.py)
else
  echo "ERROR: Can't find verify/compare_rect_vs_analytic.py"
  exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="results/verify/${STAMP}"
mkdir -p "$OUTDIR"

echo "==> Writing outputs to: $OUTDIR"
PYTHONPATH="." python3 "${ENTRY[@]}" "$@" --outdir "$OUTDIR"

echo "==> Produced files:"
find "$OUTDIR" -maxdepth 1 -type f -print | sort
