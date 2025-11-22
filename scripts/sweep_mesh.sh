#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="results/runs/$STAMP"
mkdir -p "$ROOT"

A="3.5e-8"
H="2.0e-7"
XS='[-7.0e-8,0,7.0e-8]'
VS='[0.25,0.10,0.25]'
DEG=3

# Sweep element size h (coarse -> fine)
H_LIST=("1.5e-8" "1.0e-8" "7.5e-9" "5.0e-9" "3.5e-9")

for h in "${H_LIST[@]}"; do
  echo "=== h = $h ==="
  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
   /dolfinx-env/bin/python3 rect_gate_benchmark.py \
    --deg $DEG --a $A --zfill $H --h $h \
    --xs "$XS" --Vs "$VS"

  # collect phi_p?a files into a subdir
  SUB="$ROOT/h_${h}"
  mkdir -p "$SUB"
  for f in results/phi_p{2,3,4,5}a*; do
    mv -f "$f" "$SUB"/
  done
done

echo "Done. Run set in: $ROOT"

