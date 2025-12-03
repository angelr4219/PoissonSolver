#!/usr/bin/env bash
set -euo pipefail

IMAGE="dolfinx/dolfinx:stable"
BASE_OUT="final"

mkdir -p "${BASE_OUT}"

echo "=== Case 1: point charge at (0,0,0) in vacuum ==="
docker run --rm -v "$PWD":/app -w /app "${IMAGE}" \
  /dolfinx-env/bin/python3 PC/point_charge_box.py \
    --L 1e-7 \
    --h 5e-9 \
    --epsr 1.0 \
    --q 1.602176634e-19 \
    --x0 "0.0,0.0,0.0" \
    --sigma 5e-9 \
    --deg 1 \
    --bc dirichlet \
    --outdir "${BASE_OUT}/pc_center"

echo "=== Case 2: point charge at (0,0,10 nm) in vacuum ==="
docker run --rm -v "$PWD":/app -w /app "${IMAGE}" \
  /dolfinx-env/bin/python3 PC/point_charge_box.py \
    --L 1e-7 \
    --h 5e-9 \
    --epsr 1.0 \
    --q 1.602176634e-19 \
    --x0 "0.0,0.0,1e-8" \
    --sigma 5e-9 \
    --deg 1 \
    --bc dirichlet \
    --outdir "${BASE_OUT}/pc_z10nm"

echo "=== Case 3: Jackson interface (epsr_bot=11.7, epsr_top=3.9) ==="
docker run --rm -v "$PWD":/app -w /app "${IMAGE}" \
  /dolfinx-env/bin/python3 PC/multi_point_charge_box_var_eps_layers.py \
    --L=1e-7 --h=5e-9 \
    --epsr-bot=11.7 --epsr-top=3.9 --split-z=0.0 \
    --q=1.602176634e-19 \
    --x0="0.0,0.0,1e-8" \
    --sigma=5e-9 \
    --deg=1 \
    --outdir="${BASE_OUT}/jackson_epsbot11p7_top3p9"

echo "=== Case 4: Jackson interface (epsr_bot=3.9, epsr_top=11.7) ==="
docker run --rm -v "$PWD":/app -w /app "${IMAGE}" \
  /dolfinx-env/bin/python3 PC/multi_point_charge_box_var_eps_layers.py \
    --L=1e-7 --h=5e-9 \
    --epsr-bot=3.9 --epsr-top=11.7 --split-z=0.0 \
    --q=1.602176634e-19 \
    --x0="0.0,0.0,1e-8" \
    --sigma=5e-9 \
    --deg=1 \
    --outdir="${BASE_OUT}/jackson_epsbot3p9_top11p7"

echo "=== All cases done. Outputs in '${BASE_OUT}/' ==="
