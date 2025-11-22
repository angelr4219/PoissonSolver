#!/usr/bin/env bash
set -euo pipefail

IMAGE="dolfinx/dolfinx:stable"

# --- physics knobs (same as your recent runs) ---
A=70e-9
ZFILL=2.0e-7
XS='[-7.0e-8,0.0,7.0e-8]'
VS='[0.25,0.10,0.25]'
H_ELEM=5e-9
PADS=(2 3 4 5)

# --- material mode (pick ONE block) ---
# 1) uniform epsr
MAT_ARGS="--epsr 11.7"
RUN_TAG="prefine_uniform"

# 2) two-layer (uncomment these two lines if you want two-layer instead)
# MAT_ARGS="--two-layer --t_ox 5e-9 --epsr_top 3.9 --epsr_bulk 11.7"
# RUN_TAG="prefine_twolayer"

STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_ROOT="results/runs/${RUN_TAG}_${STAMP}"
mkdir -p "${RUN_ROOT}"

echo "==> Run root: ${RUN_ROOT}"
for DEG in 1 2 3; do
  echo "---- p-refinement: degree=${DEG} ----"
  for PAD in "${PADS[@]}"; do
    echo "     solving: p${PAD}a_deg${DEG}"
    docker run --rm -v "$PWD":/app -w /app "$IMAGE" bash -lc "\
      /dolfinx-env/bin/python3 rect_gate_benchmark.py \
        --deg ${DEG} --a ${A} --zfill ${ZFILL} \
        --xs '${XS}' --Vs '${VS}' \
        --h ${H_ELEM} --pad ${PAD} \
        ${MAT_ARGS} --relerr 1 \
        --run-root '${RUN_ROOT}'"
  done
done

echo "==> p-refinement runs complete under: ${RUN_ROOT}"
echo "==> Next: summarize with scripts/summarize_p_refinement.py ${RUN_ROOT}"

