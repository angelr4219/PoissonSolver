#!/usr/bin/env bash
set -euo pipefail

# Mesh sizes in meters (20,15,10,5,3,1,0.5 nm)
H_LIST=("20e-9" "15e-9" "10e-9" "5e-9" "3e-9" "1e-9" "0.5e-9")

# Polynomial degrees for p-refinement
P_LIST=("1" "2" "3")

# Common physics params (charge closer to interface)
EPS1R=11.7
EPS2R=3.9
Q=1.602176634e-19

# Charge 5 nm above interface at z=0
Z0=5e-9

# Narrow Gaussian so it stays mostly above interface
SIGMA=1e-9

# Box size
LX=1e-7
LY=1e-7
LZ=1e-7

BASE_OUTDIR="results/jackson_sweep_z0_5nm"

for h in "${H_LIST[@]}"; do
  for p in "${P_LIST[@]}"; do
    echo "=== Running h=${h}, p=${p} (z0=${Z0}, sigma=${SIGMA}) ==="

    docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable bash -lc "
      /dolfinx-env/bin/python3 jackson_interface_error.py \
        --h ${h} \
        --deg ${p} \
        --eps1r ${EPS1R} --eps2r ${EPS2R} \
        --q ${Q} \
        --z0 ${Z0} \
        --sigma ${SIGMA} \
        --Lx ${LX} --Ly ${LY} --Lz ${LZ} \
        --outdir ${BASE_OUTDIR}/h_${h}_p${p} \
        --basename jackson_h${h}_p${p}_z0_5nm
    "
  done
done
