#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/fenics/dolfinx/dolfinx:stable"
PYFILE="cif_xml_disk_poisson.py"

for L in 500 800 1200; do
  echo "=============================================="
  echo "Running box-size test with Lx=Ly=${L} nm (hex)"
  echo "=============================================="

  docker run --rm -it \
    -v "$PWD":/work -w /work \
    "$IMAGE" \
    python3 "$PYFILE" \
      --disk_radius_nm 10 \
      --disk_cx_nm "$((L/2))" --disk_cy_nm "$((L/2))" \
      --Lx_nm "$L" --Ly_nm "$L" \
      --layer_materials Si70Ge30,sSi,Si70Ge30 \
      --layer_heights_nm 50,5,200 \
      --layer_epsr 13.05,11.7,13.05 \
      --rho_layers_C_m3 0,0,0 \
      --V_gate -1.0 \
      --V_bottom -12.0 \
      --bottom_bc_type DIRICHLET \
      --celltype hex \
      --h_nm 6 \
      --periodic none \
      --outdir "Results/boxtest_L${L}_hex_h6_neg1_neg12" \
      --basename "boxtest_L${L}_hex_h6_neg1_neg12"
done
