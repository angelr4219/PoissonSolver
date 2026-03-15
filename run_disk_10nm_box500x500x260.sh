#!/usr/bin/env bash
set -euo pipefail

./run_in_docker.sh python3 main.py charged_disk_box \
  --R 10e-9 \
  --t 4e-9 \
  --Q 1.602176634e-19 \
  --out out_mesh_fields/disk_R10nm_box500x500x260 \
  --p 2 \
  --epsr 12.0 \
  --z0 0.0 \
  --Lx 500e-9 \
  --Ly 500e-9 \
  --Lz 260e-9 \
  --xmin -250e-9 \
  --xmax  250e-9 \
  --ymin -250e-9 \
  --ymax  250e-9 \
  --zmin 0.0 \
  --zmax 260e-9 \
  --n_diam 60 \
  --refine_band_R 12.0 \
  --far_h_factor 12.0 \
  --hard_exit
