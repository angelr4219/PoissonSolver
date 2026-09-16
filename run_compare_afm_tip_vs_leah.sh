#!/bin/bash
set -e
cd ~/Desktop/poisson_solver
docker run --rm -it \
  -v "$HOME":/root/home \
  -w /root/home/Desktop/poisson_solver \
  dolfinx/dolfinx:stable \
  python3 compare_sige_afm_tip_vs_leah.py \
  --vtk "/root/home/Downloads/basePotential3d(1).vtk" \
  --gap 30 --tip-voltage 1.0 \
  --h-apex 1.0 --h-device 2.0 --h-near 5.0 --h-bottom 100 \
  --outdir Results/compare_afm_tip_vs_leah
