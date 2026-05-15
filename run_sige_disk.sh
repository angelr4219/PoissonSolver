#!/usr/bin/env bash
# ============================================================
#  SiGe / Si / SiGe quantum-dot stack  —  full benchmark run
#
#  Stack (z=0 at bottom, z=350nm at top):
#    [  0 – 100 nm]  SiGe barrier   ε_r = 13.0
#    [100 – 150 nm]  Si  well        ε_r = 11.7  ← disk here
#    [150 – 350 nm]  SiGe cap        ε_r = 13.0
#
#  Disk: R=10nm, t=4nm, Q=1e, centred at (0,0,125nm)
#
#  Runs:
#    A) Dirichlet BCs  (φ=0 on all outer faces)
#    B) Periodic  BCs  (φ periodic in x,y; Dirichlet on z-faces)
#
#  Then produces comparison plots via analysis/plot_sige_disk.py.
#
#  Usage:
#    ./run_sige_disk.sh
#    DOLFINX_IMAGE=myimage:tag ./run_sige_disk.sh
# ============================================================
set -euo pipefail

IMAGE="${DOLFINX_IMAGE:-ghcr.io/jorgensd/dolfinx_mpc:v0.9.0}"
WORKDIR="/work"

run_docker() {
  docker run --rm -i \
    --shm-size=2g \
    -v "$PWD":"$WORKDIR" \
    -w "$WORKDIR" \
    "$IMAGE" \
    "$@"
}

echo "============================================================"
echo "  CASE A: Dirichlet BC  (phi=0 on all outer faces)"
echo "============================================================"
run_docker python3 main.py charged_disk_box \
  --disk="0.0,0.0,125e-9,10e-9,4e-9,1.602176634e-19" \
  --out results/sige_dirichlet \
  --Lx 500e-9 --Ly 500e-9 --Lz 350e-9 \
  --xmin=-250e-9 --xmax=250e-9 \
  --ymin=-250e-9 --ymax=250e-9 \
  --zmin=0.0     --zmax=350e-9 \
  --n_diam 24 \
  --refine_band_R 6.0 \
  --far_h_factor 8.0 \
  --p 1 \
  --probe_n 401 \
  --layer="0.0,100e-9,13.0" \
  --layer="100e-9,150e-9,11.7" \
  --layer="150e-9,350e-9,13.0" \
  --periodic none \
  --hard_exit

echo ""
echo "============================================================"
echo "  CASE B: Periodic BC in x,y  (Dirichlet only on z-faces)"
echo "============================================================"
run_docker python3 main.py charged_disk_box \
  --disk="0.0,0.0,125e-9,10e-9,4e-9,1.602176634e-19" \
  --out results/sige_periodic \
  --Lx 500e-9 --Ly 500e-9 --Lz 350e-9 \
  --xmin=-250e-9 --xmax=250e-9 \
  --ymin=-250e-9 --ymax=250e-9 \
  --zmin=0.0     --zmax=350e-9 \
  --n_diam 24 \
  --refine_band_R 6.0 \
  --far_h_factor 8.0 \
  --p 1 \
  --probe_n 401 \
  --layer="0.0,100e-9,13.0" \
  --layer="100e-9,150e-9,11.7" \
  --layer="150e-9,350e-9,13.0" \
  --periodic xy \
  --hard_exit

echo ""
echo "============================================================"
echo "  Analysis: comparison plots"
echo "============================================================"
run_docker python3 analysis/plot_sige_disk.py \
  --dir_dirichlet results/sige_dirichlet \
  --dir_periodic  results/sige_periodic \
  --outdir        results/plots

echo ""
echo "============================================================"
echo "  All done."
echo "============================================================"
echo ""
echo "Output files:"
echo "  results/sige_dirichlet/pre_fields.xdmf   (pre-solve: rho, eps, region_id)"
echo "  results/sige_dirichlet/mesh_fields.xdmf  (solved: phi, rho, eps, region_id)"
echo "  results/sige_dirichlet/z_probe.csv        (z, phi, eps_r, rho along axis)"
echo "  results/sige_dirichlet/stats.json         (charge error, phi range, timing)"
echo "  results/sige_periodic/  (same set for periodic case)"
echo ""
echo "Plots (PNG):"
echo "  results/plots/stack_profile.png    (phi, eps, rho profiles)"
echo "  results/plots/bc_comparison.png    (Dirichlet vs Periodic side-by-side)"
echo "  results/plots/summary.txt          (charge-error table)"
echo ""
echo "View in ParaView:"
echo "  Open results/sige_dirichlet/pre_fields.xdmf   (to verify geometry)"
echo "  Open results/sige_dirichlet/mesh_fields.xdmf  (to view solved phi)"
