#!/usr/bin/env python3
"""Four-quadrant mesh-density demo.

Builds a single cubic domain split into 4 quadrants in the x-y plane
(full height in z), each driven to a different target element size:

    Quadrant 1 (x<0, y<0) : h = 20 nm
    Quadrant 2 (x>0, y<0) : h = 10 nm
    Quadrant 3 (x<0, y>0) : h =  5 nm
    Quadrant 4 (x>0, y>0) : h =  1 nm

This demonstrates that the gmsh Box-field refinement machinery in
src/poisson/refinement.py can drive distinct, independently-labelled mesh
densities within one mesh -- the building block needed for comparing local
refinement strategies against MASQUE-style multi-resolution meshes.

Output (results/quadrant_density/):
    quadrant_mesh.xdmf + .h5   region label per cell (1-4) for ParaView
    quadrant_summary.csv       per-quadrant target h, n_cells, measured h

Usage:
    ./run_dolfinx.sh python3 benchmarks/quadrant_mesh_density_demo.py [--L-nm 200]
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poisson.refinement import RefinementBox, build_refined_mesh_3d, mark_refinement_regions

COMM = MPI.COMM_WORLD
IS_ROOT = COMM.rank == 0

# Target element sizes per quadrant [nm], in build order (used as region 1..4)
QUADRANT_H_NM = [20.0, 10.0, 5.0, 1.0]


def _log(msg: str) -> None:
    if IS_ROOT:
        print(msg, flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L-nm", type=float, default=200.0,
                   help="Cube side length [nm] (default 200)")
    p.add_argument("--outdir", default="results/quadrant_density",
                   help="Output directory (default results/quadrant_density)")
    return p


def main():
    args = build_parser().parse_args()
    L_nm = args.L_nm
    s = 1e-9
    L = L_nm * s

    if IS_ROOT:
        os.makedirs(args.outdir, exist_ok=True)
        print("=" * 60)
        print("4-quadrant mesh-density demo")
        print(f"  L = {L_nm} nm   quadrant h = {QUADRANT_H_NM} nm")
        print("=" * 60)

    quarter = L / 2.0
    h_coarse_m = max(QUADRANT_H_NM) * s  # background, irrelevant since boxes tile the domain

    boxes = [
        RefinementBox(cx=-quarter / 2, cy=-quarter / 2, cz=L / 2,
                       lx=quarter, ly=quarter, lz=L, h_fine=QUADRANT_H_NM[0] * s),
        RefinementBox(cx=+quarter / 2, cy=-quarter / 2, cz=L / 2,
                       lx=quarter, ly=quarter, lz=L, h_fine=QUADRANT_H_NM[1] * s),
        RefinementBox(cx=-quarter / 2, cy=+quarter / 2, cz=L / 2,
                       lx=quarter, ly=quarter, lz=L, h_fine=QUADRANT_H_NM[2] * s),
        RefinementBox(cx=+quarter / 2, cy=+quarter / 2, cz=L / 2,
                       lx=quarter, ly=quarter, lz=L, h_fine=QUADRANT_H_NM[3] * s),
    ]

    domain = build_refined_mesh_3d(COMM, L, L, L, h_coarse_m, boxes)
    region = mark_refinement_regions(domain, boxes)

    # Per-cell volume via DG0 test-function trick: integrating the DG0 basis
    # function (which equals 1 on its own cell, 0 elsewhere) over the mesh
    # yields that cell's volume.
    V0 = fem.functionspace(domain, ("DG", 0))
    v = ufl.TestFunction(V0)
    vol_form = fem.form(v * ufl.dx)
    cell_volumes = fem_petsc.assemble_vector(vol_form).array.real

    region_vals = region.x.array.real
    num_cells = len(region_vals)

    rows = []
    for i, h_target_nm in enumerate(QUADRANT_H_NM, start=1):
        mask = region_vals[:num_cells] == float(i)
        n_cells = int(mask.sum())
        if n_cells > 0:
            # Equivalent edge length of a regular tetrahedron with this volume.
            h_eq_m = (6.0 * np.sqrt(2.0) * cell_volumes[:num_cells][mask]) ** (1.0 / 3.0)
            h_meas_nm = float(np.mean(h_eq_m) / s)
        else:
            h_meas_nm = float("nan")
        rows.append(dict(quadrant=i, h_target_nm=h_target_nm,
                          n_cells=n_cells, h_measured_nm=round(h_meas_nm, 3)))
        _log(f"  Quadrant {i}: target h={h_target_nm:>5.1f} nm   "
             f"n_cells={n_cells:>8,}   measured h≈{h_meas_nm:.2f} nm")

    if IS_ROOT:
        xdmf_path = os.path.join(args.outdir, "quadrant_mesh.xdmf")
        with XDMFFile(COMM, xdmf_path, "w") as f:
            f.write_mesh(domain)
            f.write_function(region)

        csv_path = os.path.join(args.outdir, "quadrant_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["quadrant", "h_target_nm",
                                                     "n_cells", "h_measured_nm"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nTotal cells: {num_cells:,}")
        print(f"Mesh written to {xdmf_path}")
        print(f"Summary written to {csv_path}")


if __name__ == "__main__":
    main()
