#!/usr/bin/env python3
"""
solve_rect_gates.py
Solve Laplace/Poisson on a rectangle with rectangular hole "gates" (Dirichlet on gate boundaries),
Neumann=0 on outer boundary; optional oxide/semiconductor split with different ε.

Usage:
  python -m src.cli.solve_rect_gates --Lx 1 --Ly 1 --h 0.03 \
    --holes 2 --gate-width 0.15 --gate-height 0.08 --gate-gap 0.05 --gate-y 0.8 \
    --oxide-ymin 0.7 --Vgate 0.2 --eps_semiconductor 11.7 --eps_oxide 3.9 \
    --outfile results/phi_rect_gates

Outputs:
  XDMF with mesh, phi, eps; prints diagnostics and facet counts.

"""
from __future__ import annotations
import argparse, ast
import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem
from src.geometry.make_holes_rectangular import build as build_geo
from src.geometry.helpers import gmsh_model_to_mesh
from src.physics.permittivity import eps_from_materials, EPS0
from src.solver.poisson import solve_poisson

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.03)
    ap.add_argument("--holes", type=int, default=2)
    ap.add_argument("--gate-width", type=float, default=0.15)
    ap.add_argument("--gate-height", type=float, default=0.08)
    ap.add_argument("--gate-gap", type=float, default=0.05)
    ap.add_argument("--gate-y", type=float, default=0.8)
    ap.add_argument("--oxide-ymin", type=float, default=None)
    ap.add_argument("--Vgate", type=float, default=0.2, help="Dirichlet voltage on all gate facets (gate_all)")
    ap.add_argument("--outfile", type=str, default="results/phi_rect_gates")
    ap.add_argument("--eps_semiconductor", type=float, default=11.7)
    ap.add_argument("--eps_oxide", type=float, default=3.9)
    args = ap.parse_args()

    model = build_geo(args.Lx, args.Ly, args.h, args.holes, args.gate_width, args.gate_height, args.gate_gap, args.gate_y, args.oxide_ymin)
    domain, cell_tags, facet_tags, t2n_cell, t2n_facet = gmsh_model_to_mesh(MPI.COMM_WORLD)
    import gmsh; gmsh.finalize()

    # Build ε field: tag names → εr
    overrides = {"semiconductor": args.eps_semiconductor, "oxide": args.eps_oxide, "domain": args.eps_semiconductor}
    eps_fun = eps_from_materials(domain, cell_tags, t2n_cell, overrides, space="DG", degree=0)

    # Dirichlet on gates
    dirich = {"gate_all": args.Vgate}

    uh, V, (dx, ds), diag = solve_poisson(
        domain, cell_tags, facet_tags, t2n_facet, eps_fun, f=0.0,
        dirichlet_values=dirich,
        petsc_opts={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-12}
    )

    if domain.comm.rank == 0:
        print("[rect_gates] facet counts:", diag.facet_counts)
        print(f"[rect_gates] phi range: min={diag.u_min:.4e} V, max={diag.u_max:.4e} V")
        print(f"[rect_gates] energy: {diag.energy:.6e} J")

    # Save fields
    with XDMFFile(domain.comm, f"{args.outfile}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_tags)
        xdmf.write_meshtags(facet_tags)
        xdmf.write_function(uh)
        xdmf.write_function(eps_fun)

if __name__ == "__main__":
    main()
