#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve_shapes.py — 2D Poisson/Laplace with a single inclusion (rod or disk).

BCs:
  outer boundary → 0.0 V (Dirichlet)
  inclusion boundary → Vinc (Dirichlet)

Run:
  ./run_dolfinx.sh src/cli/solve_shapes.py \
    --mode rod2d --Lx 1.0 --Ly 1.0 --R 0.15 --h 0.03 --Vinc 0.20 \
    --outfile results/phi_rod2d
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import ufl
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile

# --- Import your project-local solver; fail loudly if missing ---
try:
    from src.solver.poisson import solve_poisson
except Exception as _e:
    raise RuntimeError("Could not import src.solver.poisson.solve_poisson") from _e

# Optional: permittivity builder; fall back to eps=1
try:
    from src.physics.permittivity import eps_from_materials
except Exception:
    eps_from_materials = None


# ----------------- Gmsh helpers -----------------
def _gmsh_preamble(h: float):
    gmsh.initialize()
    gmsh.model.add("shapes2d")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)


def _finalize_to_dolfinx(comm: MPI.Comm, gdim: int = 2):
    """
    Synchronize CAD -> generate mesh -> convert to dolfinx.
    Works across dolfinx versions that return 3+ values from model_to_mesh.
    """
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)
    try:
        gmsh.model.mesh.removeDuplicateNodes()
    except Exception:
        pass
    try:
        gmsh.model.mesh.optimize("Netgen")
    except Exception:
        pass

    out = gmshio.model_to_mesh(gmsh.model, comm, rank=comm.rank, gdim=gdim)
    # take first three elements only (mesh, cell_tags, facet_tags)
    domain = out[0]
    cell_tags = out[1]
    facet_tags = out[2]
# --- Enforce Dirichlet BCs (outer=0V on tag=1, hole=Vinc on tag=2) ---
V = fem.functionspace(domain, ("Lagrange", 1))
uD_out  = fem.Constant(domain, PETSc.ScalarType(0.0))
uD_hole = fem.Constant(domain, PETSc.ScalarType(args.Vinc))
tdim = domain.topology.dim
outer_facets = facet_tags.find(1)
hole_facets  = facet_tags.find(2)
dofs_out  = fem.locate_dofs_topological(V, tdim-1, outer_facets)
dofs_hole = fem.locate_dofs_topological(V, tdim-1, hole_facets)
bc_out  = fem.dirichletbc(uD_out,  dofs_out,  V)
bc_hole = fem.dirichletbc(uD_hole, dofs_hole, V)
bcs = [bc_out, bc_hole]
print("[BC] Dirichlet set: outer→0V, hole→Vinc; dofs:", len(dofs_out), len(dofs_hole))
    return domain, cell_tags, facet_tags


def build_disk2d(comm: MPI.Comm, Lx: float, Ly: float, R: float, h: float):
    """
    Rectangle [0,Lx]x[0,Ly] with centered disk of radius R.
    Physical tags:
      cells: 1 -> bulk, 2 -> inclusion
      facets: 1 -> outer, 2 -> inclusion_boundary
    """
    _gmsh_preamble(h)
    occ = gmsh.model.occ

    # Outer rectangle and centered disk
    rect = occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)
    cx, cy = 0.5 * Lx, 0.5 * Ly
    disk = occ.addDisk(cx, cy, 0.0, R, R)

    # Carve disk
    occ.fragment([(2, rect)], [(2, disk)])
    occ.synchronize()

    # Identify surfaces by area
    surfs = gmsh.model.getEntities(2)
    areas = [(s[1], gmsh.model.occ.getMass(2, s[1])) for s in surfs]
    inc = min(areas, key=lambda x: x[1])[0]
    bulk = max(areas, key=lambda x: x[1])[0]

    # Physical cells
    gmsh.model.addPhysicalGroup(2, [bulk], 1)   # bulk
    gmsh.model.addPhysicalGroup(2, [inc], 2)    # inclusion

    # Facets
    inc_curves = [e[1] for e in gmsh.model.getBoundary([(2, inc)], oriented=False) if e[0] == 1]
    bulk_curves = [e[1] for e in gmsh.model.getBoundary([(2, bulk)], oriented=False) if e[0] == 1]
    outer_curves = list(set(bulk_curves) - set(inc_curves))
    gmsh.model.addPhysicalGroup(1, outer_curves, 1)  # outer
    gmsh.model.addPhysicalGroup(1, inc_curves, 2)    # inclusion_boundary

    return _finalize_to_dolfinx(comm)


def build_rod2d(comm: MPI.Comm, Lx: float, Ly: float, R: float, h: float):
    """
    Rectangle [0,Lx]x[0,Ly] with centered square 'rod' of size (2R)x(2R).
    Physical tags:
      cells: 1 -> bulk, 2 -> inclusion
      facets: 1 -> outer, 2 -> inclusion_boundary
    """
    _gmsh_preamble(h)
    occ = gmsh.model.occ

    rect = occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)
    cx, cy = 0.5 * Lx, 0.5 * Ly
    rod = occ.addRectangle(cx - R, cy - R, 0.0, 2 * R, 2 * R)

    occ.fragment([(2, rect)], [(2, rod)])
    occ.synchronize()

    surfs = gmsh.model.getEntities(2)
    areas = [(s[1], gmsh.model.occ.getMass(2, s[1])) for s in surfs]
    inc = min(areas, key=lambda x: x[1])[0]
    bulk = max(areas, key=lambda x: x[1])[0]

    gmsh.model.addPhysicalGroup(2, [bulk], 1)   # bulk
    gmsh.model.addPhysicalGroup(2, [inc], 2)    # inclusion

    inc_curves = [e[1] for e in gmsh.model.getBoundary([(2, inc)], oriented=False) if e[0] == 1]
    bulk_curves = [e[1] for e in gmsh.model.getBoundary([(2, bulk)], oriented=False) if e[0] == 1]
    outer_curves = list(set(bulk_curves) - set(inc_curves))
    gmsh.model.addPhysicalGroup(1, outer_curves, 1)  # outer
    gmsh.model.addPhysicalGroup(1, inc_curves, 2)    # inclusion_boundary

    return _finalize_to_dolfinx(comm)


# --------------- Utilities ----------------
def get_function_space(domain):
    """Version-agnostic helper: FunctionSpace vs functionspace."""
    try:
        return fem.FunctionSpace(domain, ("Lagrange", 1))
    except Exception:
        return fem.functionspace(domain, ("Lagrange", 1))


# --------------- Main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Poisson/Laplace on simple 2D shapes with one inclusion.")
    p.add_argument("--mode", choices=["rod2d", "disk2d"], required=True)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R", type=float, default=0.15, help="Radius (disk) or half-width (rod square).")
    p.add_argument("--h", type=float, default=0.05, help="Target mesh size.")
    p.add_argument("--Vinc", type=float, default=0.2, help="Potential on inclusion boundary (V).")
    p.add_argument("--outfile", type=str, default="results/phi")
    return p.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD

    # --- Build geometry ---
    if args.mode == "disk2d":
        domain, cell_tags, facet_tags = build_disk2d(comm, args.Lx, args.Ly, args.R, args.h)
# --- Enforce Dirichlet BCs (outer=0V on tag=1, hole=Vinc on tag=2) ---
V = fem.functionspace(domain, ("Lagrange", 1))
uD_out  = fem.Constant(domain, PETSc.ScalarType(0.0))
uD_hole = fem.Constant(domain, PETSc.ScalarType(args.Vinc))
tdim = domain.topology.dim
outer_facets = facet_tags.find(1)
hole_facets  = facet_tags.find(2)
dofs_out  = fem.locate_dofs_topological(V, tdim-1, outer_facets)
dofs_hole = fem.locate_dofs_topological(V, tdim-1, hole_facets)
bc_out  = fem.dirichletbc(uD_out,  dofs_out,  V)
bc_hole = fem.dirichletbc(uD_hole, dofs_hole, V)
bcs = [bc_out, bc_hole]
print("[BC] Dirichlet set: outer→0V, hole→Vinc; dofs:", len(dofs_out), len(dofs_hole))
    else:
        domain, cell_tags, facet_tags = build_rod2d(comm, args.Lx, args.Ly, args.R, args.h)
# --- Enforce Dirichlet BCs (outer=0V on tag=1, hole=Vinc on tag=2) ---
V = fem.functionspace(domain, ("Lagrange", 1))
uD_out  = fem.Constant(domain, PETSc.ScalarType(0.0))
uD_hole = fem.Constant(domain, PETSc.ScalarType(args.Vinc))
tdim = domain.topology.dim
outer_facets = facet_tags.find(1)
hole_facets  = facet_tags.find(2)
dofs_out  = fem.locate_dofs_topological(V, tdim-1, outer_facets)
dofs_hole = fem.locate_dofs_topological(V, tdim-1, hole_facets)
bc_out  = fem.dirichletbc(uD_out,  dofs_out,  V)
bc_hole = fem.dirichletbc(uD_hole, dofs_hole, V)
bcs = [bc_out, bc_hole]
print("[BC] Dirichlet set: outer→0V, hole→Vinc; dofs:", len(dofs_out), len(dofs_hole))

    # Quick debug of tags (rank 0)
    if comm.rank == 0:
        try:
            print("[tags] cell unique:", np.unique(cell_tags.values))
            print("[tags] facet unique:", np.unique(facet_tags.values))
        except Exception:
            pass

    # --- FE space & RHS (after domain exists) ---
    V = get_function_space(domain)
    f_expr = fem.Constant(domain, PETSc.ScalarType(0.0))  # Laplace by default

    # --- Dirichlet BCs by names (outer = 0, inclusion_boundary = Vinc) ---
    dirich = {"outer": 0.0, "inclusion_boundary": float(args.Vinc)}

    # --- ε field ---
    if eps_from_materials is not None:
        eps_fun = eps_from_materials(domain, cell_tags)
    else:
        eps_fun = fem.Constant(domain, PETSc.ScalarType(1.0))

    # --- Solve (handle both solver signatures) ---
    try:
        uh, V, (dx, ds), diag = solve_poisson(
            V=V,
            f=f_expr,
            eps_fun=eps_fun,
            dirichlet=dirich,
            neumann=None,
            ksp_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10, "ksp_max_it": 500},
        )
    except TypeError:
        uh, V, (dx, ds), diag = solve_poisson(
            domain=domain,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
# --- Enforce Dirichlet BCs (outer=0V on tag=1, hole=Vinc on tag=2) ---
V = fem.functionspace(domain, ("Lagrange", 1))
uD_out  = fem.Constant(domain, PETSc.ScalarType(0.0))
uD_hole = fem.Constant(domain, PETSc.ScalarType(args.Vinc))
tdim = domain.topology.dim
outer_facets = facet_tags.find(1)
hole_facets  = facet_tags.find(2)
dofs_out  = fem.locate_dofs_topological(V, tdim-1, outer_facets)
dofs_hole = fem.locate_dofs_topological(V, tdim-1, hole_facets)
bc_out  = fem.dirichletbc(uD_out,  dofs_out,  V)
bc_hole = fem.dirichletbc(uD_hole, dofs_hole, V)
bcs = [bc_out, bc_hole]
print("[BC] Dirichlet set: outer→0V, hole→Vinc; dofs:", len(dofs_out), len(dofs_hole))
            f=f_expr,
            eps_fun=eps_fun,
            dirichlet=dirich,
            neumann=None,
            ksp_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10, "ksp_max_it": 500},
        )

    # --- Diagnostics ---
    umin = float(uh.x.array.min())
    umax = float(uh.x.array.max())
    if comm.rank == 0:
        print(f"[solve_shapes] phi range: min={umin:.6g}, max={umax:.6g}")
        try:
            its = diag.get("ksp_its", None)
            rnorm = diag.get("ksp_rnorm", None)
            print(f"[KSP] iters={its}, rnorm={rnorm}")
        except Exception:
            pass

    # --- Output ---
    outpath = Path(args.outfile).with_suffix(".xdmf")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(comm, str(outpath), "w") as xfile:
        xfile.write_mesh(domain)
        uh.name = "phi"
        xfile.write_function(uh)

    # Clean up Gmsh
    try:
        gmsh.finalize()
    except Exception:
        pass


if __name__ == "__main__":
    main()
