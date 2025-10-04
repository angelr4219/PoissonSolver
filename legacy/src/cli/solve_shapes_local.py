#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import ufl, gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile

# ---------- Gmsh helpers ----------
def gmsh_preamble(h: float):
    gmsh.initialize()
    gmsh.model.add("shapes2d")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)

def finalize_to_dolfinx(comm: MPI.Comm, gdim: int = 2):
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)
    out = gmshio.model_to_mesh(gmsh.model, comm, rank=comm.rank, gdim=gdim)
    return out[0], out[1], out[2]

def build_disk2d(comm: MPI.Comm, Lx: float, Ly: float, R: float, h: float):
    gmsh_preamble(h)
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, Lx, Ly)
    cx, cy = 0.5*Lx, 0.5*Ly
    disk = occ.addDisk(cx, cy, 0, R, R)
    occ.fragment([(2, rect)], [(2, disk)])
    occ.synchronize()
    surfs = gmsh.model.getEntities(2)
    areas = [(s[1], gmsh.model.occ.getMass(2, s[1])) for s in surfs]
    inc = min(areas, key=lambda x: x[1])[0]
    bulk = max(areas, key=lambda x: x[1])[0]
    gmsh.model.addPhysicalGroup(2, [bulk], 1)
    gmsh.model.addPhysicalGroup(2, [inc],  2)
    inc_curves  = [e[1] for e in gmsh.model.getBoundary([(2, inc)],  oriented=False) if e[0] == 1]
    bulk_curves = [e[1] for e in gmsh.model.getBoundary([(2, bulk)], oriented=False) if e[0] == 1]
    outer_curves = list(set(bulk_curves) - set(inc_curves))
    gmsh.model.addPhysicalGroup(1, outer_curves, 1)
    gmsh.model.addPhysicalGroup(1, inc_curves,   2)
    return finalize_to_dolfinx(comm)

def build_rod2d(comm: MPI.Comm, Lx: float, Ly: float, R: float, h: float):
    gmsh_preamble(h)
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, Lx, Ly)
    cx, cy = 0.5*Lx, 0.5*Ly
    rod = occ.addRectangle(cx - R, cy - R, 0, 2*R, 2*R)
    occ.fragment([(2, rect)], [(2, rod)])
    occ.synchronize()
    surfs = gmsh.model.getEntities(2)
    areas = [(s[1], gmsh.model.occ.getMass(2, s[1])) for s in surfs]
    inc = min(areas, key=lambda x: x[1])[0]
    bulk = max(areas, key=lambda x: x[1])[0]
    gmsh.model.addPhysicalGroup(2, [bulk], 1)
    gmsh.model.addPhysicalGroup(2, [inc],  2)
    inc_curves  = [e[1] for e in gmsh.model.getBoundary([(2, inc)],  oriented=False) if e[0] == 1]
    bulk_curves = [e[1] for e in gmsh.model.getBoundary([(2, bulk)], oriented=False) if e[0] == 1]
    outer_curves = list(set(bulk_curves) - set(inc_curves))
    gmsh.model.addPhysicalGroup(1, outer_curves, 1)
    gmsh.model.addPhysicalGroup(1, inc_curves,   2)
    return finalize_to_dolfinx(comm)

# ---------- FE helpers ----------
def get_function_space(domain):
    try:
        return fem.FunctionSpace(domain, ("Lagrange", 1))
    except Exception:
        return fem.functionspace(domain, ("Lagrange", 1))

def solve_local(domain, facet_tags, Vinc: float):
    V = get_function_space(domain)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps = fem.Constant(domain, PETSc.ScalarType(1.0))
    f   = fem.Constant(domain, PETSc.ScalarType(0.0))
    dx = ufl.Measure("dx", domain=domain)

    # Dirichlet BCs by facet IDs
    outer_dofs = fem.locate_dofs_topological(V, 1, facet_tags.find(1))
    inc_dofs   = fem.locate_dofs_topological(V, 1, facet_tags.find(2))
    bc_outer = fem.dirichletbc(PETSc.ScalarType(0.0),  outer_dofs, V)
    bc_inc   = fem.dirichletbc(PETSc.ScalarType(Vinc), inc_dofs,   V)
    bcs = [bc_outer, bc_inc]

    a = ufl.inner(eps*ufl.grad(u), ufl.grad(v))*dx
    L = f*v*dx

    # Assemble (version-agnostic petsc helpers)
    try:
        from dolfinx.fem import petsc as fem_petsc
    except Exception:
        fem_petsc = fem.petsc

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = fem_petsc.assemble_matrix(a_form, bcs=bcs); A.assemble()
    b = fem_petsc.assemble_vector(L_form)
    fem_petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    # Solve with PETSc KSP into Vec, then copy to uh.x
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("cg"); ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, max_it=500)

    uh = fem.Function(V)
    x = A.createVecRight()
    ksp.solve(b, x)
    uh.x.array[:] = x.getArray()
    uh.x.scatter_forward()

    diag = {"ksp_its": ksp.getIterationNumber(), "ksp_rnorm": ksp.getResidualNorm()}
    return uh, V, diag

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Local Poisson solver with explicit Dirichlet facet tags.")
    p.add_argument("--mode", choices=["disk2d", "rod2d"], required=True)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--R",  type=float, default=0.15)
    p.add_argument("--h",  type=float, default=0.03)
    p.add_argument("--Vinc", type=float, default=0.3)
    return p.parse_args()

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD

    # Geometry
    if args.mode == "disk2d":
        domain, cell_tags, facet_tags = build_disk2d(comm, args.Lx, args.Ly, args.R, args.h)
    else:
        domain, cell_tags, facet_tags = build_rod2d(comm, args.Lx, args.Ly, args.R, args.h)

    # Solve
    uh, V, diag = solve_local(domain, facet_tags, args.Vinc)

    # Diagnostics
    umin = float(uh.x.array.min()); umax = float(uh.x.array.max())
    if comm.rank == 0:
        print("[local] facet tags:", np.unique(facet_tags.values))
        print(f"[local] phi range: min={umin:.6g}, max={umax:.6g} (target Vinc={args.Vinc})")
        print(f"[KSP] iters={diag['ksp_its']}, rnorm={diag['ksp_rnorm']}")

    # Output: Desktop/poissonresults, timestamp + mode tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("/app/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xdmf = out_dir / f"phi_{args.mode}_{timestamp}.xdmf"
    out_h5   = out_dir / f"phi_{args.mode}_{timestamp}.h5"

    with XDMFFile(comm, str(out_xdmf), "w") as xf:
        xf.write_mesh(domain)
        uh.name = "phi"
        xf.write_function(uh)

    # Print explicit paths so you always know where the files are
    if comm.rank == 0:
        print(f"[out] wrote: {out_xdmf}")
        print(f"[out] wrote: {out_h5}")

    try:
        gmsh.finalize()
    except Exception:
        pass

if __name__ == "__main__":
    main()
