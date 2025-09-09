#!/usr/bin/env python3
"""
solve_variable_eps.py
Piecewise or continuous ε(x) on basic square; supports Si/Ge/SiO2/air.

Examples:
  # Piecewise: left half Si, right half Ge
  python -m src.cli.solve_variable_eps --mode piecewise --Lx 1 --Ly 1 --h 0.04 \
    --split-x 0.5 --eps-left Si --eps-right Ge --phi-left 0.0 --phi-right 0.2 --outfile results/phi_var_eps

  # Continuous Gaussian ε(x)
  python -m src.cli.solve_variable_eps --mode continuous --Lx 1 --Ly 1 --h 0.04 \
    --phi-left 0.0 --phi-right 0.2 --eps0 3.9 --gauss 0.5 0.5 0.2 5.0

"""
from __future__ import annotations
import argparse
import numpy as np, ufl
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile
from src.physics.permittivity import EPS0, DEFAULT_EPS_R
from src.solver.poisson import solve_poisson

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["piecewise","continuous"], default="piecewise")
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.04)
    ap.add_argument("--phi-left", type=float, default=0.0)
    ap.add_argument("--phi-right", type=float, default=0.2)
    # piecewise
    ap.add_argument("--split-x", type=float, default=0.5)
    ap.add_argument("--eps-left", type=str, default="Si")
    ap.add_argument("--eps-right", type=str, default="Ge")
    # continuous gaussian: cx cy sigma scale
    ap.add_argument("--gauss", type=float, nargs=4, default=None)
    ap.add_argument("--eps0", type=float, default=3.9)
    ap.add_argument("--outfile", type=str, default="results/phi_var_eps")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0,0],[args.Lx,args.Ly]], [max(2,int(args.Lx/args.h)), max(2,int(args.Ly/args.h))], mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Facet tagging for left/right boundaries
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim-1, tdim)

    def left(x):  return np.isclose(x[0], 0.0)
    def right(x): return np.isclose(x[0], args.Lx)
    facets_left  = np.where(left(domain.geometry.x.T))[0]
    facets_right = np.where(right(domain.geometry.x.T))[0]

    # Build eps_fun
    if args.mode == "piecewise":
        erL = DEFAULT_EPS_R.get(args.eps_left, float(args.eps_left))
        erR = DEFAULT_EPS_R.get(args.eps_right, float(args.eps_right))
        W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
        eps_fun = fem.Function(W)
        # cell-wise set by centroid
        num_cells = domain.topology.index_map(tdim).size_local
        values = np.empty(num_cells, dtype=np.float64)
        x = domain.geometry.x
        c2x = fem.compute_midpoints(domain, tdim, np.arange(num_cells, dtype=np.int32))
        for cid in range(num_cells):
            cx = c2x[cid,0]
            values[cid] = EPS0 * (erL if cx < args.split_x else erR)
        eps_fun.x.array[:] = values
    else:
        # continuous εr(x) = eps0 + scale*exp(-((x-cx)^2+(y-cy)^2)/(2 sigma^2))
        cx, cy, sigma, scale = args.gauss
        W = fem.functionspace(domain, ("Lagrange", 1))
        x = ufl.SpatialCoordinate(domain)
        er_expr = args.eps0 + scale*ufl.exp(-((x[0]-cx)**2 + (x[1]-cy)**2)/(2*sigma**2))
        eps_fun = fem.Function(W)
        eps_fun.interpolate(fem.Expression(EPS0*er_expr, W.element.interpolation_points()))

    # Dirichlet on left/right
    facet_tags = None; facet_names = None
    # simpler: locate dofs geometrically
    bc_left_dofs  = fem.locate_dofs_geometrical(V, left)
    bc_right_dofs = fem.locate_dofs_geometrical(V, right)
    bcs = [
        fem.dirichletbc(PETSc.ScalarType(args.phi_left),  bc_left_dofs, V),
        fem.dirichletbc(PETSc.ScalarType(args.phi_right), bc_right_dofs, V),
    ]

    # Assemble/solve
    f = 0.0
    from src.solver.poisson import build_measures
    dx, ds = build_measures(domain, None, None)
    u = fem.Function(V, name="phi"); v = ufl.TestFunction(V); du = ufl.TrialFunction(V)
    a = ufl.inner(eps_fun*ufl.grad(du), ufl.grad(v))*dx
    L = ufl.inner(f, v)*dx
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-12})
    uh = problem.solve()

    from dolfinx.io import XDMFFile
    with XDMFFile(comm, f"{args.outfile}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain); xdmf.write_function(uh); xdmf.write_function(eps_fun)

    if comm.rank == 0:
        print(f"[variable_eps] phi range: {uh.x.array.min():.4e}..{uh.x.array.max():.4e}")

if __name__ == "__main__":
    main()
