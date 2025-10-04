#!/usr/bin/env python3
"""
mms.py
Method of Manufactured Solutions (MMS) for Poisson with constant ε.

We choose u_exact(x, y) = 1 + x^2 + 2 y^2  =>  Δu_exact = 2 + 4 = 6
Problem: - div(ε ∇u) = f  with Dirichlet BC u = u_exact on ∂Ω.
=> f = -ε * Δu_exact = -6 ε

Expected P1 rates on uniform refinements:
  L2 ~ O(h^2),   H1-semi ~ O(h)
"""

from __future__ import annotations

import argparse
import json
import math
from typing import List

import numpy as np
import ufl
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem  

# Use your material constants if available; otherwise define EPS0 here.
try:
    from src.physics.permittivity import EPS0  # preferred
except Exception:
    EPS0 = 8.8541878128e-12  # F/m (SI)

# ---------- exact solution as a Python callable ----------
def u_exact_callable(x: np.ndarray) -> np.ndarray:
    """
    x has shape (gdim, npts); return (npts,) for scalar field.
    u_exact = 1 + x^2 + 2 y^2
    """
    return 1.0 + x[0] ** 2 + 2.0 * x[1] ** 2


def mms_run(N_list: List[int], eps_r: float = 1.0, outprefix: str = "results/mms"):
    comm = MPI.COMM_WORLD

    h_vals: list[float] = []
    E_L2: list[float] = []
    E_H1: list[float] = []

    for N in N_list:
        # --- mesh and spaces
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", 1))
        dx = ufl.Measure("dx", domain=domain)

        # --- epsilon and RHS
        eps_abs = EPS0 * eps_r
        f = fem.Constant(domain, PETSc.ScalarType(-6.0 * eps_abs))  # since Δu_exact = 6

        # Represent ε as DG0 (piecewise-constant)
        W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
        eps_fun = fem.Function(W, name="epsilon")
        nloc = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
        eps_fun.x.array[:nloc] = PETSc.ScalarType(eps_abs)

        # --- exact Dirichlet BC everywhere on boundary
        def on_bdry(x):
            return (
                np.isclose(x[0], 0.0)
                | np.isclose(x[0], 1.0)
                | np.isclose(x[1], 0.0)
                | np.isclose(x[1], 1.0)
            )

        dofs = fem.locate_dofs_geometrical(V, on_bdry)
        uD = fem.Function(V, name="u_D")
        uD.interpolate(u_exact_callable)  # in-place; returns None
        bc = fem.dirichletbc(uD, dofs)

        # --- variational problem: a(u, v) = L(v)
        u  = ufl.TrialFunction(V)
        v  = ufl.TestFunction(V)
        uh = fem.Function(V, name="u")

        a = ufl.inner(eps_fun * ufl.grad(u), ufl.grad(v)) * dx
        L = ufl.inner(f, v) * dx

        problem = LinearProblem(
            a, L, bcs=[bc], u=uh,
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "gamg",      # often faster/portable than hypre on laptops
                "ksp_rtol": 1.0e-10,    # tight enough for clean MMS slopes
                "ksp_max_it": 500,
            },
        )
        uh = problem.solve()


        # --- exact field on V for error computation
        uE = fem.Function(V, name="u_exact")
        uE.interpolate(u_exact_callable)
        
        
        dx_L2 = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
        dx_H1 = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 0})

        # errors
        e = uh - uE
        E_L2_loc = fem.assemble_scalar(fem.form(ufl.inner(e, e) * dx_L2))
        E_H1_loc = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx_H1))


        # errors
        e = uh - uE  # ufl expression via overloaded operators
        E_L2_loc = fem.assemble_scalar(fem.form(ufl.inner(e, e) * dx))
        E_H1_loc = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx))
        comm.Barrier()
        E_L2_glob = comm.allreduce(E_L2_loc, op=MPI.SUM)
        E_H1_glob = comm.allreduce(E_H1_loc, op=MPI.SUM)
        err_L2 = math.sqrt(E_L2_glob)
        err_H1s = math.sqrt(E_H1_glob)

        h = 1.0 / N
        h_vals.append(h)
        E_L2.append(err_L2)
        E_H1.append(err_H1s)

        if comm.rank == 0:
            its = problem.solver.getIterationNumber() if hasattr(problem, "solver") else -1
            print(f"[MMS] N={N:4d}  h={h:.4f}  L2={err_L2:.3e}  H1s={err_H1s:.3e}  it={its}")

    if comm.rank == 0:
        # --- write CSV and slopes
        import pandas as pd
        import pathlib

        pathlib.Path(outprefix).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"h": h_vals, "L2": E_L2, "H1semi": E_H1})
        df.to_csv(f"{outprefix}.csv", index=False)

        # slopes from last 3 points
        def slope(xs, ys):
            x = np.log(xs[-3:])
            y = np.log(ys[-3:])
            m, _b = np.polyfit(x, y, 1)
            return float(m)

        sL2 = slope(h_vals, E_L2) if len(h_vals) >= 3 else float("nan")
        sH1 = slope(h_vals, E_H1) if len(h_vals) >= 3 else float("nan")
        with open(f"{outprefix}_slopes.json", "w") as fjson:
            json.dump({"L2": sL2, "H1semi": sH1}, fjson, indent=2)

        # --- plots
        plt.figure()
        plt.loglog(h_vals, E_L2, "o-", label="L2 error")
        if len(h_vals) > 0:
            ref = (np.array(h_vals) ** 2) * (E_L2[0] / (h_vals[0] ** 2))
            plt.loglog(h_vals, ref, "--", label="~h^2 ref")
        plt.gca().invert_xaxis()
        plt.grid(True, which="both")
        plt.xlabel("h")
        plt.ylabel(r"$\|e\|_{L^2}$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outprefix}_L2.png", dpi=200)

        plt.figure()
        plt.loglog(h_vals, E_H1, "o-", label="H1-semi error")
        if len(h_vals) > 0:
            ref = (np.array(h_vals) ** 1) * (E_H1[0] / (h_vals[0] ** 1))
            plt.loglog(h_vals, ref, "--", label="~h ref")
        plt.gca().invert_xaxis()
        plt.grid(True, which="both")
        plt.xlabel("h")
        plt.ylabel(r"$|e|_{H^1}$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outprefix}_H1.png", dpi=200)

        print(f"[MMS] slopes: L2≈{sL2:.2f}, H1≈{sH1:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--eps_r", type=float, default=1.0)
    ap.add_argument("--outprefix", type=str, default="results/mms")
    args = ap.parse_args()
    mms_run(args.Ns, args.eps_r, args.outprefix)


if __name__ == "__main__":
    main()
