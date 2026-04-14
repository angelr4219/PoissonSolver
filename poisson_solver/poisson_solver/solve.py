from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem

from .common import RANK, global_minmax, make_function_space
from .periodic import create_periodic_mpc


def _locate_box_boundary_dofs(V, Lx, Ly, Lz, periodic: str, tol: float):
    def on_xmin(x):
        return np.isclose(x[0], -0.5 * Lx, atol=tol)

    def on_xmax(x):
        return np.isclose(x[0], +0.5 * Lx, atol=tol)

    def on_ymin(x):
        return np.isclose(x[1], -0.5 * Ly, atol=tol)

    def on_ymax(x):
        return np.isclose(x[1], +0.5 * Ly, atol=tol)

    def on_zmin(x):
        return np.isclose(x[2], -0.5 * Lz, atol=tol)

    def on_zmax(x):
        return np.isclose(x[2], +0.5 * Lz, atol=tol)

    bcs = []

    def add_bc(marker, value=0.0):
        dofs = fem.locate_dofs_geometrical(V, marker)
        if len(dofs) > 0:
            bcs.append(fem.dirichletbc(PETSc.ScalarType(value), dofs, V))

    # Always clamp top and bottom
    add_bc(on_zmin, 0.0)
    add_bc(on_zmax, 0.0)

    # Only clamp non-periodic side directions
    if periodic != "x":
        add_bc(on_xmin, 0.0)
        add_bc(on_xmax, 0.0)

    if periodic != "y":
        add_bc(on_ymin, 0.0)
        add_bc(on_ymax, 0.0)

    return bcs


def solve_poisson_box(
    msh,
    Lx: float,
    Ly: float,
    Lz: float,
    epsilon,
    rho,
    degree: int = 1,
    periodic: str = "none",
    periodic_scale: float = 1.0,
    periodic_tol: float | None = None,
):
    V = make_function_space(msh, ("CG", degree))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    tol = periodic_tol
    if tol is None:
        tol = 1.0e-12 * max(1.0, Lx, Ly, Lz)

    bcs = _locate_box_boundary_dofs(V, Lx, Ly, Lz, periodic=periodic, tol=tol)

    a = ufl.inner(epsilon * ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-12,
        "ksp_atol": 1.0e-14,
        "ksp_max_it": 10000,
    }

    if periodic == "none":
        problem = fem.petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
        )
        phi_h = problem.solve()
        solver = problem.solver
        mpc = None
    else:
        import dolfinx_mpc

        mpc = create_periodic_mpc(
            V,
            direction=periodic,
            bcs=bcs,
            scale=periodic_scale,
            tol=tol,
        )

        problem = dolfinx_mpc.LinearProblem(
            a,
            L,
            mpc,
            bcs=bcs,
            petsc_options=petsc_options,
        )
        phi_h = problem.solve()
        solver = problem.solver

    phi_h.name = "phi"
    gmin, gmax = global_minmax(phi_h)

    if RANK == 0:
        print(f"[solve] periodic={periodic}")
        print(f"[solve] phi min={gmin:.12e}")
        print(f"[solve] phi max={gmax:.12e}")
        print(f"[solve] KSP iterations={solver.getIterationNumber()}")
        print(f"[solve] KSP converged reason={solver.getConvergedReason()}")

    return phi_h, mpc
