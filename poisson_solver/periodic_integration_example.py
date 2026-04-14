from __future__ import annotations

import argparse

import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem

from poisson_solver import create_periodic_mpc


def add_periodic_cli(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--periodic",
        choices=["none", "x", "y"],
        default="none",
        help="Optional one-direction periodic MPC.",
    )
    parser.add_argument(
        "--periodic_scale",
        type=float,
        default=1.0,
        help="Constraint scale for MPC. Normally leave at 1.0.",
    )
    parser.add_argument(
        "--periodic_tol",
        type=float,
        default=None,
        help="Optional geometric tolerance for periodic matching.",
    )


def solve_with_optional_periodic(domain, V, epsilon, rho, bcs, args):
    """
    Drop-in pattern for your main Poisson solve.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    V      : scalar CG function space
    epsilon: DG0/compatible coefficient function
    rho    : fem.Function, fem.Constant, or UFL expr
    bcs    : list of DirichletBC
    args   : parsed argparse namespace with periodic fields
    """
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(epsilon * ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-12,
        "ksp_atol": 1.0e-14,
        "ksp_max_it": 10000,
    }

    if args.periodic == "none":
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
            direction=args.periodic,
            bcs=bcs,
            scale=args.periodic_scale,
            tol=args.periodic_tol,
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

    comm = domain.comm
    local_min = float(phi_h.x.array.min()) if phi_h.x.array.size else 0.0
    local_max = float(phi_h.x.array.max()) if phi_h.x.array.size else 0.0
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print(f"[solve] periodic={args.periodic}")
        print(f"[solve] phi min={gmin:.12e}  phi max={gmax:.12e}")
        print(f"[solve] KSP iterations={solver.getIterationNumber()}")
        print(f"[solve] KSP converged reason={solver.getConvergedReason()}")

    return phi_h, mpc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_periodic_cli(parser)
    args = parser.parse_args()
    if MPI.COMM_WORLD.rank == 0:
        print("This file is a solve-hook example. Import its pattern into your real solver.")
