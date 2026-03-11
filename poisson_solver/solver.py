from __future__ import annotations

from typing import List, Optional, Tuple

import ufl
from petsc4py import PETSc
from dolfinx import fem

try:
    from dolfinx.fem.petsc import LinearProblem
except Exception as e:
    raise RuntimeError(
        "This DOLFINx build does not provide dolfinx.fem.petsc.LinearProblem."
    ) from e


def solve_scalar_problem(
    V,
    coefficient,
    bcs: List[fem.DirichletBC],
    rhs_expr: Optional[ufl.core.expr.Expr] = None,
    name: str = "phi",
    petsc_options_prefix: str = "poisson_",
    petsc_options: Optional[dict] = None,
) -> Tuple[fem.Function, Optional[PETSc.KSP]]:
    msh = V.mesh
    phi = fem.Function(V, name=name)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(coefficient * ufl.grad(u), ufl.grad(v)) * ufl.dx
    if rhs_expr is None:
        L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx
    else:
        L = rhs_expr * v * ufl.dx

    if petsc_options is None:
        petsc_options = {
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1.0e-12,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 20000,
        }

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        u=phi,
        petsc_options_prefix=petsc_options_prefix,
        petsc_options=petsc_options,
    )
    phi = problem.solve()
    phi.name = name

    ksp = None
    try:
        ksp = problem.solver
    except Exception:
        pass

    return phi, ksp
