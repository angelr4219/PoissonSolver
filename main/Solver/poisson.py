# src/poisson.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Dict
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

dx = lambda domain: ufl.Measure("dx", domain=domain)
ds = lambda domain, tags=None: (
    ufl.Measure("ds", domain=domain, subdomain_data=tags)
    if tags is not None else ufl.Measure("ds", domain=domain)
)

@dataclass
class SolveResult:
    uh: fem.Function
    L2_error: Optional[float] = None
    H1_semi_error: Optional[float] = None

def _linear_problem(a, L, bcs, prefix: str, petsc_opts: Optional[Dict[str, str]] = None):
    if petsc_opts is None:
        petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}
    # NOTE: your dolfinx requires non-empty prefix
    return LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts, petsc_options_prefix=prefix)

def solve_dirichlet(
    domain, V, eps, rhs_f, dirichlet_bcs: Iterable[fem.DirichletBC], 
    prefix: str = "dir_"
) -> fem.Function:
    """Solve -div(eps * grad u) = f with Dirichlet BCs only."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * dx(domain)
    L = (rhs_f * v) * dx(domain)
    problem = _linear_problem(a, L, list(dirichlet_bcs), prefix)
    uh = problem.solve()
    uh.name = "phi"
    return uh

def solve_mixed(
    domain, V, eps, rhs_f, dirichlet_bcs: Iterable[fem.DirichletBC],
    neumann_terms: Optional[Iterable[Tuple[ufl.core.expr.Expr, int]]] = None,
    facet_tags: Optional["dolfinx.mesh.MeshTagsMetaClass"] = None,
    prefix: str = "mix_"
) -> fem.Function:
    """
    Solve -div(eps * grad u) = f with Dirichlet and optional Neumann BCs.
    neumann_terms: iterable of (g_expr, tag) meaning add ∫_{Γ_tag} g v ds(tag)
    """
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * dx(domain)
    L = (rhs_f * v) * dx(domain)
    if neumann_terms:
        ds_mt = ds(domain, facet_tags)
        for g, tag in neumann_terms:
            L += (g * v) * ds_mt(tag)
    problem = _linear_problem(a, L, list(dirichlet_bcs), prefix)
    uh = problem.solve()
    uh.name = "phi"
    return uh

def norms(domain, V, uh: fem.Function, u_exact_ufl) -> Tuple[float, float]:
    """Return (L2 error, H1 seminorm error) against a UFL exact solution."""
    ue = fem.Function(V)
    ue_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    ue.interpolate(ue_expr)
    e = uh - ue
    L2  = np.sqrt(fem.assemble_scalar(fem.form(e**2 * dx(domain))))
    H1s = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx(domain))))
    return float(L2), float(H1s)
