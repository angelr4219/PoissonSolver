from __future__ import annotations
from dolfinx import fem
import ufl

EPS0 = 8.8541878128e-12

def build_forms(V, eps_r_field, rhs=None):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (eps_r_field * EPS0) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    if rhs is None:
        L = fem.Constant(V.mesh, 0.0) * v * ufl.dx
    else:
        L = rhs * v * ufl.dx
    return a, L, u, v
