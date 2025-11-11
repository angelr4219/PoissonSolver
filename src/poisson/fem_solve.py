from __future__ import annotations
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import fem

def assemble_solve_poisson(domain, V, eps_DG0, rho, bcs=(), options_file: str | None = None):
    """
    Solve -div(eps grad u) = rho in CG(k) space V with DG0 eps and supplied BCs.

    Args:
        domain: dolfinx mesh
        V: FunctionSpace ("CG", k)
        eps_DG0: Function in DG0 holding ε (cellwise)
        rho: UFL expression for ρ
        bcs: list of dirichletbc
        options_file: PETSc options file path

    Returns:
        u: fem.Function in V with solution
        ksp: PETSc.KSP (for residuals/timings)
    """
    u = fem.Function(V, name="phi")
    v = ufl.TestFunction(V)
    uh = ufl.TrialFunction(V)

    a_form = ufl.inner(eps_DG0 * ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L_form = rho * v * ufl.dx

    A = fem.petsc.assemble_matrix(fem.form(a_form), bcs=list(bcs))
    A.assemble()
    b = fem.petsc.assemble_vector(fem.form(L_form))
    fem.apply_lifting(b, [fem.form(a_form)], bcs=[list(bcs)])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, list(bcs))

    ksp = PETSc.KSP().create(domain.comm)
    if options_file:
        ksp.setFromOptions()  # Will read global options (pass -options_file at runtime)
    ksp.setOperators(A)
    # Sensible fallbacks if options_file not provided via CLI:
    if not ksp.getType():
        ksp.setType("cg")
        ksp.setTolerances(rtol=1e-9)
        ksp.getPC().setType("hypre")

    x = A.createVecRight()
    ksp.solve(b, x)

    with u.vector.localForm() as u_loc:
        u_loc.array[:] = x.array

    return u, ksp
