from __future__ import annotations
from dolfinx import fem
from petsc4py import PETSc
import ufl

def assemble_solve_poisson(domain, V, eps_DG0, rho, bcs=(), options_file: str | None = None):
    u = fem.Function(V, name="phi")
    v = ufl.TestFunction(V)
    uh = ufl.TrialFunction(V)
    a_form = ufl.inner(eps_DG0 * ufl.grad(uh), ufl.grad(v)) * ufl.dx
    L_form = rho * v * ufl.dx

    A = fem.petsc.assemble_matrix(fem.form(a_form), bcs=list(bcs)); A.assemble()
    b = fem.petsc.assemble_vector(fem.form(L_form))
    fem.apply_lifting(b, [fem.form(a_form)], bcs=[list(bcs)])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, list(bcs))

    ksp = PETSc.KSP().create(domain.comm)
    # Options will be read from PETSC_OPTIONS env or command line
    ksp.setFromOptions()
    if not ksp.getType():  # fallback defaults
        ksp.setType("cg"); ksp.setTolerances(rtol=1e-9); ksp.getPC().setType("hypre")
    ksp.setOperators(A)

    x = A.createVecRight()
    ksp.solve(b, x)
    with u.vector.localForm() as u_loc:
        u_loc.array[:] = x.array
    return u, ksp
