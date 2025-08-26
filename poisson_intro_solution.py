# poisson_mms.py
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem

def poisson_mms(Nx: int = 64, Ny: int = 64):
    """
    MMS: Solve −Δu = f with f = −6, exact Dirichlet u_e = 1 + x^2 + 2y^2 on all boundaries.
    Returns the FE solution uh (dolfinx.fem.Function).
    """
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, Nx, Ny, mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = 1 + x[0]**2 + 2*x[1]**2
    f_expr = fem.Constant(domain, -6.0)

    # Interpolate exact solution onto V for BCs
    u_D = fem.Function(V)
    u_D.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points()))

    def on_boundary(X):
        tol = 1e-14
        return np.isclose(X[0], 0.0, atol=tol) | np.isclose(X[0], 1.0, atol=tol) | \
               np.isclose(X[1], 0.0, atol=tol) | np.isclose(X[1], 1.0, atol=tol)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, on_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_D, dofs)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    uh = LinearProblem(a, L, bcs=[bc]).solve()

    # Optional: write XDMF + quick error print
    with io.XDMFFile(comm, "mms_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain); xdmf.write_function(uh)

    # L2 and nodal “max” errors (for sanity)
    u_exact = fem.Function(V); u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points()))
    e = fem.Function(V); e.x.array[:] = uh.x.array - u_exact.x.array
    L2_err = (fem.assemble_scalar(fem.form(e*e*ufl.dx)))**0.5
    if comm.rank == 0:
        print(f"[MMS] L2 error = {L2_err:.6e}, nodal max = {np.max(np.abs(e.x.array)):.6e}")

    return uh

if __name__ == "__main__":
    poisson_mms()
