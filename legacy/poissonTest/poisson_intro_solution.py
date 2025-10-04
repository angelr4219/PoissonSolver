# poisson_intro_solution.py
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
        return (np.isclose(X[0], 0.0, atol=tol)
                | np.isclose(X[0], 1.0, atol=tol)
                | np.isclose(X[1], 0.0, atol=tol)
                | np.isclose(X[1], 1.0, atol=tol))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, on_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_D, dofs)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    uh = LinearProblem(a, L, bcs=[bc]).solve()

    # Optional: write XDMF
    with io.XDMFFile(comm, "mms_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)

    # L2 and nodal “max” errors (for sanity)
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points()))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    L2_err = (fem.assemble_scalar(fem.form(e * e * ufl.dx)))**0.5
    if comm.rank == 0:
        print(f"[MMS] L2 error = {L2_err:.6e}, nodal max = {np.max(np.abs(e.x.array)):.6e}")

    return uh


# --- Extra diagnostics & verification ----------------------------------------
def extra_checks(uh, V=None):
    """
    More verification: H1-seminorm error and boundary flux check.
    For this MMS, ∮ ∂u/∂n ds should be 6 exactly (since Δu = 6 over unit area).
    """
    if V is None:
        V = uh.function_space
    domain = V.mesh
    x = ufl.SpatialCoordinate(domain)

    # Exact solution on V for comparisons
    u_exact_expr = 1 + x[0]**2 + 2*x[1]**2
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points()))

    # Error function
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array

    # H1-seminorm (energy) error: ||∇e||_{L2}
    H1_semi_err = np.sqrt(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    )

    # Boundary flux ∮ ∂u/∂n ds ≈ 6
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    n = ufl.FacetNormal(domain)
    flux = fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(uh), n) * ufl.ds))

    if MPI.COMM_WORLD.rank == 0:
        print(f"[MMS] H1-semi error = {H1_semi_err:.6e}")
        print(f"[MMS] Boundary flux ∮∂u/∂n ds ≈ {flux:.6f} (expected 6.000000)")


def convergence_study(levels=(8, 16, 32, 64)):
    """
    Tiny grid-refinement study for the MMS problem.
    Prints L2 errors and observed rates (should be ~O(h^2) for P1).
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("\nN     L2-error         rate")
    prev = None
    for N in levels:
        uh = poisson_mms(N, N)
        V = uh.function_space
        domain = V.mesh
        x = ufl.SpatialCoordinate(domain)

        # Build exact & error
        u_exact_expr = 1 + x[0]**2 + 2*x[1]**2
        u_exact = fem.Function(V)
        u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points()))
        e = fem.Function(V)
        e.x.array[:] = uh.x.array - u_exact.x.array

        L2_err = np.sqrt(fem.assemble_scalar(fem.form(e * e * ufl.dx)))
        rate = np.log(prev / L2_err) / np.log(2.0) if prev is not None else float("nan")
        if comm.rank == 0:
            print(f"{N:<4d}  {L2_err: .3e}    {rate: .2f}")
        prev = L2_err


if __name__ == "__main__":
    uh = poisson_mms()                # single solve
    extra_checks(uh, uh.function_space)
    # Uncomment to run a quick refinement study:
    # convergence_study((8, 16, 32, 64, 128))
