# poisson_linear.py
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem

def poisson_linear(Nx: int = 96, Ny: int = 96,
                   eps_r: float = 3.9, rho_val: float = 1e-5,
                   phi_left: float = 0.0, phi_right: float = 0.2):
    """
    Solve −ε ∇² φ = ρ on (0,1)^2 with Dirichlet on x=0,1 and natural (Neumann=0) on y=0,1.
    Returns the FE solution phi_h (dolfinx.fem.Function).
    """
    comm = MPI.COMM_WORLD
    eps0 = 8.8541878128e-12
    eps = eps0 * eps_r

    domain = mesh.create_unit_square(comm, Nx, Ny, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def left(X):  return np.isclose(X[0], 0.0)
    def right(X): return np.isclose(X[0], 1.0)

    fdim = domain.topology.dim - 1
    facets_L = mesh.locate_entities_boundary(domain, fdim, left)
    facets_R = mesh.locate_entities_boundary(domain, fdim, right)
    dofs_L = fem.locate_dofs_topological(V, fdim, facets_L)
    dofs_R = fem.locate_dofs_topological(V, fdim, facets_R)

    phi_L_fun = fem.Function(V); phi_L_fun.x.array[:] = phi_left
    phi_R_fun = fem.Function(V); phi_R_fun.x.array[:] = phi_right
    bcs = [fem.dirichletbc(phi_L_fun, dofs_L),
           fem.dirichletbc(phi_R_fun, dofs_R)]

    rho = fem.Constant(domain, rho_val)
    a = (eps * ufl.inner(ufl.grad(phi), ufl.grad(v))) * ufl.dx
    L = rho * v * ufl.dx

    phi_h = LinearProblem(a, L, bcs=bcs).solve()

    with io.XDMFFile(comm, "poisson_linear_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain); xdmf.write_function(phi_h)

    if comm.rank == 0:
        arr = phi_h.x.array
        print(f"[Linear] phi: min={arr.min():.3e} V, max={arr.max():.3e} V")
    return phi_h

if __name__ == "__main__":
    poisson_linear()
