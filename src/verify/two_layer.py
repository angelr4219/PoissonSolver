# verify/two_layer.py
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
import matplotlib.pyplot as plt

def solve(L=1.0, H=1.0, nx=100, ny=100, eps1=12.0, eps2=4.0,
          q=1.0, sigma=0.02, y_source=0.25, degree=1, plot=False):
    # Box [-L,L] × [-H,H] discretized with triangles
    domain = mesh.create_rectangle(MPI.COMM_WORLD,
                                   [np.array([-L, -H]), np.array([L, H])],
                                   n=(nx, ny), cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    # DG0 permittivity: eps1 for y>0, eps2 for y<0
    DG0 = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(DG0, name="epsilon")
    # assign bottom half default
    eps.x.array[:] = eps2
    # assign top half
    top_cells = mesh.locate_entities(domain, domain.topology.dim,
                                     lambda X: X[1] > 0.0)
    dofs_top = fem.locate_dofs_topological(DG0, domain.topology.dim, top_cells)
    eps.x.array[dofs_top] = eps1

    # Source term: Gaussian charge at (0, y_source)
    def gaussian_rho(X):
        r2 = (X[0])**2 + (X[1] - y_source)**2
        return q / (2.0*np.pi*sigma**2) * np.exp(-r2/(2.0*sigma**2))
    rho_expr = fem.Expression(gaussian_rho, V.element.interpolation_points)
    rho = fem.Function(V); rho.interpolate(rho_expr)

    # Dirichlet BC: phi = 0 on outer boundary
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim,
                                           lambda X: np.full(X.shape[1], True))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(fem.Constant(domain, 0.0), dofs, V)

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
    uh = problem.solve()

    # Sample phi just above and below y=0 along x-axis
    if MPI.COMM_WORLD.rank == 0:
        xs = np.linspace(-L, L, 801)
        pts_top = np.column_stack((xs, np.full_like(xs, 1e-3)))
        pts_bot = np.column_stack((xs, -np.full_like(xs, 1e-3)))
        phi_top = np.array([uh.eval(p)[0] for p in pts_top])
        phi_bot = np.array([uh.eval(p)[0] for p in pts_bot])
        max_jump = np.max(np.abs(phi_top - phi_bot))
        print(f"max |phi(0+) - phi(0-)| = {max_jump:.3e}")

        # Compute normal flux above/below: E_y = ∂phi/∂y
        grad_phi = uh.gradient()
        Ey_top = np.array([grad_phi.eval(p)[0][1] for p in pts_top])
        Ey_bot = np.array([grad_phi.eval(p)[0][1] for p in pts_bot])
        flux_jump = np.max(np.abs(eps1*Ey_top - eps2*Ey_bot))
        print(f"max |eps1*Ey_top - eps2*Ey_bot| = {flux_jump:.3e}")

        if plot:
            plt.figure()
            plt.plot(xs, phi_top, label="phi(y=+)")
            plt.plot(xs, phi_bot, label="phi(y=-)", linestyle="--")
            plt.legend(); plt.title("Interface continuity of phi")
            plt.figure()
            plt.plot(xs, eps1*Ey_top, label="eps1 E_y top")
            plt.plot(xs, eps2*Ey_bot, label="eps2 E_y bottom", linestyle="--")
            plt.legend(); plt.title("Normal flux continuity")
            plt.show()
    return uh

if __name__ == "__main__":
    solve(plot=True)
