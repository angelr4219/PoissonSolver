# verify/p_refinement.py
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

def solve(deg, n):
    domain = mesh.create_rectangle(MPI.COMM_WORLD,
                                   [np.array([0.0,0.0]), np.array([1.0,1.0])],
                                   n=(n, n), cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", deg))
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(np.pi*x[0]) * ufl.sin(np.pi*x[1])
    f = (deg*(deg+1)) * np.pi**2 * ufl.sin(np.pi*x[0]) * ufl.sin(np.pi*x[1])  # adjust amplitude
    eps = fem.Constant(domain, 1.0)

    u_D = fem.Function(V); u_D.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim,
                                           lambda X: np.full(X.shape[1], True))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_D, dofs, V)
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = ufl.inner(eps*ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
    uh = problem.solve()

    ue = fem.Function(V)
    ue.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    e = uh - ue
    L2  = np.sqrt(fem.assemble_scalar(fem.form(e**2 * ufl.dx)))
    H1s = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx)))
    return L2, H1s

if __name__ == "__main__":
    n = 32  # keep mesh moderately fine
    degrees = [1, 2, 3, 4, 5]
    errs = []
    for d in degrees:
        L2, H1s = solve(d, n)
        if MPI.COMM_WORLD.rank == 0:
            print(f"p={d:2d}  L2={L2:.3e}  H1semi={H1s:.3e}")
        errs.append((L2, H1s))
    if MPI.COMM_WORLD.rank == 0:
        for i in range(1, len(degrees)):
            p_ratio = degrees[i] / degrees[i-1]
            L2_rate = np.log(errs[i-1][0]/errs[i][0]) / np.log(p_ratio)
            H1_rate = np.log(errs[i-1][1]/errs[i][1]) / np.log(p_ratio)
            print(f"p refinement from {degrees[i-1]}→{degrees[i]}: "
                  f"L2 error ratio ≈ {errs[i-1][0]/errs[i][0]:.2e}, H1 ratio ≈ {errs[i-1][1]/errs[i][1]:.2e}")
