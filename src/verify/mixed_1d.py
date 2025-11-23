# verify/mixed_1d.py
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

# Domain [0,1] with N segments
N = 128
domain = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, 1.0])
V = fem.functionspace(domain, ("Lagrange", 1))
x = ufl.SpatialCoordinate(domain)[0]

# Exact solution: u(x) = x^2; then f = -2, u'(1)=2
u_exact = x**2
f = fem.Constant(domain, -2.0)
eps = fem.Constant(domain, 1.0)

# Dirichlet at x=0: u(0) = 0
dof0 = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
bc = fem.dirichletbc(fem.Constant(domain, 0.0), dof0, V)

# Neumann at x=1: grad u · n = u'(1) = 2
n = fem.Constant(domain, 1.0)
g = fem.Constant(domain, 2.0)  # flux value
# Set up ds measure by tagging facets
fdim = domain.topology.dim - 1
facets = mesh.locate_entities_boundary(domain, fdim,
                                       lambda X: np.logical_or(np.isclose(X[0], 0.0),
                                                               np.isclose(X[0], 1.0)))
values = np.array([1 if np.isclose(xi,0.0) else 2 for xi in domain.geometry.x[facets,0]], dtype=np.int32)
mt = mesh.meshtags(domain, fdim, facets, values)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(eps*ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx + g * v * ufl.ds(subdomain_data=mt, subdomain_id=2)

problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
uh = problem.solve()
# Compute error at nodes
ue = fem.Function(V); ue.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
e = uh - ue
err_Linf = np.max(np.abs(e.x.array))
if MPI.COMM_WORLD.rank == 0:
    print(f"L-infinity error = {err_Linf:.3e}")
