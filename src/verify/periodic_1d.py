# verify/periodic_1d.py
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx_mpc import MultiPointConstraint, LinearProblem

# Create a 1D mesh of [0,1]
n = 128
domain = mesh.create_interval(MPI.COMM_WORLD, n, [0.0, 1.0])
V = fem.functionspace(domain, ("Lagrange", 1))

# Periodic boundary: x=0 and x=1 should be identified
tol = 1e-12
def periodic_boundary(x):
    return np.isclose(x[0], 1.0, atol=tol)
def periodic_relation(x):
    # map x=1 to x=0
    out = np.empty_like(x)
    out[0] = x[0] - 1.0
    return out

# Essential BC: None; but we need to fix gauge. Set mean(u) = 0 by imposing DOF(0)=0
zero_dof = fem.locate_dofs_geometrical(V, lambda X: np.isclose(X[0], 0.0))
gauge_bc = fem.dirichletbc(fem.Constant(domain, 0.0), zero_dof, V)

# Build MPC
mpc = MultiPointConstraint(V)
mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, [gauge_bc])
mpc.finalize()

# Right-hand side: f = sin(2πx)  [source for -u'' = f, u = sin(2πx)/(4π²)]
x = ufl.SpatialCoordinate(domain)[0]
u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
f = ufl.sin(2 * np.pi * x)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
problem = LinearProblem(a, L, mpc, bcs=[gauge_bc])
uh = problem.solve()

# Exact solution: u = 1/(4π²) sin(2πx)
exact_expr = fem.Expression(lambda x: (1/(4*np.pi**2))*np.sin(2*np.pi*x[0]), V.element.interpolation_points)
ue = fem.Function(V); ue.interpolate(exact_expr)
# Error norm
e = uh - ue
L2_error = np.sqrt(fem.assemble_scalar(fem.form(e**2 * ufl.dx)))
if MPI.COMM_WORLD.rank == 0:
    print(f"L2 error = {L2_error:.3e}")
