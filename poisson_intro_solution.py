# poisson_mms_fenicsx.py
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import Function
from dolfinx.fem.petsc import LinearProblem

comm = MPI.COMM_WORLD

# --- Mesh ---
Nx = Ny = 64
domain = mesh.create_unit_square(comm, Nx, Ny, mesh.CellType.triangle)

# --- FE space ---
V = fem.functionspace(domain, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# --- Exact solution u_e and f = -6 ---
x = ufl.SpatialCoordinate(domain)
u_exact_expr = 1 + x[0]**2 + 2*x[1]**2
f_expr = fem.Constant(domain, -6.0)

# --- Dirichlet BC on all boundaries: u = u_e ---
u_D = fem.Function(V)
u_D_expr = fem.Expression(u_exact_expr, V.element.interpolation_points())
u_D.interpolate(u_D_expr)

def on_boundary(x):
    tol = 1e-14
    return np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], 1.0, atol=tol) | \
           np.isclose(x[1], 0.0, atol=tol) | np.isclose(x[1], 1.0, atol=tol)

facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, on_boundary)
dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, facets)
bc = fem.dirichletbc(u_D, dofs)

# --- Weak form: ∫ ∇u·∇v dx = ∫ f v dx
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f_expr * v * ufl.dx

problem = LinearProblem(a, L, bcs=[bc])
uh = problem.solve()

# --- Error norms ---
# L2: ||u - u_e||_L2
u_exact = fem.Function(V)
u_exact.interpolate(u_D_expr)

e = fem.Function(V)
e.x.array[:] = uh.x.array - u_exact.x.array
L2_error = np.sqrt(fem.assemble_scalar(fem.form(e*e*ufl.dx)))
# "Max" error at nodal points (FE L∞ proxy)
max_error = np.max(np.abs(e.x.array))

if comm.rank == 0:
    print(f"L2 error: {L2_error:.6e}, max (nodal) error: {max_error:.6e}")

# --- Save ---
with io.XDMFFile(comm, "mms_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

