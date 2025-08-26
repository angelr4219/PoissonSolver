
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import Function
from dolfinx.fem.petsc import LinearProblem

comm = MPI.COMM_WORLD

# --- Params (edit here) ---
Nx = Ny = 96
eps0 = 8.8541878128e-12
eps_r = 3.9
eps = eps0 * eps_r
rho_val = 1e-5     # C/m^3
phi_left = 0.0
phi_right = 0.2

# --- Mesh ---
domain = mesh.create_unit_square(comm, Nx, Ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
phi = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# --- Boundary locators ---
def left(x):  return np.isclose(x[0], 0.0)
def right(x): return np.isclose(x[0], 1.0)

# Dirichlet on left/right
phi_L_fun = fem.Function(V); phi_L_fun.x.array[:] = phi_left
phi_R_fun = fem.Function(V); phi_R_fun.x.array[:] = phi_right

fdim = domain.topology.dim - 1
facets_L = mesh.locate_entities_boundary(domain, fdim, left)
facets_R = mesh.locate_entities_boundary(domain, fdim, right)
dofs_L = fem.locate_dofs_topological(V, fdim, facets_L)
dofs_R = fem.locate_dofs_topological(V, fdim, facets_R)
bcs = [fem.dirichletbc(phi_L_fun, dofs_L), fem.dirichletbc(phi_R_fun, dofs_R)]

# --- Weak form: ε ∫ ∇φ·∇v dx = ∫ ρ v dx
rho = fem.Constant(domain, rho_val)
a = eps * ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx
L = rho * v * ufl.dx

problem = LinearProblem(a, L, bcs=bcs)
phi_h = problem.solve()

# --- Save & report ---
with io.XDMFFile(comm, "poisson_linear_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(phi_h)

if comm.rank == 0:
    arr = phi_h.x.array
    print(f"phi min = {arr.min():.6g}, phi max = {arr.max():.6g}")
    np.save("poisson_linear_solution.npy", arr)
