# Poisson on a unit square with simple Dirichlet BCs (linear problem)
# Run (Docker):  python3 poisson_linear.py

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io, nls
from dolfinx.fem.petsc import LinearProblem

comm = MPI.COMM_WORLD

# --- Mesh (triangles) ---
Nx = Ny = 96
domain = mesh.create_unit_square(comm, Nx, Ny, mesh.CellType.triangle)

# --- Function space ---
V = fem.functionspace(domain, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# --- Material permittivity (constant here, easily made piecewise later) ---
eps0 = 8.8541878128e-12  # F/m
eps_r = 3.9               # e.g. SiO2 just as a demo
eps = eps_r * eps0

# --- Charge density (volume) ---
# Start with a small uniform positive rho (C/m^3); set 0.0 for Laplace
rho_val = 1e-5  # (C/m^3), tiny just to see curvature; tune as needed
rho = fem.Constant(domain, rho_val)

# --- Boundary conditions (Dirichlet on x=0 and x=1) ---
def left(x):  return np.isclose(x[0], 0.0)
def right(x): return np.isclose(x[0], 1.0)

phi_L = fem.Function(V); phi_L.interpolate(lambda x: np.zeros_like(x[0]))
phi_R = fem.Function(V); phi_R.interpolate(lambda x: 0.2*np.ones_like(x[0]))  # 0.2 V

bc_left  = fem.dirichletbc(phi_L, fem.locate_dofs_geometrical(V, left))
bc_right = fem.dirichletbc(phi_R, fem.locate_dofs_geometrical(V, right))
bcs = [bc_left, bc_right]

# --- Variational form: ∫ eps ∇u·∇v dx = ∫ rho v dx ---
dx = ufl.dx
a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
L = rho * v * dx

# --- Solve ---
problem = LinearProblem(a, L, bcs=bcs, petsc_options={
    "ksp_type": "preonly", "pc_type": "lu"
})
phi = problem.solve()
phi.name = "phi"

# --- Report & write output ---
with io.XDMFFile(comm, "poisson_linear.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(phi)

loc = phi.x.array
if loc.size:
    print(f"phi: min={loc.min():.6g} V, max={loc.max():.6g} V")
