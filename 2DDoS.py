# poisson_2d_dos_fenicsx.py
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io, nls
from dolfinx.fem import Function
from dolfinx.fem.petsc import NonlinearProblem

comm = MPI.COMM_WORLD

# --- Parameters ---
Nx = Ny = 96
eps0 = 8.8541878128e-12
eps_r = 3.9
eps = eps0 * eps_r

rho_vol = 0.0  # C/m^3

phi_left = 0.0
phi_right = 0.2

q  = 1.602176634e-19
kB = 1.380649e-23
hbar = 1.054571817e-34

mu   = 0.10 * q   # 0.10 eV in Joules
Ec0  = 0.0  * q   # conduction band edge offset reference (J)
g_s  = 2.0
g_v  = 1.0
m0   = 9.1093837015e-31
m_eff = 0.19 * m0
T = 300.0
kT = kB * T

g2D = (g_s * g_v * m_eff) / (2.0 * np.pi * hbar * hbar)

# --- Mesh & function space ---
domain = mesh.create_unit_square(comm, Nx, Ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Unknown and test
phi = fem.Function(V)
v = ufl.TestFunction(V)

# --- Boundary markers for ds ---
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

def on_left(x):   return np.isclose(x[0], 0.0)
def on_right(x):  return np.isclose(x[0], 1.0)
def on_bottom(x): return np.isclose(x[1], 0.0)
def on_top(x):    return np.isclose(x[1], 1.0)

facets_left   = mesh.locate_entities_boundary(domain, fdim, on_left)
facets_right  = mesh.locate_entities_boundary(domain, fdim, on_right)
facets_bottom = mesh.locate_entities_boundary(domain, fdim, on_bottom)
facets_top    = mesh.locate_entities_boundary(domain, fdim, on_top)

# Tag: 1=left, 2=right, 3=bottom, 4=top
facet_indices = np.hstack([facets_left, facets_right, facets_bottom, facets_top])
facet_tags    = np.hstack([
    np.full_like(facets_left,   1),
    np.full_like(facets_right,  2),
    np.full_like(facets_bottom, 3),
    np.full_like(facets_top,    4)
])

from dolfinx.mesh import meshtags
facet_mt = meshtags(domain, fdim, facet_indices, facet_tags)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_mt)

# --- Dirichlet BCs on x=0,1 ---
phi_L_fun = fem.Function(V); phi_L_fun.x.array[:] = phi_left
phi_R_fun = fem.Function(V); phi_R_fun.x.array[:] = phi_right

dofs_L = fem.locate_dofs_topological(V, fdim, facets_left)
dofs_R = fem.locate_dofs_topological(V, fdim, facets_right)
bcs = [fem.dirichletbc(phi_L_fun, dofs_L),
       fem.dirichletbc(phi_R_fun, dofs_R)]

# --- Physical constants as UFL Constants ---
eps_c = fem.Constant(domain, eps)
rho_c = fem.Constant(domain, rho_vol)
q_c   = fem.Constant(domain, q)
kT_c  = fem.Constant(domain, kT)
g2D_c = fem.Constant(domain, g2D)
mu_c  = fem.Constant(domain, mu)
Ec0_c = fem.Constant(domain, Ec0)

# --- 2D-DOS surface charge model σ(φ) and its derivative σ'(φ) ---
# σ(φ) = -q * g2D * kT * log(1 + exp(η)), with η = (μ - (Ec0 - qφ))/kT
# dσ/dφ = -q * g2D * kT * ( (exp(η)/(1+exp(η))) * (q/kT) ) = -q^2 * g2D * (exp(η)/(1+exp(η)))
# Use softplus/log1p(exp(.)) style stabilization via ufl.log(1+exp) and clamp η if needed.
eta = (mu_c - (Ec0_c - q_c*phi)) / kT_c
softplus = ufl.ln(1 + ufl.exp(eta))     # log(1 + exp(η))
sigma = -q_c * g2D_c * kT_c * softplus  # σ(φ)

# logistic(η) = exp(η)/(1+exp(η))
logistic = ufl.exp(eta) / (1 + ufl.exp(eta))
dsigma_dphi = - (q_c*q_c) * g2D_c * logistic

# --- Residual and Jacobian for Newton ---
F = eps_c * ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx \
    - rho_c * v * ufl.dx \
    - sigma * v * ds(4)          # top boundary only

# Gateaux derivative: J = dF/d(phi)[w] with trial function w
w = ufl.TrialFunction(V)
J = ufl.derivative(F, phi, w)

# --- Solve nonlinear problem ---
problem = NonlinearProblem(F, phi, bcs=bcs, J=J)
solver = nls.petsc.NewtonSolver(comm, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 50

# Initial guess: linear interp between left and right
with phi.vector.localForm() as loc:
    loc.set(0.0)
# use an explicit interpolate of linear function
x = ufl.SpatialCoordinate(domain)
lin_expr = phi_left + (phi_right - phi_left)*x[0]
phi.interpolate(fem.Expression(lin_expr, V.element.interpolation_points()))

n_it, converged = solver.solve(phi)
if comm.rank == 0:
    print(f"Newton iterations: {n_it}, converged={converged}")

# --- Save & report ---
with io.XDMFFile(comm, "poisson_dos_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(phi)

if comm.rank == 0:
    arr = phi.x.array
    print(f"φ min = {arr.min():.6g}, φ max = {arr.max():.6g}")
    np.save("poisson_dos_solution.npy", arr)
