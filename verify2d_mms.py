from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

comm = MPI.COMM_WORLD
rank = comm.rank

# --- Domain and mesh (unit square) ---
domain = mesh.create_rectangle(
    comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
    n=(64, 64), cell_type=mesh.CellType.triangle
)
tdim = domain.topology.dim
fdim = tdim - 1

# Ensure needed connectivity for DG0 dof location
domain.topology.create_connectivity(tdim, tdim)

# --- DG0 permittivity: ε = 1 for x<0.5, ε = 3 for x>=0.5 ---
W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
eps = fem.Function(W, name="epsilon")
cells_left  = mesh.locate_entities(domain, tdim, lambda x: x[0] < 0.5)
cells_right = mesh.locate_entities(domain, tdim, lambda x: x[0] >= 0.5)
dofs_left   = fem.locate_dofs_topological(W, tdim, cells_left)
dofs_right  = fem.locate_dofs_topological(W, tdim, cells_right)
eps.x.array.fill(0.0)
eps.x.array[dofs_left]  = 1.0
eps.x.array[dofs_right] = 3.0

# --- Exact solution (manufactured) ---
pi = np.pi
x = ufl.SpatialCoordinate(domain)
u_exact = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])

# With cellwise-constant ε, inside each cell:  -div(ε grad u) = -ε Δu
f = (eps) * (2.0 * pi**2) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])

# --- Boundary tags: 1:x=0, 2:x=1, 3:y=0, 4:y=1 ---
def on_x0(x): return np.isclose(x[0], 0.0)
def on_x1(x): return np.isclose(x[0], 1.0)
def on_y0(x): return np.isclose(x[1], 0.0)
def on_y1(x): return np.isclose(x[1], 1.0)
facets_x0 = mesh.locate_entities_boundary(domain, fdim, on_x0)
facets_x1 = mesh.locate_entities_boundary(domain, fdim, on_x1)
facets_y0 = mesh.locate_entities_boundary(domain, fdim, on_y0)
facets_y1 = mesh.locate_entities_boundary(domain, fdim, on_y1)

indices = np.concatenate([facets_x0, facets_x1, facets_y0, facets_y1])
values  = np.concatenate([np.full_like(facets_x0, 1),
                          np.full_like(facets_x1, 2),
                          np.full_like(facets_y0, 3),
                          np.full_like(facets_y1, 4)])
order = np.argsort(indices)
mt = mesh.meshtags(domain, fdim, indices[order], values[order])

dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

# --- FE space and BCs (Dirichlet on x=0, x=1, y=0; Neumann on y=1) ---
V = fem.functionspace(domain, ("Lagrange", 1))
uD = fem.Function(V)
uD_expr = fem.Expression(u_exact, V.element.interpolation_points())
uD.interpolate(uD_expr)

dofs_x0 = fem.locate_dofs_topological(V, fdim, facets_x0)
dofs_x1 = fem.locate_dofs_topological(V, fdim, facets_x1)
dofs_y0 = fem.locate_dofs_topological(V, fdim, facets_y0)
bcs = [fem.dirichletbc(uD, dofs_x0),
       fem.dirichletbc(uD, dofs_x1),
       fem.dirichletbc(uD, dofs_y0)]

# Neumann on y=1: g = (ε ∇u_exact)·n, n = +ŷ
n = ufl.as_vector((0.0, 1.0))
g = ufl.dot((eps)*ufl.grad(u_exact), n)

# --- Variational problem ---
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(eps*ufl.grad(u), ufl.grad(v)) * dx
L = (f * v) * dx + (g * v) * ds(4)

problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
    petsc_options_prefix="ver2d_"
)
uh = problem.solve()
uh.name = "phi"

# --- Error norms: L2 and H1-seminorm ---
ue = fem.Function(V); ue.interpolate(uD_expr)
e = uh - ue
L2  = np.sqrt(fem.assemble_scalar(fem.form(e**2 * dx)))
H1s = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx)))

if rank == 0:
    print(f"[verify2d]  L2  error = {L2:.6e}")
    print(f"[verify2d] |H1| error = {H1s:.6e}")

# --- Output XDMF/HDF5 ---
with XDMFFile(comm, "results/verify2d_mms.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(ue)
    xdmf.write_function(eps)
