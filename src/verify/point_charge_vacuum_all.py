from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# ---- Parameters ----
q = 1.0
eps0 = 1.0
L = 2.0
nx = ny = nz = 48            # modest for memory
r0 = np.array([0.0, 0.0, 0.2])
sigma = 0.05

# ---- Mesh & spaces ----
domain = mesh.create_box(MPI.COMM_WORLD,
                         [np.array([-L,-L,-L]), np.array([L,L,L])],
                         [nx, ny, nz], cell_type=mesh.CellType.hexahedron)
V  = fem.functionspace(domain, ("Lagrange", 1))                         # CG1 scalar
W  = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,))) # CG1 vector
Q0 = fem.functionspace(domain, ("Discontinuous Lagrange", 0))           # DG0

x = ufl.SpatialCoordinate(domain)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# ---- Gaussian source ----
r2 = (x[0]-r0[0])**2 + (x[1]-r0[1])**2 + (x[2]-r0[2])**2
rho_expr = q / ((np.sqrt(np.pi)*sigma)**3) * ufl.exp(-r2/sigma**2)

# ---- Poisson ----
a = eps0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
Lform = rho_expr * v * ufl.dx

# Dirichlet BC at boundary: phi ≈ q/(4π eps0 r)
def on_boundary(xx):
    return np.isclose(xx[0], -L) | np.isclose(xx[0], L) | \
           np.isclose(xx[1], -L) | np.isclose(xx[1], L) | \
           np.isclose(xx[2], -L) | np.isclose(xx[2], L)
fdim   = domain.topology.dim - 1
facets = mesh.locate_entities_boundary(domain, fdim, on_boundary)
dofs   = fem.locate_dofs_topological(V, fdim, facets)

phi_bc = fem.Function(V)
def phi_exact_callable(X):
    r = np.sqrt((X[0]-r0[0])**2 + (X[1]-r0[1])**2 + (X[2]-r0[2])**2)
    return (q / (4.0*np.pi*eps0)) / np.maximum(r, 1e-12)
phi_bc.interpolate(phi_exact_callable)
bcs = [fem.dirichletbc(phi_bc, dofs)]

problem = LinearProblem(
    a, Lform, bcs=bcs,
    petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-10, "pc_type": "jacobi"},
    petsc_options_prefix="po_"
)
phi = problem.solve(); phi.name = "phi"

# ---- Project E and D to CG1-vector ----
w, zt = ufl.TrialFunction(W), ufl.TestFunction(W)
M = ufl.inner(w, zt) * ufl.dx

E_sym  = -ufl.grad(phi)
E_prob = LinearProblem(
    M, ufl.inner(E_sym, zt) * ufl.dx,
    petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
    petsc_options_prefix="E_"
)
E_proj = E_prob.solve(); E_proj.name = "E"

D_sym  = eps0 * E_sym
D_prob = LinearProblem(
    M, ufl.inner(D_sym, zt) * ufl.dx,
    petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
    petsc_options_prefix="D_"
)
D_proj = D_prob.solve(); D_proj.name = "D"

# ---- Charge density fields ----
# Exact Gaussian (DG0, by evaluating the callable)
rho = fem.Function(Q0, name="rho")
rho.interpolate(lambda X: q/((np.sqrt(np.pi)*sigma)**3) * np.exp(
    -((X[0]-r0[0])**2 + (X[1]-r0[1])**2 + (X[2]-r0[2])**2)/sigma**2))

# Recovered charge from Gauss’s law: rho_rec = -div(D) via L2 projection to DG0
tq, vq = ufl.TrialFunction(Q0), ufl.TestFunction(Q0)
a_q = ufl.inner(tq, vq) * ufl.dx
L_q = ufl.inner(-ufl.div(D_proj), vq) * ufl.dx
Q_prob = LinearProblem(
    a_q, L_q,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="q_"
)
rho_rec = Q_prob.solve(); rho_rec.name = "rho_rec"

# ---- Global checks ----
dx, ds = ufl.dx, ufl.ds
Q_vol  = fem.assemble_scalar(fem.form(rho * dx))
n      = ufl.FacetNormal(domain)
Q_flux = fem.assemble_scalar(fem.form(ufl.inner(D_proj, n) * ds))
if domain.comm.rank == 0:
    print(f"Total charge (volume): {Q_vol:.8e}")
    print(f"Total charge (flux):   {Q_flux:.8e}")

# ---- Write XDMF ----
with io.XDMFFile(domain.comm, "results/point_charge_vacuum_all.xdmf", "w") as xf:
    xf.write_mesh(domain)
    xf.write_function(phi, 0.0)
    xf.write_function(E_proj, 0.0)
    xf.write_function(D_proj, 0.0)
    xf.write_function(rho, 0.0)
    xf.write_function(rho_rec, 0.0)
