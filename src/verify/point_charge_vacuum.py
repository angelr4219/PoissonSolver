# save as: src/verify/point_charge_vacuum_all.py
from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# ---- Parameters (nondimensional-friendly; set eps0=1 for simplicity) ----
q = 1.0            # point charge magnitude
eps0 = 1.0         # vacuum permittivity (use SI value if you prefer)
L = 2.0            # domain half-size -> box = [-L,L]^3
nx = ny = nz = 96  # mesh resolution
r0 = np.array([0.0, 0.0, 0.2])  # charge position
sigma = 0.05       # Gaussian width ~ 1-2 mesh cells

# ---- Mesh & spaces ----
domain = mesh.create_box(MPI.COMM_WORLD,
                         [np.array([-L,-L,-L]), np.array([L,L,L])],
                         [nx, ny, nz], cell_type=mesh.CellType.hexahedron)
V = fem.FunctionSpace(domain, ("CG", 1))           # scalar phi
W = fem.VectorFunctionSpace(domain, ("CG", 1))     # vector E, D
Qsp = fem.FunctionSpace(domain, ("DG", 0))         # cellwise rho (and div-based rho_rec)

x = ufl.SpatialCoordinate(domain)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# ---- Source: normalized Gaussian approximating a point charge ----
r2 = (x[0]-r0[0])**2 + (x[1]-r0[1])**2 + (x[2]-r0[2])**2
rho_expr = q / ((np.sqrt(np.pi)*sigma)**3) * ufl.exp(-r2/sigma**2)

# ---- Variational problem: -div(eps0 grad phi) = rho ----
a = eps0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
Lform = rho_expr * v * ufl.dx

# Exact Dirichlet BC at outer boundary: phi = q/(4π eps0 r)
def on_boundary(xx):
    return np.isclose(xx[0], -L) | np.isclose(xx[0], L) | \
           np.isclose(xx[1], -L) | np.isclose(xx[1], L) | \
           np.isclose(xx[2], -L) | np.isclose(xx[2], L)
facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, on_boundary)
dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, facets)
r = ufl.sqrt((x[0]-r0[0])**2 + (x[1]-r0[1])**2 + (x[2]-r0[2])**2)
phi_exact = q / (4.0*np.pi*eps0) * (1.0 / ufl.max_value(r, 1e-12))
phi_bc = fem.Function(V)
phi_bc.interpolate(fem.Expression(phi_exact, V.element.interpolation_points()))
bcs = [fem.dirichletbc(phi_bc, dofs)]

# ---- Solve for phi ----
problem = LinearProblem(a, Lform, bcs=bcs, petsc_options={
    "ksp_type": "cg", "ksp_rtol": 1e-10,
    "pc_type": "hypre", "pc_hypre_type": "boomeramg",
})
phi = problem.solve()

# ---- E field: E = -grad(phi) projected to CG1-vector ----
w, zt = ufl.TrialFunction(W), ufl.TestFunction(W)
E_sym = -ufl.grad(phi)
E_proj = fem.Function(W)
M = ufl.inner(w, zt) * ufl.dx
bE = ufl.inner(E_sym, zt) * ufl.dx
E_prob = LinearProblem(M, bE, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
E_proj = E_prob.solve()

# ---- Displacement field D (in vacuum D = eps0 * E) ----
D_sym = eps0 * E_sym
D_proj = fem.Function(W)
bD = ufl.inner(D_sym, zt) * ufl.dx
D_prob = LinearProblem(M, bD, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
D_proj = D_prob.solve()

# ---- Charge density fields ----
# 1) The exact/known Gaussian source, cell-projected to DG0 for visualization:
rho = fem.Function(Qsp, name="rho")
rho.interpolate(fem.Expression(rho_expr, Qsp.element.interpolation_points()))

# 2) A “recovered” charge from Gauss’s law: rho_rec = -div(D)
rho_rec_expr = -ufl.div(D_proj)
rho_rec = fem.Function(Qsp, name="rho_rec")
rho_rec.interpolate(fem.Expression(rho_rec_expr, Qsp.element.interpolation_points()))

# ---- Global charge checks: volume integral and boundary flux ----
dx = ufl.dx; ds = ufl.ds
Q_vol = fem.assemble_scalar(fem.form(rho * dx))
# outward flux of D over the boundary should equal total free charge
n = ufl.FacetNormal(domain)
flux_form = ufl.inner(D_proj, n) * ds
Q_flux = fem.assemble_scalar(fem.form(flux_form))

if domain.comm.rank == 0:
    print(f"Total charge (volume integral of rho):   {Q_vol:.8e}")
    print(f"Total charge (boundary flux of D⋅n):     {Q_flux:.8e}")

# ---- Write to XDMF ----
with io.XDMFFile(domain.comm, "results/point_charge_vacuum_all.xdmf", "w") as xf:
    xf.write_mesh(domain)
    xf.write_function(phi, 0.0, "phi")
    xf.write_function(E_proj, 0.0, "E")
    xf.write_function(D_proj, 0.0, "D")
    xf.write_function(rho, 0.0, "rho")
    xf.write_function(rho_rec, 0.0, "rho_rec")
