"""
periodic_bc_test.py

Standalone periodic BC verification with two known-solution problems.
No physics assumptions — pure math validation.

Box: 100 × 100 × 100 nm   (structured 20×20×20 tet mesh, h≈7nm)
Periodic: x and y    Dirichlet: z=0 and z=Lz

─────────────────────────────────────────────────────
TEST 1  Laplace  (zero source, top/bottom Dirichlet)
  BCs:   φ(z=0) = 0 V,  φ(z=Lz) = -1 V
  Exact: φ(x,y,z) = -z / Lz
  Checks lateral flatness: max |φ(z=50nm) + 0.5| < tol
─────────────────────────────────────────────────────
TEST 2  Poisson  (sinusoidal source, homogeneous Dirichlet)
  Source:  ρ = sin(kx·x)·sin(ky·y)·sin(kz·z)
           kx = 2π/Lx, ky = 2π/Ly, kz = π/Lz
  BCs:   φ(z=0) = φ(z=Lz) = 0
  Exact: φ = ρ / (kx² + ky² + kz²)
  Checks L2 error vs exact solution
─────────────────────────────────────────────────────
Requires dolfinx_mpc. Run in the jorgensd/dolfinx_mpc Docker image
or any container where `pip install fenics-dolfinx-mpc` succeeds.

Outputs (written to working directory):
  periodic_test1_phi.xdmf / .h5
  periodic_test2_phi.xdmf / .h5  (also writes phi_exact and error)
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem
from dolfinx.mesh import create_box, CellType
from dolfinx.io import XDMFFile

try:
    import dolfinx_mpc
except ImportError:
    raise SystemExit(
        "\n[ERROR] dolfinx_mpc is not installed.\n"
        "Run this script in the jorgensd/dolfinx_mpc image:\n"
        "  docker run --rm -v $PWD:/work -w /work \\\n"
        "    ghcr.io/jorgensd/dolfinx_mpc:latest \\\n"
        "    bash -lc 'python3 poisson_solver/periodic_bc_test.py'\n"
    )

comm = MPI.COMM_WORLD

# ---------------------------------------------------------------------------
# Geometry (nm)
# ---------------------------------------------------------------------------
LX, LY, LZ = 100.0, 100.0, 100.0
NX, NY, NZ  = 20, 20, 20          # 20³ × 6 ≈ 48 k tets, builds in <1s

KX = 2 * np.pi / LX
KY = 2 * np.pi / LY
KZ = np.pi / LZ
K2 = KX**2 + KY**2 + KZ**2        # used in Test 2 exact solution

PETSC_OPTS = {
    "ksp_type": "cg",
    "pc_type":  "hypre",
    "ksp_rtol": 1e-12,
    "ksp_atol": 1e-14,
}

# ---------------------------------------------------------------------------
# Helper: build mesh + function space (reused by both tests)
# ---------------------------------------------------------------------------
def make_mesh_and_space():
    mesh = create_box(
        comm,
        [[0.0, 0.0, 0.0], [LX, LY, LZ]],
        [NX, NY, NZ],
        CellType.tetrahedron,
    )
    V = fem.functionspace(mesh, ("CG", 1))
    return mesh, V


# ---------------------------------------------------------------------------
# Helper: periodic MPC for x and y
# ---------------------------------------------------------------------------
def make_periodic_mpc(V, bcs):
    tol = 1e-10

    # Each DOF must be a slave EXACTLY ONCE — exclude corner from x and y edges
    def x_edge(x):    # x=Lx face, excluding the y=Ly edge
        return np.isclose(x[0], LX, atol=tol) & ~np.isclose(x[1], LY, atol=tol)
    def y_edge(x):    # y=Ly face, excluding the x=Lx edge
        return np.isclose(x[1], LY, atol=tol) & ~np.isclose(x[0], LX, atol=tol)
    def xy_corner(x): # x=Lx AND y=Ly corner only
        return np.isclose(x[0], LX, atol=tol) & np.isclose(x[1], LY, atol=tol)

    def map_x(x):   out = np.copy(x); out[0] -= LX; return out
    def map_y(x):   out = np.copy(x); out[1] -= LY; return out
    def map_xy(x):  out = np.copy(x); out[0] -= LX; out[1] -= LY; return out

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, x_edge,    map_x,  bcs, scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, y_edge,    map_y,  bcs, scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, xy_corner, map_xy, bcs, scale=1.0)
    mpc.finalize()
    return mpc


# ---------------------------------------------------------------------------
# Helper: L2 error
# ---------------------------------------------------------------------------
def l2_error(phi_h, phi_ex, mesh):
    diff = fem.Function(phi_h.function_space)
    diff.x.array[:] = phi_h.x.array - phi_ex.x.array
    err2 = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    return float(np.sqrt(comm.allreduce(err2, op=MPI.SUM)))


# ---------------------------------------------------------------------------
# TEST 1: Laplace  φ_exact = -z/Lz
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 1: Laplace  (φ_exact = -z/Lz)")
print("="*60)

mesh1, V1 = make_mesh_and_space()

# Dirichlet BCs on z=0 (phi=0) and z=Lz (phi=-1)
def bottom(x): return np.isclose(x[2], 0.0,  atol=1e-10)
def top(x):    return np.isclose(x[2], LZ, atol=1e-10)

tdim = mesh1.topology.dim
fdim = tdim - 1
mesh1.topology.create_connectivity(fdim, tdim)

bottom_dofs = fem.locate_dofs_geometrical(V1, bottom)
top_dofs    = fem.locate_dofs_geometrical(V1, top)
bcs1 = [
    fem.dirichletbc(fem.Constant(mesh1, PETSc.ScalarType(0.0)),  bottom_dofs, V1),
    fem.dirichletbc(fem.Constant(mesh1, PETSc.ScalarType(-1.0)), top_dofs,    V1),
]

mpc1 = make_periodic_mpc(V1, bcs1)

phi1 = ufl.TrialFunction(V1)
v1   = ufl.TestFunction(V1)
a1   = ufl.inner(ufl.grad(phi1), ufl.grad(v1)) * ufl.dx
L1   = fem.Constant(mesh1, PETSc.ScalarType(0.0)) * v1 * ufl.dx

problem1 = dolfinx_mpc.LinearProblem(a1, L1, mpc1, bcs=bcs1, petsc_options=PETSC_OPTS)
phi_h1   = problem1.solve()
phi_h1.name = "phi"

# Exact solution
phi_ex1 = fem.Function(V1, name="phi_exact")
x_dofs  = V1.tabulate_dof_coordinates()
phi_ex1.x.array[:] = -x_dofs[:, 2] / LZ

err1 = l2_error(phi_h1, phi_ex1, mesh1)
ksp1 = problem1.solver
print(f"  KSP iters : {ksp1.getIterationNumber()}")
print(f"  KSP reason: {ksp1.getConvergedReason()}")
print(f"  phi range : [{float(phi_h1.x.array.min()):.6f}, {float(phi_h1.x.array.max()):.6f}]")
print(f"  L2 error  : {err1:.3e}  (should be < 1e-10)")

# Lateral flatness check at z ≈ Lz/2
mid_mask = np.abs(x_dofs[:, 2] - LZ/2) < LZ / (2*NZ)
if mid_mask.any():
    phi_mid = phi_h1.x.array[mid_mask]
    lat_var = float(phi_mid.max() - phi_mid.min())
    print(f"  Lateral variation at z=Lz/2: {lat_var:.3e}  (should be ~0)")

with XDMFFile(comm, "periodic_test1_phi.xdmf", "w") as xf:
    xf.write_mesh(mesh1)
    xf.write_function(phi_h1)
    xf.write_function(phi_ex1)
print("  → periodic_test1_phi.xdmf")


# ---------------------------------------------------------------------------
# TEST 2: Poisson  φ_exact = sin(kx·x)·sin(ky·y)·sin(kz·z) / K²
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 2: Poisson  (sinusoidal source, φ_exact = ρ/K²)")
print(f"  kx={KX:.4f} ky={KY:.4f} kz={KZ:.4f} K²={K2:.4f}")
print("="*60)

mesh2, V2 = make_mesh_and_space()

bottom_dofs2 = fem.locate_dofs_geometrical(V2, bottom)
top_dofs2    = fem.locate_dofs_geometrical(V2, top)
bcs2 = [
    fem.dirichletbc(fem.Constant(mesh2, PETSc.ScalarType(0.0)), bottom_dofs2, V2),
    fem.dirichletbc(fem.Constant(mesh2, PETSc.ScalarType(0.0)), top_dofs2,    V2),
]

mpc2 = make_periodic_mpc(V2, bcs2)

# Source term as interpolated function
rho2 = fem.Function(V2, name="rho")
x2   = V2.tabulate_dof_coordinates()
rho2.x.array[:] = np.sin(KX * x2[:, 0]) * np.sin(KY * x2[:, 1]) * np.sin(KZ * x2[:, 2])

phi2 = ufl.TrialFunction(V2)
v2   = ufl.TestFunction(V2)
a2   = ufl.inner(ufl.grad(phi2), ufl.grad(v2)) * ufl.dx
L2   = rho2 * v2 * ufl.dx

problem2 = dolfinx_mpc.LinearProblem(a2, L2, mpc2, bcs=bcs2, petsc_options=PETSC_OPTS)
phi_h2   = problem2.solve()
phi_h2.name = "phi"

# Exact solution
phi_ex2 = fem.Function(V2, name="phi_exact")
phi_ex2.x.array[:] = rho2.x.array / K2

# Error field
phi_err2 = fem.Function(V2, name="error")
phi_err2.x.array[:] = phi_h2.x.array - phi_ex2.x.array

err2 = l2_error(phi_h2, phi_ex2, mesh2)

# L2 norm of exact solution for relative error
norm_ex2_sq = comm.allreduce(
    fem.assemble_scalar(fem.form(phi_ex2**2 * ufl.dx)), op=MPI.SUM
)
norm_ex2 = float(np.sqrt(norm_ex2_sq))
rel_err2  = err2 / norm_ex2 if norm_ex2 > 0 else float("inf")

ksp2 = problem2.solver
print(f"  KSP iters : {ksp2.getIterationNumber()}")
print(f"  KSP reason: {ksp2.getConvergedReason()}")
print(f"  phi range     : [{float(phi_h2.x.array.min()):.4f}, {float(phi_h2.x.array.max()):.4f}]")
print(f"  phi_exact range: [{float(phi_ex2.x.array.min()):.4f}, {float(phi_ex2.x.array.max()):.4f}]")
print(f"  L2 error (abs) : {err2:.3e}")
print(f"  L2 norm exact  : {norm_ex2:.3e}")
print(f"  Relative L2 err: {rel_err2*100:.2f}%  (expected ~3% for P1 h≈7nm)")
print(f"  max |error|    : {float(np.abs(phi_err2.x.array).max()):.3e}")

with XDMFFile(comm, "periodic_test2_phi.xdmf", "w") as xf:
    xf.write_mesh(mesh2)
    xf.write_function(phi_h2)
    xf.write_function(phi_ex2)
    xf.write_function(phi_err2)
print("  → periodic_test2_phi.xdmf")

print("\n" + "="*60)
print("PASS criteria:")
print(f"  Test 1 abs L2 error < 1e-10   : {'PASS' if err1 < 1e-10  else 'FAIL'} ({err1:.2e})")
print(f"  Test 2 relative L2 error < 5% : {'PASS' if rel_err2 < 0.05 else 'FAIL'} ({rel_err2*100:.2f}%)")
print("="*60)
