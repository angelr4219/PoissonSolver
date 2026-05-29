"""
bc_fft_comparison.py

Three-way comparison for 3D Poisson on a box:

  Mode A  FEM + Dirichlet lateral BCs  (φ=0 forced on x,y walls)
  Mode B  FEM + Periodic  lateral BCs  (φ matches on opposite faces)
  Mode C  FFT solver      (spectral, O(N log N), periodic x,y + Dirichlet z)

Part 1 – Visual comparison (Gaussian source, N=20 mesh, Modes A & B)
  Shows that Dirichlet "traps" the field while Periodic lets it interact
  with image charges in neighbouring unit cells.

Part 2 – Accuracy + timing benchmark (sinusoidal source, known exact solution)
  FFT sweep: N = 20, 40, 80, 160, 320  (all fast)
  FEM sweep: N = 20, 40, 80            (stops before OOM)
  Demonstrates O(N log N) FFT vs O(N^1.5) FEM scaling.

Box: 100 × 100 × 100 nm
Requires dolfinx_mpc.  Run with:
  docker run --rm -v "$PWD":/work -w /work \\
    ghcr.io/jorgensd/dolfinx_mpc:latest \\
    bash -lc 'python3 poisson_solver/bc_fft_comparison.py'

Outputs (working directory):
  part1_dirichlet.xdmf / .h5
  part1_periodic.xdmf  / .h5
  part2_fft_N{n}.xdmf  / .h5   (N=20, 40, 80)
"""

from __future__ import annotations
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem
from dolfinx.mesh import create_box, CellType
from dolfinx.io import XDMFFile
from scipy.fft import dst, idst
import dolfinx_mpc

comm = MPI.COMM_WORLD

LX, LY, LZ = 100.0, 100.0, 100.0   # nm

PETSC_OPTS = {
    "ksp_type":  "cg",
    "pc_type":   "hypre",
    "ksp_rtol":  1e-12,
    "ksp_atol":  1e-14,
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_mesh(N):
    return create_box(comm, [[0.0]*3, [LX, LY, LZ]], [N, N, N], CellType.tetrahedron)

def make_space(mesh):
    return fem.functionspace(mesh, ("CG", 1))

def dirichlet_bcs(V, v_bottom=0.0, v_top=0.0):
    """Dirichlet at z=0 and z=Lz only (lateral walls free or constrained elsewhere)."""
    tol = 1e-10
    def bot(x): return np.isclose(x[2], 0.0, atol=tol)
    def top(x): return np.isclose(x[2], LZ,  atol=tol)
    return [
        fem.dirichletbc(PETSc.ScalarType(v_bottom), fem.locate_dofs_geometrical(V, bot), V),
        fem.dirichletbc(PETSc.ScalarType(v_top),    fem.locate_dofs_geometrical(V, top), V),
    ]

def lateral_dirichlet_bcs(V):
    """Also pin lateral walls to 0 (Mode A)."""
    tol = 1e-10
    def x0(x):  return np.isclose(x[0], 0.0, atol=tol)
    def xL(x):  return np.isclose(x[0], LX,  atol=tol)
    def y0(x):  return np.isclose(x[1], 0.0, atol=tol)
    def yL(x):  return np.isclose(x[1], LY,  atol=tol)
    zero = PETSc.ScalarType(0.0)
    return [fem.dirichletbc(zero, fem.locate_dofs_geometrical(V, f), V)
            for f in (x0, xL, y0, yL)]

def make_periodic_mpc(V, bcs):
    tol = 1e-10
    def x_edge(x):    return  np.isclose(x[0], LX, atol=tol) & ~np.isclose(x[1], LY, atol=tol)
    def y_edge(x):    return  np.isclose(x[1], LY, atol=tol) & ~np.isclose(x[0], LX, atol=tol)
    def xy_corner(x): return  np.isclose(x[0], LX, atol=tol) &  np.isclose(x[1], LY, atol=tol)
    def map_x(x):   o = np.copy(x); o[0] -= LX; return o
    def map_y(x):   o = np.copy(x); o[1] -= LY; return o
    def map_xy(x):  o = np.copy(x); o[0] -= LX; o[1] -= LY; return o
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, x_edge,    map_x,  bcs)
    mpc.create_periodic_constraint_geometrical(V, y_edge,    map_y,  bcs)
    mpc.create_periodic_constraint_geometrical(V, xy_corner, map_xy, bcs)
    mpc.finalize()
    return mpc

def fem_solve(V, rho_fn, bcs, mpc=None, label=""):
    """Run a FEM Poisson solve. Returns (phi_fn, elapsed_s)."""
    rho = fem.Function(V, name="rho")
    coords = V.tabulate_dof_coordinates()
    rho.x.array[:] = rho_fn(coords[:, 0], coords[:, 1], coords[:, 2])

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    t0 = time.perf_counter()
    if mpc is None:
        prob = fem.petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options=PETSC_OPTS, petsc_options_prefix="cmp_")
        phi  = prob.solve()
    else:
        prob = dolfinx_mpc.LinearProblem(
            a, L, mpc, bcs=bcs, petsc_options=PETSC_OPTS, petsc_options_prefix="cmp_")
        phi  = prob.solve()
    elapsed = time.perf_counter() - t0

    phi.name = "phi"
    ncells = V.mesh.topology.index_map(3).size_local
    its    = prob.solver.getIterationNumber()
    print(f"  {label:<30} cells={ncells:>8,}  KSP={its:>4}  t={elapsed:.2f}s")
    return phi, elapsed

def l2_rel_error(phi_h, phi_ex):
    diff = phi_h.x.array - phi_ex.x.array
    return float(np.linalg.norm(diff) / (np.linalg.norm(phi_ex.x.array) + 1e-30))


# ─────────────────────────────────────────────────────────────────────────────
# FFT Poisson solver  (periodic x,y + Dirichlet z)
# ─────────────────────────────────────────────────────────────────────────────

def fft_poisson(rho_grid):
    """
    Solve -∇²φ = ρ spectrally.

    rho_grid : array (Nx, Ny, Nz_int) evaluated at
               x_i = i*dx, y_j = j*dy, z_k = (k+1)*dz  (interior z only)
    Returns phi_grid of same shape.
    """
    Nx, Ny, Nz = rho_grid.shape
    dx = LX / Nx
    dy = LY / Ny
    dz = LZ / (Nz + 1)

    # 2D FFT in x,y
    rho_hat = np.fft.fft2(rho_grid, axes=(0, 1))

    # DST-1 in z  (encodes Dirichlet at z=0 and z=Lz exactly)
    r_dst       = np.zeros_like(rho_hat)
    r_dst.real  = dst(rho_hat.real, type=1, axis=2)
    r_dst.imag  = dst(rho_hat.imag, type=1, axis=2)

    # Eigenvalues
    kx = (2 * np.pi * np.fft.fftfreq(Nx, d=dx))[:, None, None]
    ky = (2 * np.pi * np.fft.fftfreq(Ny, d=dy))[None, :, None]
    kz = (np.pi * np.arange(1, Nz + 1) / LZ)[None, None, :]
    K2 = kx**2 + ky**2 + kz**2

    phi_dst      = r_dst / K2
    phi_hat      = np.zeros_like(phi_dst)
    phi_hat.real = idst(phi_dst.real, type=1, axis=2)
    phi_hat.imag = idst(phi_dst.imag, type=1, axis=2)

    return np.fft.ifft2(phi_hat, axes=(0, 1)).real


def fft_to_xdmf(phi_grid, N, label):
    """Interpolate FFT result onto a dolfinx mesh and write XDMF."""
    mesh = make_mesh(N)
    V    = make_space(mesh)
    phi  = fem.Function(V, name="phi")
    x    = V.tabulate_dof_coordinates()

    # Grid spacing (interior z points: (k+1)*dz)
    dx = LX / N; dy = LY / N; dz = LZ / (N + 1)

    # Nearest-grid interpolation (sufficient for visual check)
    ix = np.clip(np.round(x[:, 0] / dx).astype(int), 0, N - 1)
    iy = np.clip(np.round(x[:, 1] / dy).astype(int), 0, N - 1)
    iz = np.clip(np.round(x[:, 2] / dz - 1).astype(int), 0, N - 1)
    phi.x.array[:] = phi_grid[ix, iy, iz]
    phi.x.array[np.isclose(x[:, 2], 0.0) | np.isclose(x[:, 2], LZ)] = 0.0

    with XDMFFile(comm, f"{label}.xdmf", "w") as xf:
        xf.write_mesh(mesh)
        xf.write_function(phi)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1  Visual comparison  (Gaussian source, N=20)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PART 1 — Dirichlet vs Periodic  (Gaussian source, N=20)")
print("="*60)

SIG = 15.0   # nm, Gaussian width

def rho_gauss(x, y, z):
    r2 = (x - LX/2)**2 + (y - LY/2)**2 + (z - LZ/2)**2
    return np.exp(-r2 / SIG**2)

N1 = 20
mesh1 = make_mesh(N1)
V1    = make_space(mesh1)

# Mode A – Dirichlet everywhere
bcs_a  = dirichlet_bcs(V1) + lateral_dirichlet_bcs(V1)
phi_a, t_a = fem_solve(V1, rho_gauss, bcs_a, label="Dirichlet (all walls)")

# Mode B – Periodic x,y + Dirichlet z
bcs_b  = dirichlet_bcs(V1)
mpc_b  = make_periodic_mpc(V1, bcs_b)
phi_b, t_b = fem_solve(V1, rho_gauss, bcs_b, mpc_b, label="Periodic  (x,y) + Dirichlet z")

# Difference field
phi_diff = fem.Function(V1, name="diff_P_minus_D")
phi_diff.x.array[:] = phi_b.x.array - phi_a.x.array

with XDMFFile(comm, "part1_dirichlet.xdmf", "w") as xf:
    xf.write_mesh(mesh1); xf.write_function(phi_a)
with XDMFFile(comm, "part1_periodic.xdmf", "w") as xf:
    xf.write_mesh(mesh1); xf.write_function(phi_b)
with XDMFFile(comm, "part1_diff.xdmf", "w") as xf:
    xf.write_mesh(mesh1); xf.write_function(phi_diff)

max_a = float(np.abs(phi_a.x.array).max())
max_b = float(np.abs(phi_b.x.array).max())
max_d = float(np.abs(phi_diff.x.array).max())
print(f"\n  |phi|_max  Dirichlet={max_a:.4f}   Periodic={max_b:.4f}   diff={max_d:.4f}")
print(f"  Time       Dirichlet={t_a:.2f}s       Periodic={t_b:.2f}s")
print(f"  Outputs: part1_dirichlet.xdmf  part1_periodic.xdmf  part1_diff.xdmf")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2  Accuracy + timing benchmark  (sinusoidal source, exact solution)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PART 2 — FFT vs FEM timing + accuracy  (sinusoidal source)")
print("="*60)

KX_s = 2 * np.pi / LX
KY_s = 2 * np.pi / LY
KZ_s = np.pi       / LZ
K2_s = KX_s**2 + KY_s**2 + KZ_s**2

def rho_sin(x, y, z):
    return np.sin(KX_s*x) * np.sin(KY_s*y) * np.sin(KZ_s*z)

def phi_exact_fn(x, y, z):
    return rho_sin(x, y, z) / K2_s

# ---- FFT sweep ----
print("\n  [FFT sweep]")
FFT_SIZES = [20, 40, 80, 160, 320]
fft_results = []

for N in FFT_SIZES:
    dx = LX / N; dy = LY / N; dz = LZ / (N + 1)
    xi = np.arange(N) * dx
    yi = np.arange(N) * dy
    zi = (np.arange(N) + 1) * dz
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')
    rho_grid  = rho_sin(X, Y, Z)
    phi_exact = phi_exact_fn(X, Y, Z)

    t0 = time.perf_counter()
    phi_grid = fft_poisson(rho_grid)
    t_fft = time.perf_counter() - t0

    rel_err = float(np.linalg.norm(phi_grid - phi_exact) /
                    (np.linalg.norm(phi_exact) + 1e-30))
    dof = N**3
    print(f"    N={N:<4}  dof={dof:>9,}  t={t_fft:.4f}s  rel_err={rel_err:.2e}")
    fft_results.append((N, dof, t_fft, rel_err))

    # Write XDMF for N <= 80
    if N <= 80:
        fft_to_xdmf(phi_grid, N, f"part2_fft_N{N}")

# ---- FEM sweep ----
print("\n  [FEM sweep — periodic x,y + Dirichlet z]")
FEM_SIZES   = [20, 40, 80]
MAX_CELLS   = 4_000_000
fem_results = []

for N in FEM_SIZES:
    est_cells = 6 * N**3
    if est_cells > MAX_CELLS:
        print(f"    N={N:<4}  SKIPPED (est {est_cells/1e6:.1f}M cells)")
        continue

    mesh_n = make_mesh(N)
    V_n    = make_space(mesh_n)
    bcs_n  = dirichlet_bcs(V_n)
    mpc_n  = make_periodic_mpc(V_n, bcs_n)

    phi_h, t_fem = fem_solve(V_n, rho_sin, bcs_n, mpc_n,
                             label=f"FEM periodic N={N}")

    phi_ex = fem.Function(V_n, name="phi_exact")
    coords = V_n.tabulate_dof_coordinates()
    phi_ex.x.array[:] = phi_exact_fn(coords[:, 0], coords[:, 1], coords[:, 2])

    diff   = phi_h.x.array - phi_ex.x.array
    rel_e  = float(np.linalg.norm(diff) / (np.linalg.norm(phi_ex.x.array) + 1e-30))
    ncells = mesh_n.topology.index_map(3).size_local
    fem_results.append((N, ncells, t_fem, rel_e))

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY  (sinusoidal source, 100³ nm box)")
print("="*60)
print(f"{'Method':<22} {'N':>4}  {'DOF/cells':>10}  {'t_solve':>9}  {'rel_err':>9}")
print("-"*60)

for N, dof, t, e in fft_results:
    tag = "(XDMF)" if N <= 80 else ""
    print(f"  FFT                    {N:>4}  {dof:>10,}  {t:>8.4f}s  {e:>8.2e}  {tag}")

print()
for N, nc, t, e in fem_results:
    print(f"  FEM periodic           {N:>4}  {nc:>10,}  {t:>8.2f}s  {e:>8.2e}")

if fft_results and fem_results:
    # Compare at N=20 and N=40 if available
    fft_map = {r[0]: r for r in fft_results}
    fem_map = {r[0]: r for r in fem_results}
    print(f"\nSpeedup (FEM / FFT) at matching N:")
    for N in (20, 40, 80):
        if N in fft_map and N in fem_map:
            speedup = fem_map[N][2] / fft_map[N][2]
            print(f"  N={N}: {speedup:.0f}x  "
                  f"(FEM {fem_map[N][2]:.2f}s vs FFT {fft_map[N][2]:.4f}s)  "
                  f"accuracy: FEM {fem_map[N][3]:.2e}  FFT {fft_map[N][3]:.2e}")

print(f"\nKey insight:")
print(f"  FFT at N=320 ({320**3/1e6:.1f}M points): near-zero time, spectral accuracy")
print(f"  FEM at N=80  ({6*80**3/1e6:.1f}M cells ): takes tens of seconds")
print(f"  At N=160+, FFT is the only practical option.")
print(f"\nWhen FFT applies: box geometry, smooth ε, periodic x,y, Dirichlet z.")
print(f"When FFT doesn't: dielectric steps (sSi/SiGe), gates, surface charge.")
print(f"  → Use FFT for background potential, FEM for gate perturbation.")
