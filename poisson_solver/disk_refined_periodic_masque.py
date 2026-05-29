"""
disk_refined_periodic_masque.py

MaSQE device geometry with local mesh refinement around the disk gate.
Solves both Dirichlet (lateral walls grounded) and Periodic (x,y) variants
on the same unstructured Gmsh mesh.

Device geometry (SI units throughout — metres):
  Box    : 500 × 500 × 255 nm  (x: -250→250, y: -250→250, z: 0→255 nm)
  Gate   : circular disk  R = 10 nm, centred at (0,0), top face z = 255 nm, V = −1 V
  Bottom : z = 0,  V = −12 V   (substrate back-gate)
  Layers : SiGe  0– 50 nm  ε_r = 13.05
           sSi  50– 55 nm  ε_r = 11.70   (quantum well)
           SiGe 55–255 nm  ε_r = 13.05

Mesh refinement (Gmsh Box field):
  h_fine   = 2 nm  within ±30 nm of disk centre, full z
  h_coarse = 20 nm  everywhere else

Outputs (relative to CWD = project root):
  results/disk_refined_dirichlet/disk_refined_dirichlet.xdmf / .h5
  results/disk_refined_periodic/disk_refined_periodic.xdmf   / .h5

Run in dolfinx_mpc container (needed for periodic solve):
  docker run --rm \\
    -v "/Users/angelramirez/Desktop/poisson_solver":/work \\
    -w /work \\
    ghcr.io/jorgensd/dolfinx_mpc:latest \\
    bash -lc 'python3 poisson_solver/disk_refined_periodic_masque.py'

Compare to MaSQE afterwards:
  ./MASQUE-Comparison/run_compare.sh \\
    --cases disk_refined_dirichlet disk_refined_periodic

Tip: set H_FINE = 5e-9 for a quick ~50 k-cell test run.
"""

import time
import resource
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile

try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio

try:
    import gmsh as gmsh_api
except ImportError:
    raise SystemExit("[ERROR] gmsh not found. Run inside the dolfinx_mpc Docker container.")

try:
    import dolfinx_mpc
    HAS_MPC = True
except ImportError:
    HAS_MPC = False
    print("[WARN] dolfinx_mpc not installed — periodic solve will be skipped")

from dolfinx.fem.petsc import LinearProblem as DolfinxLinearProblem

comm = MPI.COMM_WORLD

# ── Device parameters ─────────────────────────────────────────────────────────
LX, LY, LZ = 500e-9, 500e-9, 255e-9
X0, Y0     = -250e-9, -250e-9       # box origin (centred in x,y)
Z_BOT      = 0.0
Z_TOP      = LZ                      # 255e-9 m

DISK_R    = 10e-9
DISK_V    = -1.0
BOTTOM_V  = -12.0

EPS0   = 8.8541878128e-12            # F/m
LAYERS = [                           # (z_min, z_max, eps_r)  — metres
    (0.0,    50e-9, 13.05),
    (50e-9,  55e-9, 11.7),
    (55e-9, 255e-9, 13.05),
]

# Mesh sizes  (set H_FINE = 5e-9 for a ~50k-cell quick test)
H_FINE   = 2e-9    # m  inside refinement band
H_COARSE = 20e-9   # m  outside refinement band
BAND_XY  = 30e-9   # m  half-width of fine box around disk centre (all z)

# PETSc solver options (shared between solves)
_KRYLOV_OPTS = {
    "ksp_type":   "cg",
    "pc_type":    "hypre",
    "ksp_rtol":   1e-10,
    "ksp_atol":   1e-14,
    "ksp_max_it": 50000,
}

_TOL = 1e-10   # m — geometric tolerance for DOF location (>> float64 rounding)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _peak_rss_mb():
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    import sys
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 if sys.platform != "darwin" else (1024.0 * 1024.0))


def _banner(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ── Mesh build ────────────────────────────────────────────────────────────────

def build_mesh():
    _banner("MESH: Gmsh box with disk refinement")
    t0_total = time.perf_counter()

    gmsh_api.initialize()
    gmsh_api.option.setNumber("General.Terminal", 0)
    gmsh_api.model.add("disk_refined_device")

    # Single OCC box
    box_tag = gmsh_api.model.occ.addBox(X0, Y0, Z_BOT, LX, LY, LZ)
    gmsh_api.model.occ.synchronize()

    all_vols  = [v for (_, v) in gmsh_api.model.getEntities(3)]
    all_surfs = [s for (_, s) in gmsh_api.model.getEntities(2)]
    gmsh_api.model.addPhysicalGroup(3, all_vols,  tag=1, name="domain")
    gmsh_api.model.addPhysicalGroup(2, all_surfs, tag=1, name="boundary")

    # Box refinement field: fine column over the disk, coarse elsewhere
    fid = gmsh_api.model.mesh.field.add("Box")
    gmsh_api.model.mesh.field.setNumber(fid, "VIn",      H_FINE)
    gmsh_api.model.mesh.field.setNumber(fid, "VOut",     H_COARSE)
    gmsh_api.model.mesh.field.setNumber(fid, "XMin",    -BAND_XY)
    gmsh_api.model.mesh.field.setNumber(fid, "XMax",     BAND_XY)
    gmsh_api.model.mesh.field.setNumber(fid, "YMin",    -BAND_XY)
    gmsh_api.model.mesh.field.setNumber(fid, "YMax",     BAND_XY)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMin",     Z_BOT)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMax",     Z_TOP)
    gmsh_api.model.mesh.field.setNumber(fid, "Thickness", 0.0)
    gmsh_api.model.mesh.field.setAsBackgroundMesh(fid)

    gmsh_api.option.setNumber("Mesh.CharacteristicLengthMin", H_FINE * 0.5)
    gmsh_api.option.setNumber("Mesh.CharacteristicLengthMax", H_COARSE)
    gmsh_api.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay

    t0 = time.perf_counter()
    gmsh_api.model.mesh.generate(3)
    t_gmsh = time.perf_counter() - t0
    print(f"  Gmsh meshing:  {t_gmsh:.1f}s")

    t0 = time.perf_counter()
    mesh_data = gmshio.model_to_mesh(gmsh_api.model, comm, rank=0, gdim=3)
    msh = mesh_data.mesh if hasattr(mesh_data, "mesh") else mesh_data[0]
    gmsh_api.finalize()
    t_conv = time.perf_counter() - t0
    print(f"  DOLFINx conv:  {t_conv:.1f}s")

    ncells = msh.topology.index_map(msh.topology.dim).size_local
    nnodes = msh.geometry.x.shape[0]
    rss    = _peak_rss_mb()

    print(f"\n  Cells : {ncells:,}")
    print(f"  Nodes : {nnodes:,}")
    print(f"  RSS   : {rss:.0f} MB")
    print(f"  Total : {time.perf_counter() - t0_total:.1f}s")

    return msh, ncells, nnodes


# ── DG0 permittivity ──────────────────────────────────────────────────────────

def build_epsilon(msh):
    V0  = fem.functionspace(msh, ("DG", 0))
    eps = fem.Function(V0, name="epsilon")
    eps.x.array[:] = EPS0 * 13.05          # default: SiGe
    xc = V0.tabulate_dof_coordinates()
    zc = xc[:, 2]
    for (zmin, zmax, epsr) in LAYERS:
        mask = (zc >= zmin) & (zc < zmax)
        eps.x.array[mask] = EPS0 * epsr
    return eps


# ── Boundary indicator functions ──────────────────────────────────────────────

def _on_bottom(x):
    return np.isclose(x[2], Z_BOT, atol=_TOL)

def _on_disk(x):
    r2 = x[0]**2 + x[1]**2
    return np.isclose(x[2], Z_TOP, atol=_TOL) & (r2 <= DISK_R**2 + 1e-20)

def _on_sides(x):
    return (np.isclose(x[0], X0,      atol=_TOL) |
            np.isclose(x[0], X0 + LX, atol=_TOL) |
            np.isclose(x[1], Y0,      atol=_TOL) |
            np.isclose(x[1], Y0 + LY, atol=_TOL))


# ── Periodic MPC (x and y) ────────────────────────────────────────────────────

def _make_periodic_mpc(V, bcs):
    """
    Periodic constraints in x and y.
    Corners treated separately to avoid slave-of-slave conflicts.
    """
    per_tol = 1e-12    # m — much tighter than _TOL but >> float64 ULP for 250e-9
    x_max   = X0 + LX  # 250e-9
    y_max   = Y0 + LY  # 250e-9

    def x_only(x):
        return (np.isclose(x[0], x_max, atol=per_tol) &
                ~np.isclose(x[1], y_max, atol=per_tol))

    def y_only(x):
        return (np.isclose(x[1], y_max, atol=per_tol) &
                ~np.isclose(x[0], x_max, atol=per_tol))

    def xy_corner(x):
        return (np.isclose(x[0], x_max, atol=per_tol) &
                np.isclose(x[1], y_max, atol=per_tol))

    def map_x(x):  o = np.copy(x); o[0] -= LX; return o
    def map_y(x):  o = np.copy(x); o[1] -= LY; return o
    def map_xy(x): o = np.copy(x); o[0] -= LX; o[1] -= LY; return o

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, x_only,   map_x,  bcs, scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, y_only,   map_y,  bcs, scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, xy_corner, map_xy, bcs, scale=1.0)
    mpc.finalize()
    return mpc


# ── Solvers ───────────────────────────────────────────────────────────────────

def solve_dirichlet(msh, eps):
    """FEM solve with grounded lateral walls (φ = 0 on x/y sides)."""
    _banner("SOLVE: Dirichlet lateral BCs")
    V   = fem.functionspace(msh, ("CG", 1))
    phi = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a   = ufl.inner(eps * ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L   = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx

    dofs_bot   = fem.locate_dofs_geometrical(V, _on_bottom)
    dofs_disk  = fem.locate_dofs_geometrical(V, _on_disk)
    dofs_sides = fem.locate_dofs_geometrical(V, _on_sides)

    print(f"  bottom dofs : {len(dofs_bot)}")
    print(f"  disk dofs   : {len(dofs_disk)}")
    print(f"  side dofs   : {len(dofs_sides)}")

    bcs = [
        fem.dirichletbc(PETSc.ScalarType(BOTTOM_V), dofs_bot,   V),
        fem.dirichletbc(PETSc.ScalarType(DISK_V),   dofs_disk,  V),
        fem.dirichletbc(PETSc.ScalarType(0.0),      dofs_sides, V),
    ]

    t0 = time.perf_counter()
    try:
        problem = DolfinxLinearProblem(a, L, bcs=bcs,
                                       petsc_options=_KRYLOV_OPTS,
                                       petsc_options_prefix="dir_")
    except TypeError:
        problem = DolfinxLinearProblem(a, L, bcs=bcs, petsc_options=_KRYLOV_OPTS)

    phi_h = problem.solve()
    phi_h.name = "phi"
    dt = time.perf_counter() - t0

    ksp = problem.solver
    print(f"  KSP iters : {ksp.getIterationNumber()}")
    print(f"  KSP reason: {ksp.getConvergedReason()}")
    print(f"  phi range : [{float(phi_h.x.array.min()):.4f}, {float(phi_h.x.array.max()):.4f}] V")
    print(f"  solve time: {dt:.1f}s")
    return phi_h


def solve_periodic(msh, eps):
    """FEM solve with periodic x,y BCs (dolfinx_mpc required)."""
    _banner("SOLVE: Periodic x,y BCs")

    if not HAS_MPC:
        print("  [SKIP] dolfinx_mpc not available in this environment")
        return None

    V   = fem.functionspace(msh, ("CG", 1))
    phi = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a   = ufl.inner(eps * ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L   = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx

    dofs_bot  = fem.locate_dofs_geometrical(V, _on_bottom)
    dofs_disk = fem.locate_dofs_geometrical(V, _on_disk)

    print(f"  bottom dofs : {len(dofs_bot)}")
    print(f"  disk dofs   : {len(dofs_disk)}")

    bcs = [
        fem.dirichletbc(PETSc.ScalarType(BOTTOM_V), dofs_bot,  V),
        fem.dirichletbc(PETSc.ScalarType(DISK_V),   dofs_disk, V),
    ]

    print("  building periodic MPC...")
    t0 = time.perf_counter()
    mpc = _make_periodic_mpc(V, bcs)
    print(f"  MPC build: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    try:
        problem = dolfinx_mpc.LinearProblem(a, L, mpc, bcs=bcs,
                                             petsc_options=_KRYLOV_OPTS,
                                             petsc_options_prefix="per_")
    except TypeError:
        problem = dolfinx_mpc.LinearProblem(a, L, mpc, bcs=bcs,
                                             petsc_options=_KRYLOV_OPTS)

    phi_h = problem.solve()
    phi_h.name = "phi"
    dt = time.perf_counter() - t0

    ksp = problem.solver
    print(f"  KSP iters : {ksp.getIterationNumber()}")
    print(f"  KSP reason: {ksp.getConvergedReason()}")
    print(f"  phi range : [{float(phi_h.x.array.min()):.4f}, {float(phi_h.x.array.max()):.4f}] V")
    print(f"  solve time: {dt:.1f}s")
    return phi_h


# ── XDMF export ───────────────────────────────────────────────────────────────

def write_output(msh, phi_h, eps, case_name):
    outdir = Path("results") / case_name
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{case_name}.xdmf"
    with XDMFFile(comm, str(path), "w") as xf:
        xf.write_mesh(msh)
        xf.write_function(phi_h)
        xf.write_function(eps)
    print(f"  → {path}")


# ── Depth probes ──────────────────────────────────────────────────────────────

def probe_z_slices(phi_h, label):
    """Print mean φ at the same z-slices used in MASQUE comparison."""
    from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

    V   = phi_h.function_space
    msh = V.mesh
    slices_nm = [0.0, 10.0, 20.0, 51.0, 155.0, 255.0]

    print(f"\n  [{label}] φ at gate centre (0,0,z):")
    tree = bb_tree(msh, msh.topology.dim)
    for z_nm in slices_nm:
        pt = np.array([[0.0, 0.0, z_nm * 1e-9]])
        cand = compute_collisions_points(tree, pt)
        from dolfinx.geometry import compute_colliding_cells
        coll = compute_colliding_cells(msh, cand, pt)
        if coll.links(0).size > 0:
            val = float(phi_h.eval(pt, coll.links(0)[:1])[0])
            print(f"    z = {z_nm:5.1f} nm  →  φ = {val:+.5f} V")
        else:
            print(f"    z = {z_nm:5.1f} nm  →  (outside mesh)")


# ── Main ──────────────────────────────────────────────────────────────────────

_banner("disk_refined_periodic_masque.py")
print(f"  Box      : {LX*1e9:.0f} × {LY*1e9:.0f} × {LZ*1e9:.0f} nm")
print(f"  Gate     : R = {DISK_R*1e9:.0f} nm, V = {DISK_V} V  (top, z = {LZ*1e9:.0f} nm)")
print(f"  Bottom   : V = {BOTTOM_V} V  (z = 0)")
print(f"  Layers   : SiGe / sSi / SiGe  (ε_r = 13.05 / 11.70 / 13.05)")
print(f"  h_fine   : {H_FINE*1e9:.0f} nm  within ±{BAND_XY*1e9:.0f} nm of disk (all z)")
print(f"  h_coarse : {H_COARSE*1e9:.0f} nm  elsewhere")

t_wall = time.perf_counter()

msh, ncells, nnodes = build_mesh()
eps = build_epsilon(msh)

phi_dir = solve_dirichlet(msh, eps)
phi_per = solve_periodic(msh, eps)

_banner("RESULTS")
write_output(msh, phi_dir, eps, "disk_refined_dirichlet")
if phi_per is not None:
    write_output(msh, phi_per, eps, "disk_refined_periodic")

probe_z_slices(phi_dir, "Dirichlet")
if phi_per is not None:
    probe_z_slices(phi_per, "Periodic")

print(f"\n  Cells : {ncells:,}   Nodes : {nnodes:,}")
print(f"  Wall  : {time.perf_counter() - t_wall:.1f}s")
print(f"  RSS   : {_peak_rss_mb():.0f} MB")

_banner("NEXT STEP: compare to MaSQE")
cases = "disk_refined_dirichlet" + (" disk_refined_periodic" if phi_per else "")
print("  ./MASQUE-Comparison/run_compare.sh \\")
print(f"    --cases {cases} \\")
print("    --qw-exclude-nm 50 55")
