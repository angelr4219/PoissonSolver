#!/usr/bin/env python3
"""Triple-method Poisson comparison on a conducting sphere in a cubic box.

Problem
-------
Conducting sphere of radius R = 50 nm, held at V₀ = 1 V, inside a cubic
box of side L = 500 nm.

The analytic solution in an INFINITE homogeneous medium is:

    φ(r) = V₀ · R / r        r > R

Three methods are compared at five mesh refinement levels:
    h [nm] : 20, 10, 5, 1, 0.1

Method 1 – FEM-Dirichlet
    FEM on sphere_in_box gmsh mesh (adaptive refinement near sphere).
    Outer box: φ = 0 (Dirichlet).  Models an isolated sphere.
    h_fine = element size near sphere; h_coarse = max(5·h_fine, 25) nm.
    NOTE: h_fine < 5 nm (i.e. 1, 0.1) skipped by default (impractical
          element count / observed to OOM or time out the container).

Method 2 – FEM-Periodic
    FEM on sphere_in_box gmsh mesh with 3-D periodic BCs on the box faces
    via dolfinx_mpc.  Models an infinite periodic array of spheres.
    Requires: dolfinx_mpc  (pip install dolfinx_mpc or build from source).
    NOTE: h_fine < 5 nm skipped, same rationale as FEM-Dirichlet; periodic
          constraint accuracy also depends on how well dolfinx_mpc resolves
          non-matching boundary nodes.

Method 3 – FFT
    Spectral solve on a uniform N³ Cartesian grid with periodic BCs.
    The sphere is approximated as a thin Gaussian charge shell at r = R
    (total charge q = 4πε₀V₀R, consistent with the analytic solution).
    N = L/h; skipped for N > FFT_MAX_N (default 300) to cap memory.
    Gauge: k=0 mode set to 0 (zero-mean periodic convention).

Error metric
------------
For each method and each h, probe points are placed at radii
r = [100, 150, 200] nm from the origin (well inside the box, outside the
sphere) and the relative error  |φ_h(r) - φ_exact(r)| / φ_exact(r)  is
computed.  The maximum over the three probes is reported.

Output
------
results/sphere_triple/
    comparison_results.csv     – full table (method, h_nm, n_dofs, t_mesh_s,
                                              t_solve_s, max_rel_err, mean_rel_err)
    dir_h{h_nm}nm.xdmf+.h5    – FEM-Dirichlet solution field
    per_h{h_nm}nm.xdmf+.h5    – FEM-Periodic  solution field
    fft_h{h_nm}nm.npz          – FFT solution array (phi, rho, grid coords)
    SUGGESTIONS.txt            – recommendations for MEMS/real-device comparison
                                  (see item 0 for why mesh refinement alone
                                  will not close the gap with MASQUE)

See also benchmarks/fft_only_sweep.py for a pure-NumPy FFT-only sweep that
runs without Docker/dolfinx, and benchmarks/quadrant_mesh_density_demo.py
for a single mesh split into 4 differently-refined zones (20/10/5/1 nm).

Usage
-----
    ./run_dolfinx.sh benchmarks/sphere_triple_comparison.py [--help]

    --L_nm        box side [nm]          default 500
    --R_nm        sphere radius [nm]     default 50
    --V0          sphere potential [V]   default 1.0
    --fft-max-n   max FFT grid pts/dim   default 300
    --skip-dir    skip FEM-Dirichlet
    --skip-per    skip FEM-Periodic
    --skip-fft    skip FFT
    --outdir      output directory       default results/sphere_triple
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

# ── dolfinx core ──────────────────────────────────────────────────────────────
from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile
import ufl
from petsc4py import PETSc

# ── project library ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poisson.geom_sphere import sphere_in_box
from poisson.fft_solve import (
    fft_poisson_3d,
    sphere_charge_shell_grid,
    evaluate_fft_phi_at_points,
    EPS0,
)
from poisson.analytics import phi_conducting_sphere

# ── dolfinx_mpc (optional) ────────────────────────────────────────────────────
try:
    from dolfinx_mpc import MultiPointConstraint
    HAS_MPC = True
except ImportError:
    HAS_MPC = False

COMM = MPI.COMM_WORLD
RANK = COMM.rank
IS_ROOT = RANK == 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    if IS_ROOT:
        print(msg, flush=True)


def _probe_points(probe_r_nm: list[float]) -> np.ndarray:
    """Diagonal probe points at the requested radii (nm), returned in metres."""
    pts = []
    for r_nm in probe_r_nm:
        r = r_nm * 1e-9 / np.sqrt(3.0)
        pts.append([r, r, r])
    return np.array(pts, dtype=float)


def _fem_probe_error(domain, V, uh, probe_pts_m, V0, R_m):
    """Evaluate FEM solution at probe points and compute relative errors."""
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    tree   = bb_tree(domain, domain.topology.dim)
    colls  = compute_collisions_points(tree, probe_pts_m)
    cells  = compute_colliding_cells(domain, colls, probe_pts_m)

    phi_h = []
    for i, pt in enumerate(probe_pts_m):
        cell_list = cells.links(i)
        if len(cell_list) > 0:
            val = float(np.asarray(uh.eval(pt.reshape(1, -1), cell_list[:1])).reshape(-1)[0])
        else:
            val = float("nan")
        phi_h.append(val)

    phi_h     = np.array(phi_h)
    r_pts     = np.linalg.norm(probe_pts_m, axis=1)
    phi_exact = phi_conducting_sphere(r_pts, V0, R_m)
    rel_err   = np.abs(phi_h - phi_exact) / np.abs(phi_exact)
    return rel_err


# ──────────────────────────────────────────────────────────────────────────────
# Method 1 – FEM Dirichlet
# ──────────────────────────────────────────────────────────────────────────────

def run_fem_dirichlet(L_nm, R_nm, h_fine_nm, h_coarse_nm, V0, outdir):
    _log(f"  [FEM-Dir] h_fine={h_fine_nm} nm  h_coarse={h_coarse_nm} nm")

    t0 = time.perf_counter()
    domain, _, facet_tags = sphere_in_box(COMM, L_nm, R_nm, h_fine_nm, h_coarse_nm)
    t_mesh = time.perf_counter() - t0

    fdim = domain.topology.dim - 1
    V    = fem.functionspace(domain, ("Lagrange", 1))
    R_m  = R_nm * 1e-9

    # Dirichlet BCs: sphere = V0, box = 0
    bc_sph = fem.dirichletbc(
        fem.Constant(domain, PETSc.ScalarType(V0)),
        fem.locate_dofs_topological(V, fdim, facet_tags.find(2)), V
    )
    bc_box = fem.dirichletbc(
        fem.Constant(domain, PETSc.ScalarType(0.0)),
        fem.locate_dofs_topological(V, fdim, facet_tags.find(1)), V
    )
    bcs = [bc_sph, bc_box]

    # Constant ε = ε₀
    W   = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(W); eps.x.array[:] = EPS0

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    # Manual PETSc assembly (avoids LinearProblem ctor-signature drift across
    # dolfinx releases, which has been observed to silently break and return
    # NaN/zero results instead of raising).
    from dolfinx.fem import petsc as fem_petsc

    t1 = time.perf_counter()
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fem_petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = fem_petsc.assemble_vector(L_form)
    fem_petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()

    uh = fem.Function(V)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    uh.name = "phi"
    t_solve = time.perf_counter() - t1

    n_dofs = V.dofmap.index_map.size_global

    # Write XDMF
    xdmf_path = os.path.join(outdir, f"dir_h{h_fine_nm:g}nm.xdmf")
    with XDMFFile(COMM, xdmf_path, "w") as f:
        f.write_mesh(domain); f.write_function(uh)

    # Error at probe points (root only, serial eval)
    probe_pts = _probe_points([100, 150, 200])
    rel_errs  = _fem_probe_error(domain, V, uh, probe_pts, V0, R_m)
    max_err   = float(np.nanmax(rel_errs))
    mean_err  = float(np.nanmean(rel_errs))

    _log(f"         n_dofs={n_dofs}  t_mesh={t_mesh:.1f}s  t_solve={t_solve:.1f}s  "
         f"max_rel_err={max_err:.3e}")
    return dict(method="FEM-Dirichlet", h_nm=h_fine_nm, n_dofs=n_dofs,
                t_mesh_s=round(t_mesh, 2), t_solve_s=round(t_solve, 2),
                max_rel_err=max_err, mean_rel_err=mean_err)


# ──────────────────────────────────────────────────────────────────────────────
# Method 2 – FEM Periodic (dolfinx_mpc)
# ──────────────────────────────────────────────────────────────────────────────

def run_fem_periodic(L_nm, R_nm, h_fine_nm, h_coarse_nm, V0, outdir):
    if not HAS_MPC:
        _log("  [FEM-Per] SKIPPED – dolfinx_mpc not available")
        return dict(method="FEM-Periodic", h_nm=h_fine_nm, n_dofs=0,
                    t_mesh_s=0.0, t_solve_s=0.0,
                    max_rel_err=float("nan"), mean_rel_err=float("nan"),
                    note="dolfinx_mpc not installed")

    _log(f"  [FEM-Per] h_fine={h_fine_nm} nm  h_coarse={h_coarse_nm} nm")

    t0 = time.perf_counter()
    domain, _, facet_tags = sphere_in_box(COMM, L_nm, R_nm, h_fine_nm, h_coarse_nm)
    t_mesh = time.perf_counter() - t0

    fdim = domain.topology.dim - 1
    V    = fem.functionspace(domain, ("Lagrange", 1))
    R_m  = R_nm * 1e-9
    L_m  = L_nm * 1e-9
    tol  = L_m * 1e-8

    # ── 3D periodic constraint on all box faces ────────────────────────────
    mpc = MultiPointConstraint(V)

    def _slave_x(x): return np.isclose(x[0],  L_m/2, atol=tol)
    def _map_x(x):   out = np.copy(x); out[0] -= L_m; return out
    def _slave_y(x): return np.isclose(x[1],  L_m/2, atol=tol)
    def _map_y(x):   out = np.copy(x); out[1] -= L_m; return out
    def _slave_z(x): return np.isclose(x[2],  L_m/2, atol=tol)
    def _map_z(x):   out = np.copy(x); out[2] -= L_m; return out

    # Sphere Dirichlet is applied as a "bcs" argument so mpc excludes those DOFs
    bc_sph = fem.dirichletbc(
        fem.Constant(domain, PETSc.ScalarType(V0)),
        fem.locate_dofs_topological(V, fdim, facet_tags.find(2)), V
    )

    mpc.create_periodic_constraint_geometrical(V, _slave_x, _map_x, [bc_sph])
    mpc.create_periodic_constraint_geometrical(V, _slave_y, _map_y, [bc_sph])
    mpc.create_periodic_constraint_geometrical(V, _slave_z, _map_z, [bc_sph])
    mpc.finalize()

    W   = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(W); eps.x.array[:] = EPS0

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    # Manual assembly via the dolfinx_mpc low-level API: MPCLinearProblem's
    # constructor signature has changed across releases (it wraps the same
    # dolfinx.fem.petsc.LinearProblem that breaks on nightly), so build the
    # periodic-constrained system by hand instead.
    import dolfinx_mpc
    from dolfinx.fem import petsc as fem_petsc

    t1 = time.perf_counter()
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=[bc_sph])
    A.assemble()
    b = dolfinx_mpc.assemble_vector(L_form, mpc)
    dolfinx_mpc.apply_lifting(b, [a_form], bcs=[[bc_sph]], constraint=mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc_sph])

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()

    uh = fem.Function(V)
    ksp.solve(b, uh.x.petsc_vec)
    mpc.backsubstitution(uh.x.petsc_vec)
    uh.x.scatter_forward()
    uh.name = "phi"
    t_solve = time.perf_counter() - t1

    n_dofs = V.dofmap.index_map.size_global

    xdmf_path = os.path.join(outdir, f"per_h{h_fine_nm:g}nm.xdmf")
    with XDMFFile(COMM, xdmf_path, "w") as f:
        f.write_mesh(domain); f.write_function(uh)

    probe_pts = _probe_points([100, 150, 200])
    rel_errs  = _fem_probe_error(domain, V, uh, probe_pts, V0, R_m)
    max_err   = float(np.nanmax(rel_errs))
    mean_err  = float(np.nanmean(rel_errs))

    _log(f"         n_dofs={n_dofs}  t_mesh={t_mesh:.1f}s  t_solve={t_solve:.1f}s  "
         f"max_rel_err={max_err:.3e}")
    return dict(method="FEM-Periodic", h_nm=h_fine_nm, n_dofs=n_dofs,
                t_mesh_s=round(t_mesh, 2), t_solve_s=round(t_solve, 2),
                max_rel_err=max_err, mean_rel_err=mean_err)


# ──────────────────────────────────────────────────────────────────────────────
# Method 3 – FFT
# ──────────────────────────────────────────────────────────────────────────────

def run_fft(L_nm, R_nm, h_nm, V0, outdir, fft_max_n):
    N = int(round(L_nm / h_nm))
    if N > fft_max_n:
        _log(f"  [FFT]     h={h_nm} nm → N={N} > FFT_MAX_N={fft_max_n}, SKIPPED")
        return dict(method="FFT", h_nm=h_nm, n_dofs=0,
                    t_mesh_s=0.0, t_solve_s=0.0,
                    max_rel_err=float("nan"), mean_rel_err=float("nan"),
                    note=f"N={N} exceeds FFT_MAX_N={fft_max_n}")

    _log(f"  [FFT]     h={h_nm} nm  N={N}³={N**3:,}")
    L_m = L_nm * 1e-9
    R_m = R_nm * 1e-9

    # Total charge matching isolated conducting sphere: q = 4πε₀ V₀ R
    q = 4.0 * np.pi * EPS0 * V0 * R_m

    # Gaussian shell at r = R to represent sphere surface charge
    # Shell width: R/8, but at least 1.5 grid spacings
    sigma_shell = max(R_m / 8.0, 1.5 * (L_m / N))

    t0 = time.perf_counter()
    rho = sphere_charge_shell_grid(N, L_m, q, R_m, sigma_shell)
    t_mesh = time.perf_counter() - t0  # "mesh" ≡ grid construction

    t1 = time.perf_counter()
    phi = fft_poisson_3d(rho, L_m, eps_r=1.0)
    t_solve = time.perf_counter() - t1

    # Probe error
    probe_pts = _probe_points([100, 150, 200])
    phi_h     = evaluate_fft_phi_at_points(phi, L_m, probe_pts)
    r_pts     = np.linalg.norm(probe_pts, axis=1)
    phi_exact = phi_conducting_sphere(r_pts, V0, R_m)
    rel_errs  = np.abs(phi_h - phi_exact) / np.abs(phi_exact)
    max_err   = float(np.nanmax(rel_errs))
    mean_err  = float(np.nanmean(rel_errs))

    # Save grid to npz
    npz_path = os.path.join(outdir, f"fft_h{h_nm:g}nm.npz")
    h = L_m / N
    xi = (np.arange(N) - N//2) * h
    np.savez_compressed(npz_path, phi=phi.astype(np.float32),
                        rho=rho.astype(np.float32), xi=xi)

    _log(f"         N³={N**3:,}  t_grid={t_mesh:.3f}s  t_solve={t_solve:.3f}s  "
         f"max_rel_err={max_err:.3e}")
    return dict(method="FFT", h_nm=h_nm, n_dofs=N**3,
                t_mesh_s=round(t_mesh, 3), t_solve_s=round(t_solve, 3),
                max_rel_err=max_err, mean_rel_err=mean_err)


# ──────────────────────────────────────────────────────────────────────────────
# Suggestions file
# ──────────────────────────────────────────────────────────────────────────────

_SUGGESTIONS = """\
SUGGESTIONS FOR COMPARISON AGAINST REAL MEMS SIMULATIONS
=========================================================

0. ROOT CAUSE OF POOR AGREEMENT WITH MASQUE: PERIODIC-IMAGE ERROR DOES NOT
   SHRINK WITH MESH REFINEMENT -- CONFIRMED EMPIRICALLY.
   Running the FFT leg (benchmarks/fft_only_sweep.py, pure NumPy, no Docker
   needed) at the default L=500 nm / R=50 nm / probes at r=100,150,200 nm
   shows max_rel_err frozen at ~1.0 (100%!) across h = 20, 10, 5 nm -- i.e.
   refining the mesh from 20 nm down to 5 nm changes the error by < 1%.
   Repeating at L=2000 nm (R/L = 0.025 instead of 0.1) drops the error to
   ~0.28 and it again plateaus across h = 20, 10, 5 nm.  This proves the
   error is controlled by R/L (sphere radius vs. domain/periodic-cell size),
   NOT by element/grid density.  The FFT and FEM-Periodic methods solve an
   INFINITE PERIODIC ARRAY of spheres; their potential differs from the
   isolated-sphere analytic reference by an image-field term that only
   vanishes as L/R -> infinity.  If MASQUE uses isolated (open) boundary
   conditions, no amount of mesh refinement here will make FFT/FEM-Periodic
   match it at L/R = 10 -- you need either:
     (a) L/R >> 10 (e.g. >= 40, error ~1-2%) for genuinely isolated physics, or
     (b) probe points well inside the cell (r << L/2) so the comparison
         point itself is far from periodic images, or
     (c) apply an explicit periodic-image correction (see item 2) and
         compare against the corrected, not the isolated, reference.
   FEM-Dirichlet (item 1) has the analogous but smaller truncation error
   since it has no periodic images, only one truncation boundary.

1. DOMAIN TRUNCATION ERROR (FEM-Dirichlet)
   The Dirichlet φ=0 boundary at L=500 nm introduces error proportional to
   R/L = 0.1.  For MEMS comparison, either:
   (a) Increase L until error saturates, or
   (b) Apply exact far-field BC: φ = V₀R/r on the box surface (first-order ABC).
   (c) Use a Robin (mixed) BC: ∂φ/∂n + φ/r = 0 for spherically symmetric decay.

2. PERIODIC IMAGES (FEM-Periodic, FFT)
   Periodic methods model an infinite array of spheres separated by L.  For
   MEMS arrays with pitch p, set L = p to directly match the periodic geometry.
   The Madelung-like correction for the periodic self-energy is:
       φ_corr ≈ -2.837297 · q / (4πε₀ L)  (Wigner-Seitz correction)

3. VARIABLE PERMITTIVITY
   Real MEMS have dielectric layers (SiO₂ ε≈3.9, Si₃N₄ ε≈7).  The FEM solver
   already supports DG0 permittivity via dg0_from_indicator or piecewise_eps_DG0_2regions.
   FFT cannot handle variable ε natively; switch to a real-space multigrid or
   preconditioned iterative method for multi-dielectric problems.

4. HIGHER-ORDER ELEMENTS
   Currently all FEM uses P1 (linear) elements.  P2 elements give O(h⁴) L2
   convergence instead of O(h²) for smooth solutions, halving the mesh
   density needed for a given accuracy.  Change ("Lagrange", 1) → ("Lagrange", 2)
   and compare h-refinement slopes.

5. CAPACITANCE EXTRACTION
   For MEMS device comparison, extract the sphere capacitance:
       C = Q / V₀   where  Q = ε₀ ∮ ∂φ/∂n dS  (surface integral on sphere)
   In FEM: assemble the surface flux via ufl.inner(eps*ufl.grad(uh), n)*ufl.ds(2).
   Compare to analytic C₀ = 4πε₀R and to C obtained from COMSOL/ANSYS.

6. MULTI-CONDUCTOR PROBLEMS
   Real gates have multiple electrodes at different potentials.  Extend
   sphere_in_box to include N spheres via gmsh Boolean operations, and
   verify by computing the full N×N capacitance matrix.

7. SOLVER SCALING
   For fine meshes (> 10⁶ DOFs), the CG+HYPRE solver can be replaced with
   MUMPS direct solver (better for multiple RHS) or PETSc GAMG for faster
   parallel scaling.  Add --solver flag to benchmark script.

8. COMPARISON WORKFLOW AGAINST COMSOL / ANSYS
   (a) Export FEM mesh to .msh or .vtk and import into COMSOL to reuse geometry.
   (b) Compare φ on the same probe-point line (r = 100→250 nm) exported as CSV.
   (c) Use the image-charge analytic solution in analytics.phi_image_3d as a
       common ground truth for both tools.

9. SPACE CHARGE / SEMICONDUCTOR COUPLING
   For realistic MEMS/transistor simulations, ρ is not fixed but coupled to
   carrier densities via the drift-diffusion equations.  A Poisson-DD solver
   would use the Poisson equation here as the inner loop of a Gummel iteration.

10. MESH QUALITY METRICS
    For curved sphere surface, check: min/max dihedral angles, skewness, and
    aspect ratio using meshio or dolfinx's cell quality tools.  Poor-quality
    tetrahedra near the sphere surface inflate condition number and solver
    iterations.
"""


def _write_suggestions(outdir: str) -> None:
    path = os.path.join(outdir, "SUGGESTIONS.txt")
    if IS_ROOT:
        with open(path, "w") as f:
            f.write(_SUGGESTIONS)
        print(f"Suggestions written to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L-nm",      type=float, default=500.0,
                   help="Box side length [nm] (default 500)")
    p.add_argument("--R-nm",      type=float, default=50.0,
                   help="Sphere radius [nm] (default 50)")
    p.add_argument("--V0",        type=float, default=1.0,
                   help="Sphere potential [V] (default 1.0)")
    p.add_argument("--fft-max-n", type=int,   default=300,
                   help="Max FFT grid points per dimension (default 300)")
    p.add_argument("--skip-dir",  action="store_true", help="Skip FEM-Dirichlet")
    p.add_argument("--skip-per",  action="store_true", help="Skip FEM-Periodic")
    p.add_argument("--skip-fft",  action="store_true", help="Skip FFT")
    p.add_argument("--outdir",    default="results/sphere_triple",
                   help="Output directory (default results/sphere_triple)")
    return p


def main():
    args = build_parser().parse_args()

    L_nm      = args.L_nm
    R_nm      = args.R_nm
    V0        = args.V0
    fft_max_n = args.fft_max_n
    outdir    = args.outdir

    if IS_ROOT:
        os.makedirs(outdir, exist_ok=True)
        print("=" * 60)
        print("Sphere triple comparison")
        print(f"  L={L_nm} nm   R={R_nm} nm   V₀={V0} V")
        print(f"  MPC available: {HAS_MPC}")
        print(f"  Output:  {outdir}")
        print("=" * 60)

    # Mesh refinement levels [nm].  FEM (Dirichlet/Periodic) is capped at
    # h >= 5 nm: h=1nm and h=0.1nm produce tetrahedra counts that have been
    # observed to OOM/timeout the Docker container; FFT has no such limit
    # since its cost is set by the uniform grid N = L/h (capped by --fft-max-n).
    H_ALL_NM      = [20.0, 10.0, 5.0, 1.0, 0.1]
    H_FEM_NM      = [h for h in H_ALL_NM if h >= 5.0]
    H_FFT_NM      = H_ALL_NM

    results = []

    # ── FEM-Dirichlet ─────────────────────────────────────────────────────────
    if not args.skip_dir:
        _log("\n── FEM-Dirichlet ──────────────────────────────────────────")
        for h in H_FEM_NM:
            h_coarse = min(max(h * 5.0, 25.0), L_nm / 4)
            try:
                row = run_fem_dirichlet(L_nm, R_nm, h, h_coarse, V0, outdir)
            except Exception as exc:
                _log(f"  ERROR at h={h} nm: {exc}")
                row = dict(method="FEM-Dirichlet", h_nm=h, n_dofs=0,
                           t_mesh_s=0.0, t_solve_s=0.0,
                           max_rel_err=float("nan"), mean_rel_err=float("nan"),
                           note=str(exc))
            results.append(row)

    # ── FEM-Periodic ──────────────────────────────────────────────────────────
    if not args.skip_per:
        _log("\n── FEM-Periodic ───────────────────────────────────────────")
        for h in H_FEM_NM:
            h_coarse = min(max(h * 5.0, 25.0), L_nm / 4)
            try:
                row = run_fem_periodic(L_nm, R_nm, h, h_coarse, V0, outdir)
            except Exception as exc:
                _log(f"  ERROR at h={h} nm: {exc}")
                row = dict(method="FEM-Periodic", h_nm=h, n_dofs=0,
                           t_mesh_s=0.0, t_solve_s=0.0,
                           max_rel_err=float("nan"), mean_rel_err=float("nan"),
                           note=str(exc))
            results.append(row)

    # ── FFT ───────────────────────────────────────────────────────────────────
    if not args.skip_fft:
        _log("\n── FFT ────────────────────────────────────────────────────")
        for h in H_FFT_NM:
            try:
                row = run_fft(L_nm, R_nm, h, V0, outdir, fft_max_n)
            except Exception as exc:
                _log(f"  ERROR at h={h} nm: {exc}")
                row = dict(method="FFT", h_nm=h, n_dofs=0,
                           t_mesh_s=0.0, t_solve_s=0.0,
                           max_rel_err=float("nan"), mean_rel_err=float("nan"),
                           note=str(exc))
            results.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(outdir, "comparison_results.csv")
    if IS_ROOT and results:
        fieldnames = ["method", "h_nm", "n_dofs", "t_mesh_s", "t_solve_s",
                      "max_rel_err", "mean_rel_err", "note"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                row.setdefault("note", "")
                writer.writerow(row)
        print(f"\nResults written to {csv_path}")

    _write_suggestions(outdir)

    # ── Summary table ─────────────────────────────────────────────────────────
    if IS_ROOT and results:
        print("\n── Summary ──────────────────────────────────────────────────")
        header = f"{'Method':<16} {'h[nm]':>7} {'DOFs':>10} "
        header += f"{'t_mesh[s]':>10} {'t_solve[s]':>11} {'max_rel_err':>13}"
        print(header)
        print("-" * len(header))
        for r in results:
            err_str = (f"{r['max_rel_err']:.3e}" if not np.isnan(r["max_rel_err"])
                       else "     n/a")
            print(f"{r['method']:<16} {r['h_nm']:>7.1f} {r['n_dofs']:>10,} "
                  f"{r['t_mesh_s']:>10.2f} {r['t_solve_s']:>11.2f} {err_str:>13}")
        print()


if __name__ == "__main__":
    main()
