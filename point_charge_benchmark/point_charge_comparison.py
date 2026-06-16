#!/usr/bin/env python3
"""Point-charge benchmark: two cases, one consolidated report.

Case A -- MESH SWEEP x BOX-SIZE SWEEP, two-region dielectric, FEM-Dirichlet
    A point charge q sits at z = +d_nm inside a half-space of permittivity
    eps_r1; the other half-space (z < 0) has eps_r2.  FEM-Dirichlet (phi=0
    on the outer box) is solved on a structured tetrahedral box mesh, and
    compared against the analytic two-dielectric image-charge solution
    (poisson.analytics.phi_image_3d).  Two 1-D sweeps are run:
        A1: box size fixed, mesh size h varied   (mesh-density sweep)
        A2: mesh size fixed, box size L varied    (domain-size sweep)

Case B -- DIRICHLET vs PERIODIC BOUNDARY CONDITIONS, uniform dielectric
    The same point charge (now in a single uniform medium, eps_r) is solved
    two ways at matched resolution:
        FEM-Dirichlet : structured box mesh, phi=0 on the outer box
        FFT-Periodic  : spectral solve on a uniform periodic grid
    Both are compared against the analytic isolated Coulomb potential
    (poisson.analytics.phi_point_charge) at the same probe points, so the
    *only* thing that differs between the two rows at a given probe radius
    is the boundary condition.

    NOTE: a true FEM-periodic leg (dolfinx_mpc) could not be run -- the
    dolfinx/dolfinx:nightly image used here does not ship dolfinx_mpc, and
    it is not pip-installable (no PyPI wheel; needs building from source
    against this exact dolfinx build).  FFT is used as the periodic-BC
    reference instead, since it solves the same periodic Poisson problem
    exactly (spectrally) rather than approximately.

Output: ONE consolidated report, results/point_charge/report.txt
        (plus a .csv per case for raw numbers, and XDMF fields for ParaView)

Usage:
    ./run_dolfinx.sh benchmarks/point_charge_comparison.py [--help]
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
from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poisson.charges import gaussian_rho
from poisson.analytics import phi_image_3d, phi_point_charge, EPS0
from poisson.fft_solve import fft_poisson_3d, gaussian_rho_grid, evaluate_fft_phi_at_points

COMM = MPI.COMM_WORLD
RANK = COMM.rank
IS_ROOT = RANK == 0


def _log(msg: str) -> None:
    if IS_ROOT:
        print(msg, flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Shared mesh / solve helpers
# ──────────────────────────────────────────────────────────────────────────

def build_box_mesh(L_m: float, h_m: float):
    N = max(2, int(round(L_m / h_m)))
    p0 = np.array([-L_m / 2] * 3)
    p1 = np.array([+L_m / 2] * 3)
    domain = dmesh.create_box(COMM, points=[p0, p1], n=[N, N, N],
                              cell_type=dmesh.CellType.tetrahedron)
    return domain, N


def solve_fem_dirichlet(domain, eps_form, rho_form, V):
    """Manual PETSc assembly: -div(eps*grad(u)) = rho, u=0 on boundary."""
    fdim = domain.topology.dim - 1
    facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc_dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(eps_form * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho_form * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fem_petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = fem_petsc.assemble_vector(L_form)
    fem_petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc])

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
    return uh


def eval_at_points(domain, uh, pts_m):
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(domain, domain.topology.dim)
    colls = compute_collisions_points(tree, pts_m)
    cells = compute_colliding_cells(domain, colls, pts_m)
    vals = []
    for i, pt in enumerate(pts_m):
        cell_list = cells.links(i)
        if len(cell_list) > 0:
            v = float(np.asarray(uh.eval(pt.reshape(1, -1), cell_list[:1])).reshape(-1)[0])
        else:
            v = float("nan")
        vals.append(v)
    return np.array(vals)


def probe_points_z_axis(probe_r_nm, z0_nm=0.0):
    """Probe points displaced along x, offset to z0 (used for the dielectric
    interface case so probes stay on one side of the interface)."""
    pts = []
    for r_nm in probe_r_nm:
        pts.append([r_nm * 1e-9, 0.0, z0_nm * 1e-9])
    return np.array(pts, dtype=float)


def probe_points_diag(probe_r_nm):
    pts = []
    for r_nm in probe_r_nm:
        r = r_nm * 1e-9 / np.sqrt(3.0)
        pts.append([r, r, r])
    return np.array(pts, dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# Case A -- dielectric, mesh-density x box-size sweep
# ──────────────────────────────────────────────────────────────────────────

def run_case_a_point(L_nm, h_nm, q, d_nm, sigma_nm, eps_r1, eps_r2, probe_r_nm):
    L_m, h_m, d_m, sigma_m = L_nm * 1e-9, h_nm * 1e-9, d_nm * 1e-9, sigma_nm * 1e-9

    t0 = time.perf_counter()
    domain, N = build_box_mesh(L_m, h_m)
    t_mesh = time.perf_counter() - t0

    V = fem.functionspace(domain, ("Lagrange", 1))
    W = fem.functionspace(domain, ("DG", 0))

    eps1, eps2 = EPS0 * eps_r1, EPS0 * eps_r2
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    cells_top = dmesh.locate_entities(domain, tdim, lambda x: x[2] >= 0.0)
    eps = fem.Function(W)
    eps.x.array[:] = eps2
    if len(cells_top) > 0:
        dofs_top = fem.locate_dofs_topological(W, tdim, cells_top)
        eps.x.array[dofs_top] = eps1

    rho = gaussian_rho(domain, (0.0, 0.0, d_m), q, sigma_m)

    t1 = time.perf_counter()
    uh = solve_fem_dirichlet(domain, eps, rho, V)
    t_solve = time.perf_counter() - t1

    n_dofs = V.dofmap.index_map.size_global

    pts = probe_points_z_axis(probe_r_nm, z0_nm=d_nm)
    phi_h = eval_at_points(domain, uh, pts)
    phi_ex = phi_image_3d(pts, q, d_m, eps1, eps2)
    rel_err = np.abs(phi_h - phi_ex) / np.abs(phi_ex)

    return dict(L_nm=L_nm, h_nm=h_nm, N=N, n_dofs=n_dofs,
                t_mesh_s=round(t_mesh, 3), t_solve_s=round(t_solve, 3),
                phi_h=phi_h, phi_ex=phi_ex, rel_err=rel_err,
                max_rel_err=float(np.max(rel_err)), mean_rel_err=float(np.mean(rel_err)))


def run_case_a(args, probe_r_nm):
    _log("\n── Case A: dielectric point charge, mesh x box-size sweep ──────")
    rows = []

    _log(f"  A1: fixed L={args.caseA_L_fixed} nm, h in {args.caseA_h_sweep}")
    for h_nm in args.caseA_h_sweep:
        r = run_case_a_point(args.caseA_L_fixed, h_nm, args.q, args.d_nm,
                              args.sigma_nm, args.eps_r1, args.eps_r2, probe_r_nm)
        r["sweep"] = "A1_mesh_density"
        rows.append(r)
        _log(f"    h={h_nm:>5.1f}nm  N={r['N']:>3d}  dofs={r['n_dofs']:>8,}  "
             f"t_mesh={r['t_mesh_s']:>6.2f}s  t_solve={r['t_solve_s']:>6.2f}s  "
             f"max_rel_err={r['max_rel_err']:.3e}")

    _log(f"  A2: fixed h={args.caseA_h_fixed} nm, L in {args.caseA_L_sweep}")
    for L_nm in args.caseA_L_sweep:
        r = run_case_a_point(L_nm, args.caseA_h_fixed, args.q, args.d_nm,
                              args.sigma_nm, args.eps_r1, args.eps_r2, probe_r_nm)
        r["sweep"] = "A2_box_size"
        rows.append(r)
        _log(f"    L={L_nm:>5.1f}nm  N={r['N']:>3d}  dofs={r['n_dofs']:>8,}  "
             f"t_mesh={r['t_mesh_s']:>6.2f}s  t_solve={r['t_solve_s']:>6.2f}s  "
             f"max_rel_err={r['max_rel_err']:.3e}")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Case B -- Dirichlet vs Periodic (FFT), uniform dielectric
# ──────────────────────────────────────────────────────────────────────────

def run_case_b(args, probe_r_nm):
    _log("\n── Case B: Dirichlet (FEM) vs Periodic (FFT), uniform dielectric ─")
    L_nm, h_nm = args.caseB_L, args.caseB_h
    L_m, h_m = L_nm * 1e-9, h_nm * 1e-9
    eps_r = args.caseB_eps_r
    q = args.q
    sigma_m = args.sigma_nm * 1e-9

    pts = probe_points_diag(probe_r_nm)
    r_pts = np.linalg.norm(pts, axis=1)
    phi_ex = phi_point_charge(r_pts, q, eps_r)

    rows = []

    # FEM-Dirichlet
    t0 = time.perf_counter()
    domain, N = build_box_mesh(L_m, h_m)
    t_mesh = time.perf_counter() - t0
    V = fem.functionspace(domain, ("Lagrange", 1))
    rho = gaussian_rho(domain, (0.0, 0.0, 0.0), q, sigma_m)
    t1 = time.perf_counter()
    uh = solve_fem_dirichlet(domain, fem.Constant(domain, PETSc.ScalarType(EPS0 * eps_r)), rho, V)
    t_solve = time.perf_counter() - t1
    phi_h_dir = eval_at_points(domain, uh, pts)
    rel_err_dir = np.abs(phi_h_dir - phi_ex) / np.abs(phi_ex)
    n_dofs = V.dofmap.index_map.size_global
    _log(f"  FEM-Dirichlet  N={N:>3d}  dofs={n_dofs:>8,}  t_mesh={t_mesh:.2f}s  "
         f"t_solve={t_solve:.2f}s  max_rel_err={float(np.max(rel_err_dir)):.3e}")
    rows.append(dict(method="FEM-Dirichlet", L_nm=L_nm, h_nm=h_nm, N=N, n_dofs=n_dofs,
                      t_mesh_s=round(t_mesh, 3), t_solve_s=round(t_solve, 3),
                      phi_h=phi_h_dir, phi_ex=phi_ex, rel_err=rel_err_dir,
                      max_rel_err=float(np.max(rel_err_dir)),
                      mean_rel_err=float(np.mean(rel_err_dir))))

    xdmf_path = os.path.join(args.outdir, "caseB_fem_dirichlet.xdmf")
    with XDMFFile(COMM, xdmf_path, "w") as f:
        f.write_mesh(domain)
        f.write_function(uh)

    # FFT-Periodic
    N_fft = max(2, int(round(L_m / h_m)))
    t0 = time.perf_counter()
    rho_grid = gaussian_rho_grid(N_fft, L_m, q, sigma_m)
    t_grid = time.perf_counter() - t0
    t1 = time.perf_counter()
    phi_grid = fft_poisson_3d(rho_grid, L_m, eps_r=eps_r)
    t_solve_fft = time.perf_counter() - t1
    phi_h_fft = evaluate_fft_phi_at_points(phi_grid, L_m, pts)
    rel_err_fft = np.abs(phi_h_fft - phi_ex) / np.abs(phi_ex)
    _log(f"  FFT-Periodic   N={N_fft:>3d}  dofs={N_fft**3:>8,}  t_grid={t_grid:.2f}s  "
         f"t_solve={t_solve_fft:.2f}s  max_rel_err={float(np.max(rel_err_fft)):.3e}")
    rows.append(dict(method="FFT-Periodic", L_nm=L_nm, h_nm=h_nm, N=N_fft, n_dofs=N_fft**3,
                      t_mesh_s=round(t_grid, 3), t_solve_s=round(t_solve_fft, 3),
                      phi_h=phi_h_fft, phi_ex=phi_ex, rel_err=rel_err_fft,
                      max_rel_err=float(np.max(rel_err_fft)),
                      mean_rel_err=float(np.mean(rel_err_fft))))

    np.savez_compressed(os.path.join(args.outdir, "caseB_fft_periodic.npz"),
                         phi=phi_grid.astype(np.float32), rho=rho_grid.astype(np.float32))

    return rows, r_pts


# ──────────────────────────────────────────────────────────────────────────
# Consolidated report
# ──────────────────────────────────────────────────────────────────────────

def write_report(args, rows_a, rows_b, probe_r_a_nm, probe_r_b_nm, r_pts_b, outdir):
    path = os.path.join(outdir, "report.txt")
    lines = []
    lines.append("=" * 78)
    lines.append("POINT-CHARGE BENCHMARK -- CONSOLIDATED RAW RESULTS")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Charge q = {args.q:.6e} C   Gaussian sigma = {args.sigma_nm} nm")
    lines.append("")

    # ---- Case A ----
    lines.append("-" * 78)
    lines.append("CASE A: point charge in a two-region dielectric (FEM-Dirichlet),")
    lines.append(f"        mesh-density sweep x box-size sweep")
    lines.append(f"        eps_r1 (z>=0) = {args.eps_r1}, eps_r2 (z<0) = {args.eps_r2}, "
                 f"charge at z = +{args.d_nm} nm")
    lines.append(f"        Reference: analytic two-dielectric image-charge solution")
    lines.append(f"        Probe points: x = {probe_r_a_nm} nm along x-axis at z = {args.d_nm} nm")
    lines.append("-" * 78)
    header = (f"{'sweep':<16} {'L[nm]':>7} {'h[nm]':>7} {'N':>5} {'dofs':>9} "
              f"{'t_mesh[s]':>10} {'t_solve[s]':>11} {'max_relerr':>11} {'mean_relerr':>12}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows_a:
        lines.append(f"{r['sweep']:<16} {r['L_nm']:>7.1f} {r['h_nm']:>7.1f} {r['N']:>5d} "
                     f"{r['n_dofs']:>9,} {r['t_mesh_s']:>10.3f} {r['t_solve_s']:>11.3f} "
                     f"{r['max_rel_err']:>11.3e} {r['mean_rel_err']:>12.3e}")
    lines.append("")
    lines.append("  Raw phi at each probe point (FEM vs analytic), per run:")
    for r in rows_a:
        lines.append(f"  [{r['sweep']} L={r['L_nm']}nm h={r['h_nm']}nm]")
        for rr, ph, pe, re in zip(probe_r_a_nm, r["phi_h"], r["phi_ex"], r["rel_err"]):
            lines.append(f"      r={rr:>6.1f}nm   phi_FEM={ph: .6e} V   "
                         f"phi_exact={pe: .6e} V   rel_err={re:.3e}")
    lines.append("")

    # ---- Case B ----
    lines.append("-" * 78)
    lines.append("CASE B: point charge, uniform dielectric -- Dirichlet vs Periodic BC")
    lines.append(f"        eps_r = {args.caseB_eps_r}, L = {args.caseB_L} nm, h = {args.caseB_h} nm")
    lines.append(f"        Reference: analytic isolated Coulomb potential")
    lines.append(f"        NOTE: true FEM-Periodic (dolfinx_mpc) unavailable in this image")
    lines.append(f"        (no PyPI wheel; not preinstalled) -- FFT used as the periodic-BC leg")
    lines.append(f"        since it solves the exact periodic Poisson equation spectrally.")
    lines.append("-" * 78)
    header2 = (f"{'method':<16} {'N':>6} {'dofs':>10} {'t_mesh/grid[s]':>15} "
               f"{'t_solve[s]':>11} {'max_relerr':>11} {'mean_relerr':>12}")
    lines.append(header2)
    lines.append("-" * len(header2))
    for r in rows_b:
        lines.append(f"{r['method']:<16} {r['N']:>6d} {r['n_dofs']:>10,} "
                     f"{r['t_mesh_s']:>15.3f} {r['t_solve_s']:>11.3f} "
                     f"{r['max_rel_err']:>11.3e} {r['mean_rel_err']:>12.3e}")
    lines.append("")
    lines.append("  Raw phi at each probe point (diagonal r = x=y=z), per method:")
    for r in rows_b:
        lines.append(f"  [{r['method']}]")
        for rr, ph, pe, re in zip(r_pts_b, r["phi_h"], r["phi_ex"], r["rel_err"]):
            lines.append(f"      r={rr*1e9:>6.1f}nm   phi={ph: .6e} V   "
                         f"phi_exact={pe: .6e} V   rel_err={re:.3e}")
    lines.append("")
    lines.append("=" * 78)

    text = "\n".join(lines)
    if IS_ROOT:
        with open(path, "w") as f:
            f.write(text)
        print("\n" + text)
        print(f"\nConsolidated report written to {path}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q", type=float, default=1.602176634e-19, help="Charge [C] (default +e)")
    p.add_argument("--sigma-nm", type=float, default=5.0, help="Gaussian regularisation width [nm]")
    p.add_argument("--d-nm", type=float, default=30.0, help="Case A: charge height above interface [nm]")
    p.add_argument("--eps-r1", type=float, default=1.0, help="Case A: eps_r for z>=0")
    p.add_argument("--eps-r2", type=float, default=3.9, help="Case A: eps_r for z<0 (SiO2-like)")
    p.add_argument("--caseA-L-fixed", type=float, default=400.0, help="Case A1: fixed box size [nm]")
    p.add_argument("--caseA-h-sweep", type=float, nargs="+", default=[40.0, 20.0, 10.0, 5.0],
                   help="Case A1: mesh sizes to sweep [nm]")
    p.add_argument("--caseA-h-fixed", type=float, default=10.0, help="Case A2: fixed mesh size [nm]")
    p.add_argument("--caseA-L-sweep", type=float, nargs="+", default=[200.0, 400.0, 800.0],
                   help="Case A2: box sizes to sweep [nm]")
    p.add_argument("--caseB-L", type=float, default=400.0, help="Case B: box size [nm]")
    p.add_argument("--caseB-h", type=float, default=10.0, help="Case B: mesh/grid size [nm]")
    p.add_argument("--caseB-eps-r", type=float, default=1.0, help="Case B: uniform eps_r")
    p.add_argument("--outdir", default="results/point_charge", help="Output directory")
    return p


def main():
    args = build_parser().parse_args()
    if IS_ROOT:
        os.makedirs(args.outdir, exist_ok=True)
        print("=" * 60)
        print("Point-charge benchmark: Case A (mesh x box sweep) + Case B (BC comparison)")
        print("=" * 60)

    probe_r_a_nm = [40.0, 60.0, 80.0]
    probe_r_b_nm = [40.0, 80.0, 150.0]

    rows_a = run_case_a(args, probe_r_a_nm)
    rows_b, r_pts_b = run_case_b(args, probe_r_b_nm)

    # CSV dumps (scalar summary rows only; raw per-probe values are in report.txt)
    csv_a = os.path.join(args.outdir, "caseA_results.csv")
    with open(csv_a, "w", newline="") as f:
        fieldnames = ["sweep", "L_nm", "h_nm", "N", "n_dofs", "t_mesh_s", "t_solve_s",
                      "max_rel_err", "mean_rel_err"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_a)

    csv_b = os.path.join(args.outdir, "caseB_results.csv")
    with open(csv_b, "w", newline="") as f:
        fieldnames = ["method", "L_nm", "h_nm", "N", "n_dofs", "t_mesh_s", "t_solve_s",
                      "max_rel_err", "mean_rel_err"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_b)

    write_report(args, rows_a, rows_b, probe_r_a_nm, probe_r_b_nm, r_pts_b, args.outdir)


if __name__ == "__main__":
    main()
