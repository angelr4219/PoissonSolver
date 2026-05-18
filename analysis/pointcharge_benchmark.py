#!/usr/bin/env python3
"""
analysis/pointcharge_benchmark.py
-----------------------------------
Point-charge benchmark in the layered SiGe/sSi/SiGe stack.

IMPORTANT: This is PHYSICALLY SEPARATE from the Dirichlet disk-gate benchmark.
  - disk-gate cases: phi is set by Dirichlet BCs on the gate surface
  - point-charge cases: phi is driven by a localized volumetric charge source

This script places a Gaussian charge blob (approximating a point charge Q)
at a configurable location near the top of the semiconductor stack.
Analyzes the resulting potential for:
  1. Integrated charge accuracy
  2. Radial 1/r scaling (in the homogeneous-medium limit)
  3. Vertical decay through the stack
  4. Eps_r contrast effects at layer interfaces

All outputs are separate from disk-gate outputs by design.

Run via Docker:
  docker run --rm -it -v "$PWD":/app -w /app dolfinx/dolfinx:nightly \\
    sh -lc 'pip install -q scipy 2>/dev/null; \\
            /dolfinx-env/bin/python3 -u analysis/pointcharge_benchmark.py \\
              --outdir outputs/pointcharge_benchmark'

Configurable parameters (all have defaults):
  --Q_e          charge in units of e (default: 1.0)
  --sigma_nm     Gaussian width of charge blob (default: 5.0 nm)
  --source_x_nm  source centre x in box coordinates (default: box centre)
  --source_y_nm  source centre y in box coordinates
  --source_z_nm  source depth from top surface (default: 10 nm = in top SiGe layer)
  --Lx_nm --Ly_nm --Lz_nm  box size (default: 500 x 500 x 255 nm)
  --h_nm         mesh spacing (default: 5 nm)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io, mesh
from dolfinx.mesh import CellType
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS0     = 8.8541878128e-12
E_CHARGE = 1.602176634e-19


# ── Standard SiGe/sSi/SiGe layered stack ─────────────────────────────────────

LAYER_HEIGHTS_NM = [50.0, 5.0, 200.0]
LAYER_EPSR       = [13.05, 11.7, 13.05]
LAYER_NAMES      = ["SiGe_top", "sSi_QW", "SiGe_bot"]


def build_eps_r(msh, layer_heights_nm, layer_epsr):
    Q0 = fem.functionspace(msh, ("DG", 0))
    eps_fn = fem.Function(Q0)
    eps_fn.name = "eps_r"
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    cells = np.arange(ncells, dtype=np.int32)
    mids = mesh.compute_midpoints(msh, tdim, cells)
    zmid_nm = mids[:, 2] * 1e9
    arr = np.zeros(ncells)
    z0 = 0.0
    for h, eps in zip(layer_heights_nm, layer_epsr):
        sel = (zmid_nm >= z0 - 1e-9) & (zmid_nm < z0 + h + 1e-9)
        arr[sel] = eps
        z0 += h
    eps_fn.x.array[:] = arr
    return eps_fn


def gaussian_charge(msh, Q_C, sigma_m, cx_m, cy_m, cz_m):
    Q0 = fem.functionspace(msh, ("DG", 0))
    rho = fem.Function(Q0)
    rho.name = "rho_dg0"
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    cells = np.arange(ncells, dtype=np.int32)
    mids = mesh.compute_midpoints(msh, tdim, cells)
    dx = mids[:,0] - cx_m
    dy = mids[:,1] - cy_m
    dz = mids[:,2] - cz_m
    r2 = dx**2 + dy**2 + dz**2
    norm = Q_C / ((2*np.pi * sigma_m**2)**1.5)
    rho.x.array[:] = norm * np.exp(-r2 / (2*sigma_m**2))
    return rho


def eval_at_pts(u, pts_m):
    msh = u.function_space.mesh
    tree = bb_tree(msh, msh.topology.dim)
    cand = compute_collisions_points(tree, pts_m)
    coll = compute_colliding_cells(msh, cand, pts_m)
    n = pts_m.shape[0]
    inside = np.zeros(n, dtype=bool)
    cells_arr = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        links = coll.links(i)
        if len(links):
            inside[i] = True; cells_arr[i] = links[0]
    vals = np.full(n, np.nan)
    if np.any(inside):
        vals[inside] = np.asarray(u.eval(pts_m[inside], cells_arr[inside])).reshape(-1)
    return vals


def run(args):
    comm = MPI.COMM_WORLD

    Q_C    = args.Q_e * E_CHARGE
    Lx_m   = args.Lx_nm * 1e-9
    Ly_m   = args.Ly_nm * 1e-9
    Lz_m   = sum(LAYER_HEIGHTS_NM) * 1e-9
    sigma_m = args.sigma_nm * 1e-9
    cx_m   = args.source_x_nm * 1e-9 if args.source_x_nm is not None else 0.0
    cy_m   = args.source_y_nm * 1e-9 if args.source_y_nm is not None else 0.0
    cz_m   = args.source_z_nm * 1e-9   # depth from top = z coordinate

    if comm.rank == 0:
        print("\n=== Point-charge benchmark (layered stack) ===")
        print(f"  Q         = {Q_C:.3e} C  ({args.Q_e:.2f} e)")
        print(f"  sigma     = {args.sigma_nm:.1f} nm")
        print(f"  source    = ({cx_m*1e9:.0f}, {cy_m*1e9:.0f}, {cz_m*1e9:.0f}) nm  "
              f"(z measured from top surface)")
        print(f"  box       = {args.Lx_nm:.0f} × {args.Ly_nm:.0f} × {Lz_m*1e9:.0f} nm")
        print(f"  h         = {args.h_nm} nm")
        print("  SEPARATE from Dirichlet disk-gate cases")

    Nx = max(1, int(np.ceil(args.Lx_nm / args.h_nm)))
    Ny = max(1, int(np.ceil(args.Ly_nm / args.h_nm)))
    Nz = max(1, int(np.ceil(Lz_m * 1e9  / args.h_nm)))

    msh = mesh.create_box(
        comm,
        [[-0.5*Lx_m, -0.5*Ly_m, 0.0], [0.5*Lx_m, 0.5*Ly_m, Lz_m]],
        [Nx, Ny, Nz],
        cell_type=CellType.tetrahedron,
    )

    eps_fn = build_eps_r(msh, LAYER_HEIGHTS_NM, LAYER_EPSR)
    rho_fn = gaussian_charge(msh, Q_C, sigma_m, cx_m, cy_m, cz_m)

    # Verify integrated charge
    dx_meas = ufl.dx(domain=msh)
    Q_int_local = fem.assemble_scalar(fem.form(rho_fn * dx_meas))
    Q_int = comm.allreduce(Q_int_local, op=MPI.SUM)
    if comm.rank == 0:
        print(f"  Q_integrated = {Q_int:.6e} C  (target {Q_C:.6e}, err {abs(Q_int-Q_C)/abs(Q_C):.4%})")

    # Poisson solve: -div(eps0 * eps_r * grad(phi)) = rho
    V = fem.functionspace(msh, ("CG", 1))
    u_trial = ufl.TrialFunction(V)
    v_test  = ufl.TestFunction(V)

    # Dirichlet phi=0 on all walls
    all_bfacets = mesh.exterior_facet_indices(msh.topology)
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, all_bfacets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    a = ufl.inner(EPS0 * eps_fn * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = rho_fn * v_test * ufl.dx

    phi = fem.Function(V, name="phi")
    problem = LinearProblem(
        a, L, u=phi, bcs=[bc_zero],
        petsc_options_prefix="pc_bench_",
        petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                       "ksp_rtol": 1e-12, "ksp_atol": 1e-16},
    )
    problem.solve()
    phi.x.scatter_forward()

    if comm.rank == 0:
        print(f"  phi range: [{phi.x.array.min():.4e}, {phi.x.array.max():.4e}] V")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # XDMF output
    xdmf_path = outdir / "pointcharge_benchmark.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xf:
        xf.write_mesh(msh)
        xf.write_function(phi)

    if comm.rank != 0:
        return

    # ── Radial line cut (horizontal, through source centre) ──────────────────
    r_max = min(0.4*args.Lx_nm, args.Lx_nm/4) * 1e-9
    rs_m = np.linspace(2*sigma_m, r_max, 80)
    pts_rad = np.column_stack([cx_m + rs_m, np.full_like(rs_m, cy_m),
                               np.full_like(rs_m, cz_m)])
    phi_rad = eval_at_pts(phi, pts_rad)

    # Analytic: phi_analytic = Q / (4 pi eps0 eps_r(source) r) — free-space approx
    eps_at_source = LAYER_EPSR[0] if cz_m*1e9 < LAYER_HEIGHTS_NM[0] else \
                    LAYER_EPSR[1] if cz_m*1e9 < LAYER_HEIGHTS_NM[0]+LAYER_HEIGHTS_NM[1] else \
                    LAYER_EPSR[2]
    phi_analytic = Q_C / (4*np.pi*EPS0*eps_at_source * rs_m)

    valid = np.isfinite(phi_rad) & (phi_rad > 0)
    rs_nm_arr = rs_m * 1e9

    # Log-log slope
    if valid.sum() > 4:
        n_fit = max(4, 2*valid.sum()//3)
        idx = np.where(valid)[0][:n_fit]
        slope, _ = np.polyfit(np.log(rs_m[idx]), np.log(phi_rad[idx]), 1)
    else:
        slope = float("nan")
    print(f"  Radial 1/r slope (inner 2/3): {slope:.4f}  (expected ≈ -1.00)")

    # ── Vertical line cut (through source x,y) ────────────────────────────────
    z_arr_m = np.linspace(0.01*Lz_m, 0.99*Lz_m, 150)
    pts_vert = np.column_stack([np.full_like(z_arr_m, cx_m),
                                np.full_like(z_arr_m, cy_m), z_arr_m])
    phi_vert = eval_at_pts(phi, pts_vert)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Point-charge benchmark  Q={args.Q_e:.0f}e  σ={args.sigma_nm:.0f}nm  "
        f"source=({cx_m*1e9:.0f},{cy_m*1e9:.0f},{cz_m*1e9:.0f})nm",
        fontsize=10
    )

    ax = axes[0]
    ax.plot(rs_nm_arr[valid], phi_rad[valid],  "b.-", label="FEM phi")
    ax.plot(rs_nm_arr[valid], phi_analytic[valid], "r--", label=f"1/(4πε₀ε_r r), ε_r={eps_at_source:.2f}")
    ax.set_xlabel("r (nm)"); ax.set_ylabel("φ (V)")
    ax.set_title("Radial profile at source depth")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(rs_nm_arr[valid], phi_rad[valid],  "b.-", label=f"FEM (slope≈{slope:.2f})")
    ax.loglog(rs_nm_arr[valid], phi_analytic[valid], "r--", label="1/r analytic")
    ref_y = phi_analytic[valid]
    ax.loglog(rs_nm_arr[valid], ref_y[0]*(rs_nm_arr[valid]/rs_nm_arr[valid][0])**-1,
              "k:", alpha=0.5, label="1/r ref")
    ax.set_xlabel("r (nm)"); ax.set_ylabel("φ (V)")
    ax.set_title("log-log: slope ≈ -1 in homogeneous limit")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    for i, (h, nm, eps) in enumerate(zip(
            np.cumsum([0]+LAYER_HEIGHTS_NM[:-1]), LAYER_NAMES, LAYER_EPSR)):
        ax.axhspan(h, h+LAYER_HEIGHTS_NM[i], alpha=0.06,
                   color=["tab:blue","tab:orange","tab:green"][i],
                   label=f"{nm} ε={eps:.2f}")
    ax.plot(phi_vert, z_arr_m*1e9, "k", lw=1.5, label="phi(z)")
    ax.axhline(cz_m*1e9, color="red", ls="--", lw=0.8, label=f"source z={cz_m*1e9:.0f}nm")
    ax.set_xlabel("φ (V)"); ax.set_ylabel("z (nm)")
    ax.set_title("Vertical profile through source axis")
    ax.invert_yaxis()   # z=0 at top
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "pointcharge_benchmark.png", dpi=150)
    plt.close(fig)

    # Metrics JSON
    metrics = {
        "Q_e": args.Q_e, "Q_C": Q_C, "Q_integrated_C": Q_int,
        "charge_rel_err": float(abs(Q_int-Q_C)/abs(Q_C)),
        "sigma_nm": args.sigma_nm,
        "source_xyz_nm": [float(cx_m*1e9), float(cy_m*1e9), float(cz_m*1e9)],
        "box_nm": [args.Lx_nm, args.Ly_nm, float(Lz_m*1e9)],
        "h_nm": args.h_nm,
        "radial_slope": float(slope), "slope_target": -1.0,
        "slope_pass": bool(abs(slope - (-1.0)) < 0.15),
        "note": "SEPARATE from Dirichlet disk-gate benchmark — different physics",
    }
    (outdir / "pointcharge_benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2))

    print(f"\n  RESULT: {'PASS' if metrics['slope_pass'] else 'FAIL'}")
    print(f"    slope = {slope:.4f}  (target -1.00 ± 0.15)")
    print(f"  Outputs: {outdir}/")


def parse_args():
    ap = argparse.ArgumentParser(description="Point-charge benchmark in SiGe stack.")
    ap.add_argument("--Q_e",          type=float, default=1.0,
                    help="Charge in units of e (default 1.0)")
    ap.add_argument("--sigma_nm",     type=float, default=5.0,
                    help="Gaussian blob sigma in nm (default 5.0)")
    ap.add_argument("--source_x_nm",  type=float, default=None,
                    help="Source centre x in physical coords (default: 0 = box centre)")
    ap.add_argument("--source_y_nm",  type=float, default=None,
                    help="Source centre y (default: 0 = box centre)")
    ap.add_argument("--source_z_nm",  type=float, default=10.0,
                    help="Source depth from top surface z=0 in nm (default 10 = in top SiGe)")
    ap.add_argument("--Lx_nm",        type=float, default=500.0)
    ap.add_argument("--Ly_nm",        type=float, default=500.0)
    ap.add_argument("--h_nm",         type=float, default=5.0,
                    help="Uniform mesh spacing in nm (default 5.0)")
    ap.add_argument("--outdir",       default="outputs/pointcharge_benchmark")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
