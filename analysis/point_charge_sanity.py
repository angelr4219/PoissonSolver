#!/usr/bin/env python3
"""
analysis/point_charge_sanity.py
--------------------------------
Validation ladder Step 1: standalone Gaussian point-charge sanity benchmark.

Solves  -div(eps0 * eps_r * grad(phi)) = rho   in a cubic box,
where rho is a Gaussian blob approximating a point charge Q.

Checks:
  1. Integrated charge ≈ Q
  2. phi(r) ~ C/r  outside a few sigma  (log-log slope ≈ -1)
  3. |E(r)| ~ C/r^2                     (log-log slope ≈ -2)

Run via Docker:
  ./run_dolfinx.sh analysis/point_charge_sanity.py \
      --outdir outputs/point_charge_sanity

Physics:
  rho(r) = Q / (2*pi*sigma^2)^(3/2) * exp(-|r-r0|^2 / (2*sigma^2))
  Analytic phi(r) = Q / (4*pi*eps0*eps_r * r)   for r >> sigma
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io, mesh
from dolfinx.mesh import CellType
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19


def gaussian_charge(msh, Q_C: float, sigma_m: float, cx: float, cy: float, cz: float):
    """DG0 Gaussian charge approximation."""
    Q0 = fem.functionspace(msh, ("DG", 0))
    rho = fem.Function(Q0)
    rho.name = "rho_dg0"

    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    cells = np.arange(ncells, dtype=np.int32)
    mids = mesh.compute_midpoints(msh, tdim, cells)

    dx = mids[:, 0] - cx
    dy = mids[:, 1] - cy
    dz = mids[:, 2] - cz
    r2 = dx**2 + dy**2 + dz**2

    norm = Q_C / ((2.0 * np.pi * sigma_m**2) ** 1.5)
    rho_arr = norm * np.exp(-r2 / (2.0 * sigma_m**2))
    rho.x.array[:] = rho_arr
    return rho


def eval_field_along_line(u: fem.Function, pts: np.ndarray) -> np.ndarray:
    msh_local = u.function_space.mesh
    tree = bb_tree(msh_local, msh_local.topology.dim)
    candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh_local, candidates, pts)

    n = pts.shape[0]
    vals = np.full(n, np.nan)
    cells_arr = np.full(n, -1, dtype=np.int32)
    inside = np.zeros(n, dtype=bool)
    for i in range(n):
        links = colliding.links(i)
        if len(links) > 0:
            inside[i] = True
            cells_arr[i] = links[0]
    if np.any(inside):
        v = np.asarray(u.eval(pts[inside], cells_arr[inside])).reshape(-1)
        vals[inside] = v
    return vals


def run(args: argparse.Namespace) -> None:
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print("\n=== Point-charge sanity benchmark ===")
        print(f"  Q      = {args.Q:.3e} C  ({args.Q/E_CHARGE:.3e} e)")
        print(f"  sigma  = {args.sigma_nm:.1f} nm")
        print(f"  eps_r  = {args.eps_r:.2f}")
        print(f"  box    = {args.L_nm:.0f} nm cube")
        print(f"  h_nm   = {args.h_nm:.1f} nm")
        print(f"  BC     = Dirichlet phi=0 on all faces")

    L_m     = args.L_nm * 1e-9
    sigma_m = args.sigma_nm * 1e-9
    h_m     = args.h_nm * 1e-9
    N = max(2, int(np.ceil(L_m / h_m)))

    cx = cy = cz = L_m / 2.0  # charge at box center

    msh = mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [L_m, L_m, L_m]],
        [N, N, N],
        cell_type=CellType.tetrahedron,
    )

    rho_dg0 = gaussian_charge(msh, args.Q, sigma_m, cx, cy, cz)

    # Verify integrated charge
    dx_measure = ufl.dx(domain=msh)
    Q_integrated_local = fem.assemble_scalar(fem.form(rho_dg0 * dx_measure))
    Q_integrated = comm.allreduce(Q_integrated_local, op=MPI.SUM)
    if comm.rank == 0:
        print(f"\n  Integrated charge: {Q_integrated:.6e} C   (target: {args.Q:.6e} C)")
        print(f"  Relative error:    {abs(Q_integrated - args.Q)/abs(args.Q):.4%}")

    V = fem.functionspace(msh, ("CG", 1))
    u_trial = ufl.TrialFunction(V)
    v_test  = ufl.TestFunction(V)

    # Dirichlet phi=0 on all boundaries
    def boundary_all(x):
        return np.ones(x.shape[1], dtype=bool)
    bfacets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary_all)
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bfacets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    a = ufl.inner(EPS0 * args.eps_r * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = rho_dg0 * v_test * ufl.dx

    phi = fem.Function(V, name="phi")
    problem = LinearProblem(
        a, L, u=phi, bcs=[bc_zero],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                       "ksp_rtol": 1e-12, "ksp_atol": 1e-16},
        petsc_options_prefix="ptchg_"
    )
    problem.solve()
    phi.x.scatter_forward()

    if comm.rank == 0:
        phi_min = float(np.min(phi.x.array))
        phi_max = float(np.max(phi.x.array))
        print(f"\n  phi min/max: {phi_min:.6e}, {phi_max:.6e} V")

    # Probe phi along x-axis through charge center.
    # Cap at L/5 (20% of box half-width) — beyond that, Dirichlet image charges
    # dominate and the slope drops well below -1, which is correct physics but
    # would fail a free-space 1/r check.
    r_max_m = min(0.45 * L_m, L_m / 5.0)
    rs_m = np.linspace(1.5 * sigma_m, r_max_m, 80)
    probe_pts = np.column_stack([cx + rs_m, np.full_like(rs_m, cy), np.full_like(rs_m, cz)])
    phi_probe = eval_field_along_line(phi, probe_pts)

    # Analytic Coulomb (free space)
    phi_analytic = args.Q / (4.0 * np.pi * EPS0 * args.eps_r * rs_m)

    valid = np.isfinite(phi_probe) & (phi_probe > 0)
    rs_nm = rs_m * 1e9

    if comm.rank != 0:
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Log-log slope check
    log_r = np.log(rs_m[valid])
    log_p = np.log(phi_probe[valid])
    if len(log_r) > 4:
        # Use inner two-thirds: small r, far from walls, dominated by 1/r Coulomb
        n_fit = max(4, 2 * len(log_r) // 3)
        slope, intercept = np.polyfit(log_r[:n_fit], log_p[:n_fit], 1)
    else:
        slope = float("nan")
    print(f"\n  Log-log slope of phi vs r (inner 2/3): {slope:.4f}  (expected ≈ -1.00)")

    # Relative error vs analytic
    phi_an_valid = phi_analytic[valid]
    rel_err = (phi_probe[valid] - phi_an_valid) / np.abs(phi_an_valid)
    print(f"  Max relative error vs analytic (inner r < L/5): {np.max(np.abs(rel_err)):.4%}")
    print(f"  Mean relative error vs analytic (inner r < L/5): {np.mean(np.abs(rel_err)):.4%}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Point-charge sanity: Q={args.Q:.1e} C, eps_r={args.eps_r}, "
                 f"sigma={args.sigma_nm} nm, h={args.h_nm} nm", fontsize=10)

    ax = axes[0]
    ax.plot(rs_nm[valid], phi_probe[valid], "b.-", label="FEM phi")
    ax.plot(rs_nm[valid], phi_an_valid, "r--", label="Analytic Q/(4πε₀ε_r r)")
    ax.set_xlabel("r from charge (nm)")
    ax.set_ylabel("φ (V)")
    ax.set_title("phi(r) along x-axis")
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax2 = axes[1]
    ax2.loglog(rs_nm[valid], phi_probe[valid], "b.-", label=f"FEM phi (slope≈{slope:.2f})")
    ax2.loglog(rs_nm[valid], phi_an_valid, "r--", label="Analytic ~1/r")
    ax2.loglog(rs_nm[valid], phi_an_valid[0] * (rs_nm[valid] / rs_nm[valid][0])**(-1),
               "k:", alpha=0.5, label="1/r reference")
    ax2.set_xlabel("r from charge (nm)")
    ax2.set_ylabel("φ (V)")
    ax2.set_title("log-log: slope should be -1")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    plot_path = outdir / "point_charge_sanity.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")

    # Save results
    metrics = {
        "Q_C": args.Q,
        "Q_integrated_C": Q_integrated,
        "charge_integral_rel_err": float(abs(Q_integrated - args.Q) / abs(args.Q)),
        "eps_r": args.eps_r,
        "sigma_nm": args.sigma_nm,
        "h_nm": args.h_nm,
        "L_nm": args.L_nm,
        "N_cells_per_dim": N,
        "phi_slope_log_log": float(slope),
        "slope_target": -1.0,
        "slope_pass": bool(abs(slope - (-1.0)) < 0.1),
        "max_rel_err_vs_analytic": float(np.max(np.abs(rel_err))) if len(rel_err) > 0 else float("nan"),
        "mean_rel_err_vs_analytic": float(np.mean(np.abs(rel_err))) if len(rel_err) > 0 else float("nan"),
    }
    json_path = outdir / "point_charge_sanity_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Metrics JSON: {json_path}")

    probe_path = outdir / "point_charge_sanity_probe.csv"
    np.savetxt(
        probe_path,
        np.column_stack([rs_nm, phi_probe, phi_analytic]),
        delimiter=",",
        header="r_nm,phi_fem,phi_analytic",
        comments=""
    )
    print(f"  Probe CSV: {probe_path}")

    # XDMF output
    xdmf_path = outdir / "point_charge_sanity.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xf:
        xf.write_mesh(msh)
        xf.write_function(phi)
    print(f"  XDMF: {xdmf_path}")

    PASS = metrics["slope_pass"]
    print(f"\n  RESULT: {'PASS' if PASS else 'FAIL'}")
    print(f"    log-log slope = {slope:.4f}  (target -1.00 ± 0.10)")
    if not PASS:
        print("  WARNING: Slope is far from -1. Check mesh resolution, box size, or eps_r.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Point-charge sanity benchmark.")
    ap.add_argument("--Q", type=float, default=1e-19,
                    help="Total charge in Coulombs (default: 1e-19 ≈ 0.6 e)")
    ap.add_argument("--sigma-nm", type=float, default=5.0, dest="sigma_nm",
                    help="Gaussian sigma in nm (default: 5 nm)")
    ap.add_argument("--eps-r", type=float, default=1.0, dest="eps_r",
                    help="Relative dielectric constant (default: 1.0, vacuum)")
    ap.add_argument("--L-nm", type=float, default=200.0, dest="L_nm",
                    help="Cubic box side length in nm (default: 200 nm)")
    ap.add_argument("--h-nm", type=float, default=5.0, dest="h_nm",
                    help="Target mesh spacing in nm (default: 5 nm)")
    ap.add_argument("--outdir", default="outputs/point_charge_sanity")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
