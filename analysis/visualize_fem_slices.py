#!/usr/bin/env python3
"""
analysis/visualize_fem_slices.py
----------------------------------
Standalone FEM slice visualizer — no VTK comparison needed.

Reads a DOLFINx XDMF/H5 result and produces publication-quality 2D
heatmaps of phi at specified z-depths. Useful for showing Leah that
the FEM produces the correct disk-gate signature at each depth.

Run via Docker:
  docker run --rm -it -v "$PWD":/app -w /app dolfinx/dolfinx:nightly \
    sh -lc 'pip install -q scipy 2>/dev/null; \
            /dolfinx-env/bin/python3 -u analysis/visualize_fem_slices.py \
              --xdmf results/basepot_R10_exact_tet_h6_neg1_neg12/basepot_R10_exact_tet_h6_neg1_neg12.xdmf \
              --outdir outputs/fem_slices'
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import griddata

from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points


# Physics parameters for gate-active case
V_GATE   = -1.0   # V
V_BOTTOM = -12.0  # V
R_DISK_NM = 10.0  # nm  (for circle overlay)


def eval_phi_at_z(u: fem.Function, z_m: float, nx: int, ny: int,
                  x_lim: tuple, y_lim: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate phi on an nx×ny Cartesian grid at fixed z = z_m."""
    msh = u.function_space.mesh
    xg = np.linspace(x_lim[0], x_lim[1], nx)
    yg = np.linspace(y_lim[0], y_lim[1], ny)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    pts = np.column_stack([
        XX.ravel(),
        YY.ravel(),
        np.full(nx * ny, z_m),
    ])

    tree = bb_tree(msh, msh.topology.dim)
    candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh, candidates, pts)

    phi_flat = np.full(nx * ny, np.nan)
    inside = np.zeros(nx * ny, dtype=bool)
    cells_arr = np.full(nx * ny, -1, dtype=np.int32)
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            inside[i] = True
            cells_arr[i] = links[0]
    if np.any(inside):
        vals = np.asarray(u.eval(pts[inside], cells_arr[inside])).ravel()
        phi_flat[inside] = vals

    phi_2d = phi_flat.reshape(nx, ny)
    return xg, yg, phi_2d


def run(args: argparse.Namespace) -> None:
    comm = MPI.COMM_WORLD
    outdir = Path(args.outdir)
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== FEM slice visualizer ===")
        print(f"  XDMF  : {args.xdmf}")
        print(f"  z-slices (nm): {args.slices_nm}")
        print(f"  grid  : {args.grid_n}×{args.grid_n}")

    with io.XDMFFile(comm, args.xdmf, "r") as xf:
        msh = xf.read_mesh(name="mesh")
        V = fem.functionspace(msh, ("CG", 1))
        u = fem.Function(V, name="phi")
        xf.read_function(u)

    coords = msh.geometry.x
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    Lx_nm = (x_max - x_min) * 1e9
    Ly_nm = (y_max - y_min) * 1e9
    cx_nm = (x_min + x_max) / 2 * 1e9
    cy_nm = (y_min + y_max) / 2 * 1e9

    if comm.rank == 0:
        print(f"  Box   : {Lx_nm:.0f}×{Ly_nm:.0f} nm, "
              f"z=[{z_min*1e9:.0f}, {z_max*1e9:.0f}] nm")

    n = args.grid_n
    slices = sorted(args.slices_nm)

    # ----- per-slice plots -----
    for z_nm in slices:
        z_m = z_nm * 1e-9
        if z_m < z_min or z_m > z_max:
            if comm.rank == 0:
                print(f"  [SKIP] z={z_nm} nm outside mesh range")
            continue

        xg, yg, phi = eval_phi_at_z(
            u, z_m, n, n,
            (x_min, x_max), (y_min, y_max)
        )
        xg_nm = xg * 1e9
        yg_nm = yg * 1e9

        if comm.rank != 0:
            continue

        valid = np.isfinite(phi)
        phi_min = float(np.nanmin(phi))
        phi_max = float(np.nanmax(phi))
        print(f"  z={z_nm:5.0f} nm : phi [{phi_min:.4f}, {phi_max:.4f}] V  "
              f"  (Δ = {phi_max - phi_min:.4f} V)")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(
            xg_nm, yg_nm, phi.T,
            shading="auto", cmap="RdBu_r",
            vmin=V_BOTTOM, vmax=V_GATE,
        )
        cb = fig.colorbar(im, ax=ax, label="φ (V)")
        cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # Circle overlay marking disk gate boundary
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(cx_nm + R_DISK_NM * np.cos(theta),
                cy_nm + R_DISK_NM * np.sin(theta),
                "k--", lw=1.0, label=f"disk R={R_DISK_NM:.0f} nm")

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(
            f"DOLFINx φ at z={z_nm:.0f} nm\n"
            f"V_gate={V_GATE} V, V_bottom={V_BOTTOM} V, "
            f"R_disk={R_DISK_NM:.0f} nm",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.set_aspect("equal")

        fig.tight_layout()
        fname = outdir / f"fem_phi_z{z_nm:.0f}nm.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"    → {fname}")

    # ----- summary panel: all slices in one figure -----
    if comm.rank != 0:
        return

    n_slices = len(slices)
    ncols = min(4, n_slices)
    nrows = (n_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for idx, z_nm in enumerate(slices):
        z_m = z_nm * 1e-9
        if z_m < z_min or z_m > z_max:
            axes_flat[idx].set_visible(False)
            continue

        xg, yg, phi = eval_phi_at_z(
            u, z_m, n, n,
            (x_min, x_max), (y_min, y_max)
        )
        xg_nm = xg * 1e9
        yg_nm = yg * 1e9

        ax = axes_flat[idx]
        im = ax.pcolormesh(xg_nm, yg_nm, phi.T,
                           shading="auto", cmap="RdBu_r",
                           vmin=V_BOTTOM, vmax=V_GATE)
        fig.colorbar(im, ax=ax, shrink=0.85, label="φ (V)")

        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(cx_nm + R_DISK_NM * np.cos(theta),
                cy_nm + R_DISK_NM * np.sin(theta),
                "k--", lw=0.8)
        ax.set_title(f"z = {z_nm:.0f} nm", fontsize=9)
        ax.set_xlabel("x (nm)", fontsize=7)
        ax.set_ylabel("y (nm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")

    for idx in range(n_slices, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"DOLFINx gate-active FEM — φ at multiple depths\n"
        f"V_gate={V_GATE} V, V_bottom={V_BOTTOM} V, "
        f"R_disk={R_DISK_NM:.0f} nm, box={Lx_nm:.0f}×{Ly_nm:.0f} nm",
        fontsize=10,
    )
    fig.tight_layout()
    summary_path = outdir / "fem_phi_all_slices.png"
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\n  Summary panel: {summary_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FEM slice visualizer.")
    ap.add_argument("--xdmf", required=True,
                    help="Path to DOLFINx XDMF file")
    ap.add_argument("--slices-nm", type=float, nargs="+", dest="slices_nm",
                    default=[0, 10, 20, 50, 100, 155],
                    help="z-depths to visualize in nm (default: 0 10 20 50 100 155)")
    ap.add_argument("--grid-n", type=int, default=301, dest="grid_n",
                    help="Grid points per axis (default 301, higher = smoother)")
    ap.add_argument("--outdir", default="outputs/fem_slices",
                    help="Output directory for PNGs")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
