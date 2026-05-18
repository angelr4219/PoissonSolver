#!/usr/bin/env python3
"""
analysis/compare_background_cases.py
--------------------------------------
Compare two XDMF background cases (e.g. blank vs biased_bg) on shared z-slices.

Typical use:
  blank_exact      = results/blank_exact_tet_h6/
  biased_bg_exact  = results/biased_bg_exact_tet_h6/

Produces per-slice figures:
  [Case A phi | Case B phi | diff B-A | % diff]
Plus a depth-profile comparison.

Run via Docker:
  docker run --rm -it -v "$PWD":/app -w /app dolfinx/dolfinx:nightly \\
    sh -lc 'pip install -q h5py scipy 2>/dev/null; \\
            /dolfinx-env/bin/python3 -u analysis/compare_background_cases.py \\
              --xdmf-a results/blank_exact_tet_h6/blank_exact_tet_h6.xdmf \\
              --xdmf-b results/biased_bg_exact_tet_h6/biased_bg_exact_tet_h6.xdmf \\
              --label-a "Blank (V=0)" \\
              --label-b "Biased BG (Vbot=-12V)" \\
              --outdir outputs/background_compare \\
              --slices-nm 0 10 20 50 100 155 255'
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same-directory import when run inside Docker with PYTHONPATH=/app
sys.path.insert(0, str(Path(__file__).parent))
from xdmf_utils import (load_xdmf_field, eval_on_grid, mesh_xy_range,
                         eval_depth_line, disk_circle)

PCT_FLOOR = 0.01   # 10 mV denominator floor for % error
R_DISK_NM = 10.0   # disk circle overlay radius


def compare_at_z(u_a, u_b, z_nm, grid_n, x_range, y_range, label_a, label_b,
                 outdir, prefix):
    xg, yg, phi_a = eval_on_grid(u_a, z_nm, grid_n, x_range, y_range)
    xg, yg, phi_b = eval_on_grid(u_b, z_nm, grid_n, x_range, y_range)

    mask = np.isfinite(phi_a) & np.isfinite(phi_b)
    diff = phi_b - phi_a
    denom = np.maximum(np.abs(phi_a), PCT_FLOOR)
    pct  = np.where(mask, 100.0 * np.abs(diff) / denom, np.nan)

    cx_nm = 0.5 * (x_range[0] + x_range[1])
    cy_nm = 0.5 * (y_range[0] + y_range[1])

    vlo = min(np.nanmin(phi_a), np.nanmin(phi_b))
    vhi = max(np.nanmax(phi_a), np.nanmax(phi_b))
    dabs = np.nanmax(np.abs(diff)) or 1e-6

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Background comparison   z = {z_nm:.0f} nm", fontsize=11)

    def _im(ax, data, title, cmap, vmin=None, vmax=None):
        ext = [xg.min(), xg.max(), yg.min(), yg.max()]
        im = ax.imshow(data.T, origin="lower", extent=ext, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (nm)", fontsize=8); ax.set_ylabel("y (nm)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        disk_circle(ax, cx_nm, cy_nm, R_DISK_NM)

    _im(axes[0,0], phi_a, f"{label_a}\nφ (V)", "RdBu_r", vlo, vhi)
    _im(axes[0,1], phi_b, f"{label_b}\nφ (V)", "RdBu_r", vlo, vhi)
    _im(axes[1,0], diff,  "Diff: B − A  (V)", "seismic", -dabs, dabs)
    _im(axes[1,1], pct,   f"% |diff| / max(|A|, {PCT_FLOOR}V)", "hot_r", 0, None)

    fig.tight_layout()
    path = outdir / f"{prefix}_z{z_nm:.0f}nm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved: {path.name}")

    return {
        "z_nm": z_nm,
        "a_min": float(np.nanmin(phi_a)), "a_max": float(np.nanmax(phi_a)),
        "b_min": float(np.nanmin(phi_b)), "b_max": float(np.nanmax(phi_b)),
        "diff_max_V": float(np.nanmax(np.abs(diff))),
        "diff_rms_V": float(np.sqrt(np.nanmean(diff[mask]**2))) if mask.any() else float("nan"),
        "mean_pct":   float(np.nanmean(pct[mask])) if mask.any() else float("nan"),
    }


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run with plain python3, not mpiexec.")

    ap = argparse.ArgumentParser(description="Compare two background XDMF cases.")
    ap.add_argument("--xdmf-a", required=True, dest="xdmf_a")
    ap.add_argument("--xdmf-b", required=True, dest="xdmf_b")
    ap.add_argument("--label-a", default="Case A", dest="label_a")
    ap.add_argument("--label-b", default="Case B", dest="label_b")
    ap.add_argument("--outdir", default="outputs/background_compare")
    ap.add_argument("--slices-nm", nargs="+", type=float, dest="slices_nm",
                    default=[0, 10, 20, 50, 100, 155, 255])
    ap.add_argument("--grid-n", type=int, default=201, dest="grid_n")
    ap.add_argument("--disk-cx-nm", type=float, default=None, dest="cx_nm",
                    help="Disk centre x (nm); auto-detected if omitted")
    ap.add_argument("--disk-cy-nm", type=float, default=None, dest="cy_nm")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "bg_compare"

    print(f"\n=== Background comparison ===")
    print(f"  A: {args.xdmf_a}  ({args.label_a})")
    print(f"  B: {args.xdmf_b}  ({args.label_b})")

    u_a = load_xdmf_field(Path(args.xdmf_a))
    u_b = load_xdmf_field(Path(args.xdmf_b))

    x_min, x_max, y_min, y_max = mesh_xy_range(u_a)
    cx_nm = args.cx_nm if args.cx_nm is not None else 0.5 * (x_min + x_max)
    cy_nm = args.cy_nm if args.cy_nm is not None else 0.5 * (y_min + y_max)

    rows = []
    for z_nm in sorted(args.slices_nm):
        print(f"\n  z = {z_nm:.0f} nm")
        row = compare_at_z(u_a, u_b, z_nm, args.grid_n,
                           (x_min, x_max), (y_min, y_max),
                           args.label_a, args.label_b, outdir, prefix)
        rows.append(row)

    # Depth profiles
    z_a, phi_a_depth = eval_depth_line(u_a, cx_nm, cy_nm)
    z_b, phi_b_depth = eval_depth_line(u_b, cx_nm, cy_nm)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(z_a, phi_a_depth, label=args.label_a)
    axes[0].plot(z_b, phi_b_depth, "--", label=args.label_b)
    axes[0].set_xlabel("z (nm)"); axes[0].set_ylabel("φ (V)")
    axes[0].set_title("φ at disk centre vs depth")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(z_b, phi_b_depth - phi_a_depth, "k")
    axes[1].set_xlabel("z (nm)"); axes[1].set_ylabel("B − A  (V)")
    axes[1].set_title("Difference vs depth")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{args.label_a}  vs  {args.label_b}  — depth profile", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_depth.png", dpi=150)
    plt.close(fig)

    # Save CSV
    keys = list(rows[0].keys())
    with open(outdir / f"{prefix}_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    # Save JSON summary
    summary = {
        "label_a": args.label_a, "xdmf_a": str(args.xdmf_a),
        "label_b": args.label_b, "xdmf_b": str(args.xdmf_b),
        "grid_n": args.grid_n,
        "slices": rows,
    }
    (outdir / f"{prefix}_summary.json").write_text(
        __import__("json").dumps(summary, indent=2))

    print(f"\nOutputs: {outdir}/")


if __name__ == "__main__":
    main()
