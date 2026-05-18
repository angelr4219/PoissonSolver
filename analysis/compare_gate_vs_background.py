#!/usr/bin/env python3
"""
analysis/compare_gate_vs_background.py
----------------------------------------
Full 4-panel comparison of a gate-active case against its paired background.
Produces both absolute phi plots and the delta (gate perturbation).

Designed for the Stage 2 validation: once MaSQE's totalPotential3d.vtk is
available, the DOLFINx delta from this script can be compared against
  delta_masque = totalPotential - basePotential

For now (without totalPotential3d.vtk) this script shows the DOLFINx gate
perturbation in isolation.

Run via Docker (signed convention):
  docker run --rm -it -v "$PWD":/app -w /app dolfinx/dolfinx:nightly \\
    sh -lc 'pip install -q h5py scipy 2>/dev/null; \\
            /dolfinx-env/bin/python3 -u analysis/compare_gate_vs_background.py \\
              --xdmf-gate results/neg1_neg12_exact_tet_h6/neg1_neg12_exact_tet_h6.xdmf \\
              --xdmf-bg   results/biased_bg_exact_tet_h6/biased_bg_exact_tet_h6.xdmf \\
              --label-gate "neg1 gate" --label-bg "biased_bg" \\
              --outdir outputs/gate_vs_bg_neg'
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
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent))
from xdmf_utils import (load_xdmf_field, eval_on_grid, mesh_xy_range,
                         eval_depth_line, disk_circle)

PCT_FLOOR = 0.01
R_DISK_NM = 10.0

QW_LO_NM, QW_HI_NM = 50.0, 55.0   # sSi quantum well — band-offset zone


def compare_slice(u_gate, u_bg, z_nm, grid_n, x_range, y_range,
                  label_gate, label_bg, outdir, prefix, cx_nm, cy_nm):
    xg, yg, phi_gate = eval_on_grid(u_gate, z_nm, grid_n, x_range, y_range)
    xg, yg, phi_bg   = eval_on_grid(u_bg,   z_nm, grid_n, x_range, y_range)

    delta = phi_gate - phi_bg
    mask  = np.isfinite(phi_gate) & np.isfinite(phi_bg)
    dmax  = float(np.nanmax(np.abs(delta))) or 1e-6
    vlo   = min(np.nanmin(phi_gate), np.nanmin(phi_bg))
    vhi   = max(np.nanmax(phi_gate), np.nanmax(phi_bg))

    # % error on delta vs background
    denom = np.maximum(np.abs(phi_bg), PCT_FLOOR)
    pct_delta = np.where(mask, 100.0 * np.abs(delta) / denom, np.nan)

    qw_flag = QW_LO_NM <= z_nm <= QW_HI_NM
    title_suffix = "  ⚠ QW band-offset zone" if qw_flag else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Gate vs Background   z = {z_nm:.0f} nm{title_suffix}\n"
        f"gate: {label_gate}   bg: {label_bg}",
        fontsize=10, color="darkorange" if qw_flag else "black"
    )

    def _im(ax, data, title, cmap, vmin=None, vmax=None):
        ext = [xg.min(), xg.max(), yg.min(), yg.max()]
        im = ax.imshow(data.T, origin="lower", extent=ext, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (nm)", fontsize=8); ax.set_ylabel("y (nm)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        disk_circle(ax, cx_nm, cy_nm, R_DISK_NM)
        if qw_flag:
            ax.text(0.5, 0.97, "QW zone", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color="darkorange",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    _im(axes[0,0], phi_bg,   f"Background φ (V)\n{label_bg}",   "RdBu_r", vlo, vhi)
    _im(axes[0,1], phi_gate, f"Gate-active φ (V)\n{label_gate}", "RdBu_r", vlo, vhi)
    _im(axes[1,0], delta,    "Δφ = gate − bg  (V)",              "seismic", -dmax, dmax)
    _im(axes[1,1], pct_delta, "% |Δφ| / max(|bg|, 10mV)",       "hot_r", 0, None)

    fig.tight_layout()
    path = outdir / f"{prefix}_z{z_nm:.0f}nm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved: {path.name}")

    peak_mV  = float(np.nanmax(np.abs(delta[mask])) * 1e3) if mask.any() else float("nan")
    mean_pct = float(np.nanmean(pct_delta[mask]))          if mask.any() else float("nan")
    return {
        "z_nm": z_nm, "qw_zone": qw_flag,
        "gate_min_V": float(np.nanmin(phi_gate)),
        "gate_max_V": float(np.nanmax(phi_gate)),
        "bg_min_V":   float(np.nanmin(phi_bg)),
        "bg_max_V":   float(np.nanmax(phi_bg)),
        "delta_peak_mV": peak_mV,
        "mean_pct_delta": mean_pct,
    }


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run with plain python3, not mpiexec.")

    ap = argparse.ArgumentParser(
        description="Full gate vs background comparison with delta maps.")
    ap.add_argument("--xdmf-gate", required=True, dest="xdmf_gate")
    ap.add_argument("--xdmf-bg",   required=True, dest="xdmf_bg")
    ap.add_argument("--label-gate", default="gate-active", dest="label_gate")
    ap.add_argument("--label-bg",   default="background",  dest="label_bg")
    ap.add_argument("--outdir", default="outputs/gate_vs_bg")
    ap.add_argument("--slices-nm", nargs="+", type=float, dest="slices_nm",
                    default=[0, 10, 20, 50, 55, 100, 155, 255])
    ap.add_argument("--grid-n", type=int, default=201, dest="grid_n")
    ap.add_argument("--disk-cx-nm", type=float, default=None, dest="cx_nm")
    ap.add_argument("--disk-cy-nm", type=float, default=None, dest="cy_nm")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "gvb"

    print(f"\n=== Gate vs Background ===")
    print(f"  gate : {args.xdmf_gate}  ({args.label_gate})")
    print(f"  bg   : {args.xdmf_bg}   ({args.label_bg})")
    print(f"  QW band-offset zone z={QW_LO_NM:.0f}–{QW_HI_NM:.0f} nm flagged on plots")

    u_gate = load_xdmf_field(Path(args.xdmf_gate))
    u_bg   = load_xdmf_field(Path(args.xdmf_bg))

    x_min, x_max, y_min, y_max = mesh_xy_range(u_gate)
    cx_nm = args.cx_nm if args.cx_nm is not None else 0.5*(x_min+x_max)
    cy_nm = args.cy_nm if args.cy_nm is not None else 0.5*(y_min+y_max)

    rows = []
    for z_nm in sorted(args.slices_nm):
        print(f"\n  z = {z_nm:.0f} nm")
        row = compare_slice(u_gate, u_bg, z_nm, args.grid_n,
                            (x_min, x_max), (y_min, y_max),
                            args.label_gate, args.label_bg,
                            outdir, prefix, cx_nm, cy_nm)
        rows.append(row)

    # Depth profiles
    z_g, phi_g_z = eval_depth_line(u_gate, cx_nm, cy_nm)
    z_b, phi_b_z = eval_depth_line(u_bg,   cx_nm, cy_nm)
    delta_z = phi_g_z - phi_b_z

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(z_g, phi_g_z, label=args.label_gate)
    axes[0].plot(z_b, phi_b_z, "--", label=args.label_bg)
    axes[0].axvspan(QW_LO_NM, QW_HI_NM, alpha=0.15, color="orange", label="QW zone")
    axes[0].set_xlabel("z (nm)"); axes[0].set_ylabel("φ (V)")
    axes[0].set_title("φ at disk centre vs depth")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(z_g, delta_z*1e3, "k", lw=2)
    axes[1].axvspan(QW_LO_NM, QW_HI_NM, alpha=0.15, color="orange")
    axes[1].set_xlabel("z (nm)"); axes[1].set_ylabel("Δφ (mV)")
    axes[1].set_title("Gate perturbation Δφ(z)\n(decays through semiconductor stack)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(z_g, np.abs(delta_z)*1e3, "b", lw=1.5, label="|Δφ| (mV)")
    axes[2].set_xlabel("z (nm)"); axes[2].set_ylabel("|Δφ| (mV)")
    axes[2].set_yscale("log")
    axes[2].set_title("|Δφ| log scale — decay rate")
    axes[2].grid(True, alpha=0.3, which="both")

    fig.suptitle(f"gate: {args.label_gate}  |  bg: {args.label_bg}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_depth.png", dpi=150)
    plt.close(fig)

    # Verdict summary
    non_qw = [r for r in rows if not r["qw_zone"]]
    if non_qw:
        peak_depths = [(r["z_nm"], r["delta_peak_mV"]) for r in non_qw]
        print(f"\n  Gate perturbation peak Δφ at disk centre:")
        for z, pk in peak_depths:
            print(f"    z={z:5.0f} nm : {pk:.3f} mV")

    # CSV
    with open(outdir / f"{prefix}_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {
        "label_gate": args.label_gate, "xdmf_gate": str(args.xdmf_gate),
        "label_bg":   args.label_bg,   "xdmf_bg":   str(args.xdmf_bg),
        "qw_zone_nm": [QW_LO_NM, QW_HI_NM],
        "slices": rows,
    }
    (outdir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs: {outdir}/")


if __name__ == "__main__":
    main()
