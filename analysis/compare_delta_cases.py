#!/usr/bin/env python3
"""
analysis/compare_delta_cases.py
---------------------------------
Compute and compare gate perturbation deltas across case families.

delta_phi = phi_gate_active - phi_background

This script is explicit about which background is paired with which gate case.
It supports two delta pairs (e.g., positive and signed) on the same plot.

Supported delta pairings (must be stated explicitly via CLI):
  Signed delta:   neg1_neg12  - biased_bg   (same V_bottom=-12, gate vs no-gate)
  Positive delta: pos1_p12    - blank        (V_gate=0 reference; no +12V bg exists)

Run via Docker:
  docker run --rm -it -v "$PWD":/app -w /app dolfinx/dolfinx:nightly \\
    sh -lc 'pip install -q h5py scipy 2>/dev/null; \\
            /dolfinx-env/bin/python3 -u analysis/compare_delta_cases.py \\
              --xdmf-gate  results/neg1_neg12_exact_tet_h6/neg1_neg12_exact_tet_h6.xdmf \\
              --xdmf-bg    results/biased_bg_exact_tet_h6/biased_bg_exact_tet_h6.xdmf \\
              --label-gate "neg1_neg12 gate" \\
              --label-bg   "biased_bg (Vbot=-12, no gate)" \\
              --outdir outputs/delta_neg \\
              --slices-nm 0 10 20 50 100 155'
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

sys.path.insert(0, str(Path(__file__).parent))
from xdmf_utils import (load_xdmf_field, eval_on_grid, mesh_xy_range,
                         eval_depth_line, disk_circle)

PCT_FLOOR = 0.01   # 10 mV floor
R_DISK_NM = 10.0


def compute_delta_slice(u_gate, u_bg, z_nm, grid_n, x_range, y_range):
    xg, yg, phi_gate = eval_on_grid(u_gate, z_nm, grid_n, x_range, y_range)
    xg, yg, phi_bg   = eval_on_grid(u_bg,   z_nm, grid_n, x_range, y_range)
    delta = phi_gate - phi_bg
    return xg, yg, phi_gate, phi_bg, delta


def plot_delta_slice(xg, yg, phi_gate, phi_bg, delta, z_nm,
                     label_gate, label_bg, outdir, prefix, cx_nm, cy_nm):
    dmax = float(np.nanmax(np.abs(delta))) or 1e-6
    vlo = min(np.nanmin(phi_gate), np.nanmin(phi_bg))
    vhi = max(np.nanmax(phi_gate), np.nanmax(phi_bg))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Δφ = gate − background   z = {z_nm:.0f} nm\n"
        f"gate: {label_gate}    bg: {label_bg}",
        fontsize=10
    )

    def _im(ax, data, title, cmap, vmin=None, vmax=None):
        ext = [xg.min(), xg.max(), yg.min(), yg.max()]
        im = ax.imshow(data.T, origin="lower", extent=ext, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (nm)", fontsize=8); ax.set_ylabel("y (nm)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        disk_circle(ax, cx_nm, cy_nm, R_DISK_NM)

    _im(axes[0,0], phi_bg,   f"Background φ (V)\n{label_bg}",   "RdBu_r", vlo, vhi)
    _im(axes[0,1], phi_gate, f"Gate-active φ (V)\n{label_gate}", "RdBu_r", vlo, vhi)
    _im(axes[1,0], delta,    "Δφ = gate − bg  (V)",              "seismic", -dmax, dmax)
    _im(axes[1,1], np.abs(delta)*1e3, "| Δφ | (mV)",            "hot_r", 0, None)

    fig.tight_layout()
    path = outdir / f"{prefix}_z{z_nm:.0f}nm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved: {path.name}")

    mask = np.isfinite(delta)
    return {
        "z_nm": z_nm,
        "delta_max_mV":  float(np.nanmax( delta[mask]) * 1e3) if mask.any() else float("nan"),
        "delta_min_mV":  float(np.nanmin( delta[mask]) * 1e3) if mask.any() else float("nan"),
        "delta_peak_mV": float(np.nanmax(np.abs(delta[mask])) * 1e3) if mask.any() else float("nan"),
        "delta_rms_mV":  float(np.sqrt(np.nanmean(delta[mask]**2)) * 1e3) if mask.any() else float("nan"),
    }


def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run with plain python3, not mpiexec.")

    ap = argparse.ArgumentParser(
        description="Compute delta_phi = gate_active - background and visualise slices.")
    ap.add_argument("--xdmf-gate", required=True, dest="xdmf_gate",
                    help="Gate-active XDMF")
    ap.add_argument("--xdmf-bg",   required=True, dest="xdmf_bg",
                    help="Background XDMF (paired background for this gate case)")
    ap.add_argument("--label-gate", default="gate-active", dest="label_gate")
    ap.add_argument("--label-bg",   default="background",  dest="label_bg")
    ap.add_argument("--outdir", default="outputs/delta")
    ap.add_argument("--slices-nm", nargs="+", type=float, dest="slices_nm",
                    default=[0, 10, 20, 50, 100, 155, 255])
    ap.add_argument("--grid-n", type=int, default=201, dest="grid_n")
    ap.add_argument("--disk-cx-nm", type=float, default=None, dest="cx_nm")
    ap.add_argument("--disk-cy-nm", type=float, default=None, dest="cy_nm")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "delta"

    print(f"\n=== Delta comparison: Δφ = gate − background ===")
    print(f"  gate : {args.xdmf_gate}  ({args.label_gate})")
    print(f"  bg   : {args.xdmf_bg}   ({args.label_bg})")
    print(f"  NOTE : delta pairing is EXPLICIT — caller is responsible for")
    print(f"         choosing a physically meaningful gate/bg pair.")

    u_gate = load_xdmf_field(Path(args.xdmf_gate))
    u_bg   = load_xdmf_field(Path(args.xdmf_bg))

    x_min, x_max, y_min, y_max = mesh_xy_range(u_gate)
    cx_nm = args.cx_nm if args.cx_nm is not None else 0.5*(x_min+x_max)
    cy_nm = args.cy_nm if args.cy_nm is not None else 0.5*(y_min+y_max)

    rows = []
    for z_nm in sorted(args.slices_nm):
        print(f"\n  z = {z_nm:.0f} nm")
        xg, yg, phi_gate, phi_bg, delta = compute_delta_slice(
            u_gate, u_bg, z_nm, args.grid_n, (x_min, x_max), (y_min, y_max))
        row = plot_delta_slice(xg, yg, phi_gate, phi_bg, delta, z_nm,
                               args.label_gate, args.label_bg,
                               outdir, prefix, cx_nm, cy_nm)
        rows.append(row)

    # Depth profile of delta at disk centre
    z_gate, phi_gate_z = eval_depth_line(u_gate, cx_nm, cy_nm)
    z_bg,   phi_bg_z   = eval_depth_line(u_bg,   cx_nm, cy_nm)
    delta_z = phi_gate_z - phi_bg_z

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(z_gate, phi_gate_z, label=args.label_gate)
    axes[0].plot(z_bg,   phi_bg_z,   "--", label=args.label_bg)
    axes[0].set_xlabel("z (nm)"); axes[0].set_ylabel("φ (V)")
    axes[0].set_title("φ at disk centre vs depth"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z_gate, delta_z * 1e3, "k", lw=2)
    axes[1].set_xlabel("z (nm)"); axes[1].set_ylabel("Δφ (mV)")
    axes[1].set_title("Gate perturbation Δφ(z) at disk centre\n(how gate voltage decays into semiconductor)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Δφ depth profile  |  gate: {args.label_gate}  bg: {args.label_bg}", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_depth.png", dpi=150)
    plt.close(fig)

    # Save CSV + JSON
    with open(outdir / f"{prefix}_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {
        "label_gate": args.label_gate, "xdmf_gate": str(args.xdmf_gate),
        "label_bg":   args.label_bg,   "xdmf_bg":   str(args.xdmf_bg),
        "pairing_note": "delta = gate_active - background (explicit pairing by caller)",
        "slices": rows,
    }
    (outdir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs: {outdir}/")


if __name__ == "__main__":
    main()
