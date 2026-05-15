#!/usr/bin/env python3
"""
analysis/plot_sige_disk.py
--------------------------
Compare Dirichlet vs periodic charged-disk results.

Reads from two output directories (each produced by main.py charged_disk_box):
  - z_probe.csv     : z[m], phi[V], eps_r[-], rho[C/m³]
  - stats.json      : charge error, phi min/max, solve time, …

Produces PNG figures in --outdir:
  - stack_profile.png   : phi(z), eps_r(z), rho(z) for both BC modes
  - bc_comparison.png   : phi(z) Dirichlet vs periodic side-by-side
  - summary.txt         : printed charge-error table

Usage (inside Docker or with matplotlib/numpy installed):
  python3 analysis/plot_sige_disk.py \\
      --dir_dirichlet results/sige_dirichlet \\
      --dir_periodic  results/sige_periodic  \\
      --outdir        results/plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works inside Docker)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── colours ────────────────────────────────────────────────────────────────
C_DIR = "#1f77b4"   # blue  – Dirichlet
C_PER = "#d62728"   # red   – periodic
C_SIGE = "#aec7e8"  # light-blue shading for SiGe
C_SI   = "#ffbb78"  # orange shading for Si


def _load(directory):
    d = Path(directory)
    csv  = np.loadtxt(d / "z_probe.csv", delimiter=",", skiprows=1)
    with open(d / "stats.json") as f:
        stats = json.load(f)
    return csv, stats


def _layer_patches(ax, layers, y0, y1):
    """Shade dielectric layer backgrounds."""
    colours = {13.0: C_SIGE, 11.7: C_SI}
    for lyr in layers:
        zmin_nm = lyr["zmin"] * 1e9
        zmax_nm = lyr["zmax"] * 1e9
        c = colours.get(round(lyr["epsr"], 1), "#dddddd")
        ax.axvspan(zmin_nm, zmax_nm, alpha=0.18, color=c, zorder=0)
    # disk position
    disk = None  # filled in by caller
    return disk


def _mark_disk(ax, stats):
    if not stats["disks"]:
        return
    d = stats["disks"][0]
    zc_nm = d["zc"] * 1e9
    R_nm  = d["R"]  * 1e9
    ax.axvline(zc_nm, color="gray", ls="--", lw=0.8, label="disk centre")
    ax.axvspan(zc_nm - R_nm, zc_nm + R_nm, alpha=0.12, color="green", zorder=0)


def plot_stack_profile(csv_dir, csv_per, stats_dir, stats_per, outdir):
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    z_nm = csv_dir[:, 0] * 1e9
    z_nm_p = csv_per[:, 0] * 1e9
    layers = stats_dir["layers"]

    for ax in axes:
        _layer_patches(ax, layers, 0, 1)
        _mark_disk(ax, stats_dir)

    # --- phi ---
    ax = axes[0]
    ax.plot(z_nm,   csv_dir[:, 1] * 1e3, color=C_DIR, lw=1.6, label="Dirichlet")
    ax.plot(z_nm_p, csv_per[:, 1] * 1e3, color=C_PER, lw=1.6, ls="--", label="Periodic xy")
    ax.set_ylabel("φ  (mV)")
    ax.legend(fontsize=8)
    ax.set_title("Electrostatic potential along z-axis  (x=y=0)")

    # --- eps_r ---
    ax = axes[1]
    ax.step(z_nm, csv_dir[:, 2], color=C_DIR, lw=1.4, where="mid", label="Dirichlet")
    ax.set_ylabel("ε_r  (-)")
    ax.set_ylim(0, max(csv_dir[:, 2]) * 1.3)

    # add legend patches for layers
    sige_patch = mpatches.Patch(color=C_SIGE, alpha=0.5, label="SiGe (ε≈13)")
    si_patch   = mpatches.Patch(color=C_SI,   alpha=0.5, label="Si   (ε≈11.7)")
    ax.legend(handles=[sige_patch, si_patch], fontsize=8)
    ax.set_title("Dielectric profile")

    # --- rho ---
    ax = axes[2]
    rho_dir = csv_dir[:, 3]
    mask = rho_dir != 0.0
    if mask.any():
        # show non-zero region bar
        ax.fill_between(
            z_nm, np.where(mask, rho_dir, 0),
            color="green", alpha=0.6, step="mid", label=r"ρ (C/m³)",
        )
    ax.set_ylabel(r"ρ  (C/m³)")
    ax.set_xlabel("z  (nm)")
    ax.legend(fontsize=8)
    ax.set_title("Charge density")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(
        "SiGe / Si / SiGe quantum-dot stack  —  z-axis profiles",
        fontsize=11, y=0.99,
    )

    path = Path(outdir) / "stack_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_bc_comparison(csv_dir, csv_per, stats_dir, stats_per, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, csv, stats, label, colour in [
        (axes[0], csv_dir, stats_dir, "Dirichlet  (φ=0 all faces)", C_DIR),
        (axes[1], csv_per, stats_per, "Periodic  xy  (φ=0 z-faces)", C_PER),
    ]:
        z_nm = csv[:, 0] * 1e9
        phi_mV = csv[:, 1] * 1e3
        layers = stats["layers"]
        _layer_patches(ax, layers, 0, 1)
        _mark_disk(ax, stats)
        ax.plot(z_nm, phi_mV, color=colour, lw=1.8)
        ax.set_xlabel("z  (nm)")
        ax.set_title(label, fontsize=9)

        Q_gauss = stats["Q_gauss_C"]
        Q_target = stats["Q_target_C"]
        err = stats["charge_err_pct"]
        ax.text(
            0.97, 0.97,
            f"Q_target = {Q_target:.3e} C\n"
            f"Q_Gauss  = {Q_gauss:.3e} C\n"
            f"err      = {err:.3f} %",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    axes[0].set_ylabel("φ  (mV)")

    sige_patch = mpatches.Patch(color=C_SIGE, alpha=0.5, label="SiGe")
    si_patch   = mpatches.Patch(color=C_SI,   alpha=0.5, label="Si")
    fig.legend(
        handles=[sige_patch, si_patch],
        loc="lower center", ncol=2, fontsize=8, framealpha=0.9,
    )

    fig.suptitle(
        "Dirichlet vs Periodic BC — φ along z-axis through disk centre",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    path = Path(outdir) / "bc_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(stats_dir, stats_per, outdir):
    lines = [
        "=" * 60,
        "  SiGe / Si / SiGe  —  Charge-error summary",
        "=" * 60,
        f"{'Quantity':<28} {'Dirichlet':>14} {'Periodic xy':>14}",
        "-" * 60,
    ]

    rows = [
        ("Q_target (C)",       f"{stats_dir['Q_target_C']:.4e}",    f"{stats_per['Q_target_C']:.4e}"),
        ("Q_gauss  (C)",       f"{stats_dir['Q_gauss_C']:.4e}",     f"{stats_per['Q_gauss_C']:.4e}"),
        ("charge error (%)",   f"{stats_dir['charge_err_pct']:.4f}", f"{stats_per['charge_err_pct']:.4f}"),
        ("phi_min  (mV)",      f"{stats_dir['phi_min_V']*1e3:.4f}", f"{stats_per['phi_min_V']*1e3:.4f}"),
        ("phi_max  (mV)",      f"{stats_dir['phi_max_V']*1e3:.4f}", f"{stats_per['phi_max_V']*1e3:.4f}"),
        ("solve time (s)",     f"{stats_dir['solve_time_s']:.2f}",   f"{stats_per['solve_time_s']:.2f}"),
        ("n_charged_cells",    str(stats_dir['n_charged_cells']),    str(stats_per['n_charged_cells'])),
        ("h_disk  (nm)",       f"{stats_dir['h_disk_m']*1e9:.2f}",  f"{stats_per['h_disk_m']*1e9:.2f}"),
    ]

    for (label, d, p) in rows:
        lines.append(f"  {label:<26} {d:>14} {p:>14}")

    lines.append("=" * 60)

    txt = "\n".join(lines)
    print(txt)

    path = Path(outdir) / "summary.txt"
    path.write_text(txt + "\n")
    print(f"Saved: {path}")

    # Warn if either run is above 1 %
    for label, stats in [("Dirichlet", stats_dir), ("Periodic", stats_per)]:
        err = stats["charge_err_pct"]
        if err > 1.0:
            print(
                f"\n[WARN] {label} charge error = {err:.3f}% > 1%.  "
                "Increase --n_diam or --refine_band_R."
            )
        else:
            print(f"[OK]   {label} charge error = {err:.3f}% < 1%.")


def main():
    ap = argparse.ArgumentParser(description="Plot and compare SiGe-disk runs.")
    ap.add_argument("--dir_dirichlet", required=True)
    ap.add_argument("--dir_periodic",  required=True)
    ap.add_argument("--outdir", default="results/plots")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"Loading Dirichlet results from: {args.dir_dirichlet}")
    csv_dir, stats_dir = _load(args.dir_dirichlet)

    print(f"Loading Periodic  results from: {args.dir_periodic}")
    csv_per, stats_per = _load(args.dir_periodic)

    plot_stack_profile(csv_dir, csv_per, stats_dir, stats_per, args.outdir)
    plot_bc_comparison(csv_dir, csv_per, stats_dir, stats_per, args.outdir)
    print_summary(stats_dir, stats_per, args.outdir)

    print(f"\nAll plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
