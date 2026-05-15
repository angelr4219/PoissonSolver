#!/usr/bin/env python3
"""
analysis/run_validation_ladder.py
----------------------------------
Multi-case VTK vs XDMF validation ladder for the disk-gate benchmark.

Loops over a defined set of cases, runs the compare_vtk_xdmf_disk10nm logic
for each one, and writes a master summary CSV + JSON to outputs/.

Run inside Docker via run_validation.sh:
  ./run_validation.sh

Or directly from inside a Docker shell:
  /dolfinx-env/bin/python3 -u analysis/run_validation_ladder.py \
      --vtk basePotential3d.vtk \
      --outdir outputs/disk10nm_comparison

The script searches for XDMF files in results/<case_name>/<case_name>.xdmf.
Cases that do not yet exist are skipped with a warning.

Acceptance criterion (per spec):
  z=155 nm mean percent error < 1.0%  →  ACCEPTED
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir so we can import compare_vtk_xdmf_disk10nm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_vtk_xdmf_disk10nm as cmp

from mpi4py import MPI


# ── Case registry ─────────────────────────────────────────────────────────────

CASES = [
    # label                                     celltype  h_nm  Lx_nm  notes
    ("basepot_R10_exact_tet_h6_neg1_neg12",     "tet",    6.0,  500,   "Case A: exact-box tet h=6"),
    ("basepot_R10_exact_hex_h6_neg1_neg12",     "hex",    6.0,  500,   "Case B: exact-box hex h=6"),
    ("basepot_R10_exact_tet_h3_neg1_neg12",     "tet",    3.0,  500,   "Case C: exact-box tet h=3"),
    ("basepot_R10_exact_hex_h3_neg1_neg12",     "hex",    3.0,  500,   "Case D: exact-box hex h=3"),
    ("basepot_R10_L800_tet_h6_neg1_neg12",      "tet",    6.0,  800,   "L800 tet h=6"),
    ("basepot_R10_L800_hex_h6_neg1_neg12",      "hex",    6.0,  800,   "L800 hex h=6"),
]

# z=5 nm has no exact VTK point (nearest is z=0), so use z=51 (first sSi point) instead.
# VTK z array has exact points at: 0, 10, 20, 30, 40, 50, 51, 52, 53, 54, 55, 57, ... 255
SLICES_NM = [0.0, 10.0, 20.0, 51.0, 155.0, 255.0]
ACCEPT_Z_NM = 155.0
ACCEPT_THRESHOLD_PCT = 1.0   # mean percent error must be below this

PHI_VMIN = -12.0   # shared color scale for all phi plots
PHI_VMAX = -1.0


# ── Per-case comparison ────────────────────────────────────────────────────────

def run_case(
    case_name: str,
    case_notes: str,
    vtk_x, vtk_y, vtk_z, vtk_phi,
    results_root: Path,
    outdir: Path,
    grid_n: int,
    percent_floor: float,
    solver_scale: float,
    save_npy: bool,
) -> dict | None:
    """
    Run the full comparison for one XDMF case.
    Returns a dict with all slice metrics, or None if the case files are missing.
    """
    xdmf_path = results_root / case_name / f"{case_name}.xdmf"
    h5_path   = results_root / case_name / f"{case_name}.h5"

    if not xdmf_path.exists():
        print(f"  [SKIP] {case_name}: XDMF not found at {xdmf_path}")
        return None
    if not h5_path.exists():
        print(f"  [SKIP] {case_name}: H5 not found at {h5_path}")
        return None

    case_outdir = outdir / case_name
    case_outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  CASE: {case_name}")
    print(f"  Notes: {case_notes}")
    print(f"{'='*70}")

    try:
        msh, V, u, h5_used, ds_used = cmp.load_xdmf_function(
            xdmf_path, h5_path, "phi", None
        )
        cmp.print_xdmf_info(msh, V, u, xdmf_path, h5_used, ds_used, solver_scale)
    except Exception as exc:
        print(f"  [ERROR] loading XDMF for {case_name}: {exc}")
        return None

    slice_rows = []
    for z_nm in sorted(SLICES_NM):
        result = cmp.compare_slice(
            z_nm=z_nm,
            vtk_x=vtk_x, vtk_y=vtk_y, vtk_z=vtk_z, vtk_phi=vtk_phi,
            u=u,
            solver_scale=solver_scale,
            grid_n=grid_n,
            percent_floor=percent_floor,
            outdir=case_outdir,
            basename=case_name,
            save_npy=save_npy,
        )
        if result is not None:
            result["case_name"] = case_name
            result["case_notes"] = case_notes
            slice_rows.append(result)

    cmp.write_summary_csv(slice_rows, case_outdir, case_name)
    cmp.write_summary_json(slice_rows, case_outdir, case_name)

    cmp.print_final_table(slice_rows)

    z155_rows = [r for r in slice_rows if abs(r.get("z_requested_nm", 999) - ACCEPT_Z_NM) < 0.1]
    accepted = False
    z155_mean_pct = float("nan")
    if z155_rows:
        z155_mean_pct = z155_rows[0].get("mean_pct_err", float("nan"))
        accepted = z155_mean_pct < ACCEPT_THRESHOLD_PCT
        status = "ACCEPTED ✓" if accepted else "NOT ACCEPTED ✗"
        print(f"\n  z={ACCEPT_Z_NM:.0f} nm mean % error = {z155_mean_pct:.4f}%  →  {status}")

    return {
        "case_name":       case_name,
        "case_notes":      case_notes,
        "n_slices":        len(slice_rows),
        "z155_mean_pct":   z155_mean_pct,
        "accepted":        accepted,
        "slices":          slice_rows,
    }


# ── Master summary plots ───────────────────────────────────────────────────────

def plot_z155_bar(case_results: list[dict], outdir: Path) -> None:
    labels   = [r["case_name"].replace("basepot_R10_", "").replace("_neg1_neg12", "") for r in case_results]
    values   = [r["z155_mean_pct"] for r in case_results]
    accepted = [r["accepted"] for r in case_results]
    colors   = ["green" if a else "red" for a in accepted]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(ACCEPT_THRESHOLD_PCT, color="blue", linestyle="--", linewidth=1.2,
               label=f"Target {ACCEPT_THRESHOLD_PCT:.0f}%")
    ax.set_ylabel("Mean % error at z=155 nm")
    ax.set_title("Validation ladder — z=155 nm acceptance test\n(green = accepted, red = not accepted)")
    ax.set_ylim(0, max(max(values, default=1) * 1.4, ACCEPT_THRESHOLD_PCT * 2))
    ax.legend()
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    path = outdir / "z155_acceptance_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved bar chart: {path}")


def plot_error_vs_depth(case_results: list[dict], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, cr in enumerate(case_results):
        rows = sorted(cr["slices"], key=lambda r: r.get("z_requested_nm", 0))
        zs   = [r["z_requested_nm"] for r in rows]
        errs = [r.get("mean_pct_err", float("nan")) for r in rows]
        label = cr["case_name"].replace("basepot_R10_", "").replace("_neg1_neg12", "")
        ax.plot(zs, errs, marker=markers[i % len(markers)], label=label, linewidth=1.5)

    ax.axhline(ACCEPT_THRESHOLD_PCT, color="blue", linestyle="--", linewidth=1,
               label=f"Target {ACCEPT_THRESHOLD_PCT:.0f}%")
    ax.set_xlabel("z depth (nm)")
    ax.set_ylabel("Mean % error")
    ax.set_title("Mean % error vs z depth — all cases\n(V_gate=-1 V, V_bottom=-12 V, R=10 nm)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-5, 260)
    ax.set_ylim(0, None)
    fig.tight_layout()
    path = outdir / "error_vs_depth_all_cases.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved depth-error plot: {path}")


# ── Master CSV / JSON ──────────────────────────────────────────────────────────

def write_master_summary(case_results: list[dict], outdir: Path) -> None:
    master_rows = []
    for cr in case_results:
        for sr in cr["slices"]:
            row = {"case_name": cr["case_name"], "accepted_155nm": cr["accepted"]}
            row.update(sr)
            row.pop("case_name", None)
            master_rows.append(row)

    if master_rows:
        path = outdir / "master_comparison_summary.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
            w.writeheader()
            w.writerows(master_rows)
        print(f"  Master CSV: {path}")

    path_j = outdir / "master_comparison_summary.json"
    with open(path_j, "w", encoding="utf-8") as f:
        json.dump(case_results, f, indent=2, default=str)
    print(f"  Master JSON: {path_j}")


def write_master_report(case_results: list[dict], outdir: Path, vtk_path: Path) -> None:
    lines = [
        "Validation Ladder Report",
        "=" * 60,
        f"VTK reference : {vtk_path}",
        f"Date          : see file mtime",
        f"Acceptance    : z={ACCEPT_Z_NM:.0f} nm mean % error < {ACCEPT_THRESHOLD_PCT:.1f}%",
        "",
        f"{'Case':<45} {'z155_mean_%':>12} {'Status':>12}",
        "-" * 72,
    ]
    for cr in case_results:
        v = cr["z155_mean_pct"]
        s = "ACCEPTED" if cr["accepted"] else "NOT ACCEPTED"
        lines.append(f"  {cr['case_name']:<43} {v:>12.4f} {s:>12}")
    lines.append("=" * 72)

    # Per-case per-slice table
    lines.append("")
    lines.append("Per-slice error table")
    lines.append("-" * 72)
    for cr in case_results:
        lines.append(f"\n  {cr['case_name']}")
        lines.append(f"  {'z_req':>8}  {'z_act':>9}  {'coverage':>9}  {'MAE (V)':>12}  {'mean_%':>8}")
        for sr in sorted(cr["slices"], key=lambda r: r.get("z_requested_nm", 0)):
            lines.append(
                f"    {sr.get('z_requested_nm',0):>6.1f}    "
                f"{sr.get('z_actual_nm',0):>8.4f}  "
                f"{sr.get('coverage',0):>8.3%}  "
                f"{sr.get('mae_V', float('nan')):>12.4e}  "
                f"{sr.get('mean_pct_err', float('nan')):>8.3f}"
            )

    path = outdir / "validation_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {path}")

    # Recommended next step
    not_accepted = [cr for cr in case_results if not cr["accepted"]]
    accepted = [cr for cr in case_results if cr["accepted"]]
    lines2 = ["\nNext-step recommendation:"]
    if accepted:
        lines2.append(f"  ACCEPTED cases: {[c['case_name'] for c in accepted]}")
    if not_accepted:
        best = min(not_accepted, key=lambda c: c["z155_mean_pct"])
        lines2.append(f"  Best not-accepted: {best['case_name']} ({best['z155_mean_pct']:.4f}%)")
        lines2.append("  Suggested next experiment (change ONE variable):")
        lines2.append("    → Decrease h_nm from current best to next refinement level")
        lines2.append("      OR investigate top-surface BC mismatch (MaSQE gate representation)")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines2) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Multi-case VTK vs XDMF validation ladder for disk-gate benchmark."
    )
    ap.add_argument("--vtk", default="basePotential3d.vtk",
                    help="ASCII VTK RectilinearGrid reference (default: basePotential3d.vtk)")
    ap.add_argument("--vtk-field", default=None, dest="vtk_field",
                    help="SCALARS field name in VTK (auto-detected if omitted)")
    ap.add_argument("--results-root", default="results", dest="results_root",
                    help="Root dir containing per-case subdirs (default: results/)")
    ap.add_argument("--outdir", default="outputs/disk10nm_comparison",
                    help="Output directory for plots and CSVs")
    ap.add_argument("--grid-n", type=int, default=201, dest="grid_n")
    ap.add_argument("--solver-scale", type=float, default=1e9, dest="solver_scale")
    ap.add_argument("--percent-floor", type=float, default=1e-12, dest="percent_floor")
    ap.add_argument("--save-npy", action="store_true", dest="save_npy")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Limit to specific case names (default: all registered)")
    return ap.parse_args()


def main() -> None:
    cmp._require_serial(MPI.COMM_WORLD)
    args = parse_args()

    vtk_path     = Path(args.vtk)
    results_root = Path(args.results_root)
    outdir       = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not vtk_path.exists():
        print(f"ERROR: VTK reference not found: {vtk_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading VTK reference: {vtk_path}")
    vtk_x, vtk_y, vtk_z, vtk_phi, vtk_field_name = cmp.load_vtk(vtk_path, args.vtk_field)
    cmp.print_vtk_info(vtk_x, vtk_y, vtk_z, vtk_phi, vtk_field_name, vtk_path)

    # Filter cases if --cases is specified
    active_cases = CASES
    if args.cases:
        active_cases = [c for c in CASES if c[0] in args.cases]
        print(f"\nRunning subset: {[c[0] for c in active_cases]}")

    case_results = []
    for case_name, celltype, h_nm, Lx_nm, notes in active_cases:
        cr = run_case(
            case_name=case_name,
            case_notes=notes,
            vtk_x=vtk_x, vtk_y=vtk_y, vtk_z=vtk_z, vtk_phi=vtk_phi,
            results_root=results_root,
            outdir=outdir,
            grid_n=args.grid_n,
            percent_floor=args.percent_floor,
            solver_scale=args.solver_scale,
            save_npy=args.save_npy,
        )
        if cr is not None:
            case_results.append(cr)

    if not case_results:
        print("\nNo cases were processed. Check --results-root and case names.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("  MASTER SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Case':<45} {'z155_%':>8}  {'Status'}")
    print(f"  {'-'*65}")
    for cr in case_results:
        s = "ACCEPTED ✓" if cr["accepted"] else "NOT ACCEPTED ✗"
        print(f"  {cr['case_name']:<45} {cr['z155_mean_pct']:>8.4f}%  {s}")

    write_master_summary(case_results, outdir)
    write_master_report(case_results, outdir, vtk_path)
    plot_z155_bar(case_results, outdir)
    plot_error_vs_depth(case_results, outdir)

    n_accepted = sum(1 for cr in case_results if cr["accepted"])
    print(f"\n  {n_accepted}/{len(case_results)} cases ACCEPTED (z=155 nm mean % < {ACCEPT_THRESHOLD_PCT:.1f}%)")
    print(f"\n  All outputs written to: {outdir}/\n")


if __name__ == "__main__":
    main()
