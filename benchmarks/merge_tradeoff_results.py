#!/usr/bin/env python3
"""Merge per-config JSON results from concurrent mesh_refinement_tradeoff.py
--only runs into the same tradeoff_results.csv + report.txt format that a
single sequential (no --only) run produces.

Usage:
    python3 benchmarks/merge_tradeoff_results.py <outdir> [--probe-r-nm 5 10 15]

Expects <outdir>/{uniform_coarse,uniform_fine,quadrant_mixed}_result.json,
written by `mesh_refinement_tradeoff.py --only <name> --outdir <outdir>`.
"""
from __future__ import annotations
import argparse
import csv
import json
import os


ORDER = ["uniform_coarse", "uniform_fine", "quadrant_mixed"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="Directory containing <name>_result.json files")
    args = p.parse_args()

    results = []
    for name in ORDER:
        json_path = os.path.join(args.outdir, f"{name}_result.json")
        if not os.path.exists(json_path):
            raise SystemExit(f"Missing {json_path} -- did all 3 configs finish?")
        with open(json_path) as f:
            results.append(json.load(f))

    csv_path = os.path.join(args.outdir, "tradeoff_results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["name", "h_label", "n_cells", "n_dofs", "t_mesh", "t_solve",
                      "t_total", "max_rel_err", "mean_rel_err"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    report_path = os.path.join(args.outdir, "report.txt")
    lines = []
    lines.append("=" * 78)
    lines.append("MESH-REFINEMENT TRADEOFF -- uniform-coarse vs uniform-fine vs 4-quadrant")
    lines.append("(merged from concurrent --only runs)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"{'config':<16}{'mesh':<24}{'n_cells':>10}{'n_dofs':>10}"
                  f"{'t_mesh[s]':>11}{'t_solve[s]':>12}{'max_relerr':>12}{'mean_relerr':>13}")
    lines.append("-" * 100)
    for r in results:
        lines.append(f"{r['name']:<16}{r['h_label']:<24}{r['n_cells']:>10,}{r['n_dofs']:>10,}"
                      f"{r['t_mesh']:>11.2f}{r['t_solve']:>12.2f}"
                      f"{r['max_rel_err']:>12.3e}{r['mean_rel_err']:>13.3e}")
    lines.append("")
    lines.append("  Raw phi at each probe point (FEM vs analytic), per config:")
    for r in results:
        lines.append(f"  [{r['name']}  ({r['h_label']})]")
        for rr, ph, pe, re in zip(r["probe_r_nm"], r["phi_h"], r["phi_ex"], r["rel_err"]):
            lines.append(f"      r={rr:6.1f}nm   phi_FEM={ph: .6e} V   "
                          f"phi_exact={pe: .6e} V   rel_err={re:.3e}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("TRADEOFF SUMMARY")
    lines.append("-" * 78)
    c, fi, qd = results[0], results[1], results[2]
    speedup_vs_fine = (fi["t_mesh"] + fi["t_solve"]) / max(qd["t_mesh"] + qd["t_solve"], 1e-9)
    dof_ratio = fi["n_dofs"] / max(qd["n_dofs"], 1)
    lines.append(f"  uniform_fine total time   = {fi['t_mesh']+fi['t_solve']:.2f}s "
                 f"({fi['n_dofs']:,} dofs, max_rel_err={fi['max_rel_err']:.3e})")
    lines.append(f"  quadrant_mixed total time = {qd['t_mesh']+qd['t_solve']:.2f}s "
                 f"({qd['n_dofs']:,} dofs, max_rel_err={qd['max_rel_err']:.3e})")
    lines.append(f"  uniform_coarse total time = {c['t_mesh']+c['t_solve']:.2f}s "
                 f"({c['n_dofs']:,} dofs, max_rel_err={c['max_rel_err']:.3e})")
    lines.append(f"  -> quadrant_mixed uses {dof_ratio:.1f}x fewer DOFs than uniform_fine, "
                 f"runs {speedup_vs_fine:.1f}x {'faster' if speedup_vs_fine > 1 else 'slower'}, "
                 f"for {'comparable' if abs(qd['max_rel_err']-fi['max_rel_err']) < 0.1*fi['max_rel_err'] else 'different'} "
                 f"near-charge accuracy.")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nMerged report written to {report_path}")
    print(f"Merged CSV written to {csv_path}")


if __name__ == "__main__":
    main()
