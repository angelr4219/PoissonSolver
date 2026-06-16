#!/usr/bin/env python3
"""FFT-leg mesh-density sweep, runnable WITHOUT Docker/dolfinx.

fft_solve.py and analytics.py are pure NumPy, so this script loads them
directly (bypassing src/poisson/__init__.py, which eagerly imports the
gmsh-dependent modules) and runs only the FFT comparison. Useful for a
quick check of the FFT leg, or in any environment without a working
dolfinx Docker image, before committing to a full FEM run.

Usage:
    python3 benchmarks/fft_only_sweep.py [--L-nm 500] [--R-nm 50] [--V0 1.0]
                                          [--fft-max-n 300] [--outdir results/sphere_triple]
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src" / "poisson"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fft_solve = _load("fft_solve", "fft_solve.py")
analytics = _load("analytics", "analytics.py")

H_ALL_NM = [20.0, 10.0, 5.0, 1.0, 0.1]


def probe_points(probe_r_nm):
    pts = [[r := r_nm * 1e-9 / np.sqrt(3.0), r, r] for r_nm in probe_r_nm]
    return np.array(pts, dtype=float)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L-nm", type=float, default=500.0)
    p.add_argument("--R-nm", type=float, default=50.0)
    p.add_argument("--V0", type=float, default=1.0)
    p.add_argument("--fft-max-n", type=int, default=300)
    p.add_argument("--outdir", default="results/sphere_triple")
    return p


def main():
    args = build_parser().parse_args()
    EPS0 = fft_solve.EPS0
    L_m = args.L_nm * 1e-9
    R_m = args.R_nm * 1e-9
    V0 = args.V0
    q = 4.0 * np.pi * EPS0 * V0 * R_m

    pts = probe_points([100, 150, 200])
    r_pts = np.linalg.norm(pts, axis=1)
    phi_exact = analytics.phi_conducting_sphere(r_pts, V0, R_m)

    if max(100, 150, 200) > 0.3 * args.L_nm:
        print(f"WARNING: probe radii approach L/2={args.L_nm/2:.0f} nm -- "
              f"periodic-image contamination will dominate the error budget, "
              f"independent of mesh density. See SUGGESTIONS.txt item 0.")

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    header = f"{'h[nm]':>7} {'N':>6} {'N^3':>12} {'t_grid[s]':>10} {'t_solve[s]':>11} {'max_rel_err':>13} {'mean_rel_err':>13}"
    print(header)
    print("-" * len(header))

    for h_nm in H_ALL_NM:
        N = int(round(args.L_nm / h_nm))
        if N > args.fft_max_n:
            print(f"{h_nm:7.1f} {N:6d} {'-':>12} {'-':>10} {'-':>11} "
                  f"{'SKIPPED N>'+str(args.fft_max_n):>13}")
            rows.append(dict(h_nm=h_nm, N=N, t_grid_s=None, t_solve_s=None,
                              max_rel_err=None, mean_rel_err=None,
                              note=f"N={N} exceeds FFT_MAX_N={args.fft_max_n}"))
            continue

        sigma_shell = max(R_m / 8.0, 1.5 * (L_m / N))

        t0 = time.perf_counter()
        rho = fft_solve.sphere_charge_shell_grid(N, L_m, q, R_m, sigma_shell)
        t_grid = time.perf_counter() - t0

        t1 = time.perf_counter()
        phi = fft_solve.fft_poisson_3d(rho, L_m, eps_r=1.0)
        t_solve = time.perf_counter() - t1

        phi_h = fft_solve.evaluate_fft_phi_at_points(phi, L_m, pts)
        rel_errs = np.abs(phi_h - phi_exact) / np.abs(phi_exact)
        max_err = float(np.max(rel_errs))
        mean_err = float(np.mean(rel_errs))

        print(f"{h_nm:7.1f} {N:6d} {N**3:12,} {t_grid:10.3f} {t_solve:11.3f} "
              f"{max_err:13.3e} {mean_err:13.3e}")
        rows.append(dict(h_nm=h_nm, N=N, t_grid_s=round(t_grid, 3),
                          t_solve_s=round(t_solve, 3), max_rel_err=max_err,
                          mean_rel_err=mean_err, note=""))

    csv_path = os.path.join(args.outdir, "fft_only_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["h_nm", "N", "t_grid_s", "t_solve_s",
                                                 "max_rel_err", "mean_rel_err", "note"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten to {csv_path}")


if __name__ == "__main__":
    main()
