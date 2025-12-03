#!/usr/bin/env python3
"""
Compute absolute and relative error for Griffiths 2.7 spherical shell data.

Input:
  CSV from griffiths_2p7_sphere.py with columns:
    z [m], E_z [V/m], phi [V]

We reconstruct the analytic E_z(z) and phi(z) using R and sigma, then compute:

  - abs error: |num - exact|
  - rel error: |num - exact| / |exact| (where |exact| > tol)

We treat regions where the exact value is zero separately (absolute error only).
"""

import argparse
from pathlib import Path

import numpy as np

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


def parse_args():
    p = argparse.ArgumentParser(description="Error analysis for Griffiths 2.7 spherical shell")
    p.add_argument("--csv", type=str, required=True,
                   help="Path to griffiths_2p7_sphere.csv")
    p.add_argument("--R", type=float, required=True,
                   help="Sphere radius R [m]")
    p.add_argument("--sigma", type=float, required=True,
                   help="Surface charge density sigma [C/m^2]")
    p.add_argument("--tol", type=float, default=1e-12,
                   help="Tolerance for treating exact values as zero")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    z = data[:, 0]
    Ez_num = data[:, 1]
    phi_num = data[:, 2]

    R = args.R
    sigma = args.sigma
    tol = args.tol

    # Total charge and prefactor
    q = sigma * 4.0 * np.pi * R**2
    prefac = 1.0 / (4.0 * np.pi * EPS0)

    r = np.abs(z)

    # ---- Exact E_z(z) along axis ----
    Ez_exact = np.zeros_like(z)
    mask_out = r > R
    # Avoid division by zero at r=0 using mask
    Ez_exact[mask_out] = prefac * q / (r[mask_out]**2) * np.sign(z[mask_out])

    # ---- Exact phi(z) along axis ----
    phi_exact = np.zeros_like(z)
    mask_in = r < R
    phi_exact[mask_in] = prefac * q / R
    phi_exact[mask_out] = prefac * q / r[mask_out]

    # ---- Errors for E_z ----
    E_err_abs = np.abs(Ez_num - Ez_exact)

    # Where exact E is effectively zero (interior), report absolute error only
    mask_E_exact_zero = np.abs(Ez_exact) < tol
    mask_E_exact_nonzero = ~mask_E_exact_zero

    E_rel = np.zeros_like(E_err_abs)
    E_rel[mask_E_exact_nonzero] = (
        E_err_abs[mask_E_exact_nonzero] / np.abs(Ez_exact[mask_E_exact_nonzero])
    )

    # ---- Errors for phi ----
    phi_err_abs = np.abs(phi_num - phi_exact)
    mask_phi_exact_zero = np.abs(phi_exact) < tol
    mask_phi_exact_nonzero = ~mask_phi_exact_zero

    phi_rel = np.zeros_like(phi_err_abs)
    phi_rel[mask_phi_exact_nonzero] = (
        phi_err_abs[mask_phi_exact_nonzero] / np.abs(phi_exact[mask_phi_exact_nonzero])
    )

    # ---- Summary stats ----
    print("=== Griffiths 2.7 error analysis ===")
    print(f"CSV   : {csv_path}")
    print(f"R     : {R:.3e} m")
    print(f"sigma : {sigma:.3e} C/m^2")
    print(f"q     : {q:.6e} C")
    print(f"tol   : {tol:.1e}")

    # E-field stats
    print("\n--- E_z error ---")
    print(f"max |E_num - E_exact| (all points) = {E_err_abs.max():.3e} V/m")

    if np.any(mask_E_exact_nonzero):
        E_rel_nonzero = E_rel[mask_E_exact_nonzero]
        print(f"max relative error (where |E_exact| > tol): {E_rel_nonzero.max():.3e}")
        print(f"mean relative error: {E_rel_nonzero.mean():.3e}")
    else:
        print("No points with |E_exact| > tol for relative error.")

    if np.any(mask_E_exact_zero):
        E_err_inside = E_err_abs[mask_E_exact_zero]
        print(f"max |E| in region where exact E = 0 (|r|<R): {E_err_inside.max():.3e} V/m")

    # phi stats
    print("\n--- phi error ---")
    print(f"max |phi_num - phi_exact| (all points) = {phi_err_abs.max():.3e} V")

    if np.any(mask_phi_exact_nonzero):
        phi_rel_nonzero = phi_rel[mask_phi_exact_nonzero]
        print(f"max relative error (where |phi_exact| > tol): {phi_rel_nonzero.max():.3e}")
        print(f"mean relative error: {phi_rel_nonzero.mean():.3e}")
    else:
        print("No points with |phi_exact| > tol for relative error.")

    if np.any(mask_phi_exact_zero):
        phi_err_zero = phi_err_abs[mask_phi_exact_zero]
        print(f"max |phi| in region where exact phi ≈ 0: {phi_err_zero.max():.3e} V")


if __name__ == "__main__":
    main()
