#!/usr/bin/env python3
"""
Griffiths Problem 2.7: Field on the axis of a uniformly charged spherical shell.

Given:
  - Spherical surface of radius R
  - Uniform surface charge density sigma
  - Total charge q = sigma * 4*pi*R^2

We compute the electric field and potential along the z-axis, a distance z
from the center, for both:
  - inside (|z| < R)
  - outside (|z| > R)

Results:
  - E_z(z): axial electric field
  - phi(z): potential (with phi -> 0 as z -> infinity)

Usage example:

  python PC/verify/griffiths_2p7_sphere.py \
    --R 1e-7 \
    --sigma 1e-3 \
    --zmin -3e-7 \
    --zmax  3e-7 \
    --n 501 \
    --outdir results/griffiths_2p7_sphere
"""

import argparse
from pathlib import Path

import numpy as np

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


def parse_args():
    p = argparse.ArgumentParser(description="Griffiths 2.7 spherical shell E(z), phi(z)")
    p.add_argument("--R", type=float, default=1e-7,
                   help="Sphere radius R [m]")
    p.add_argument("--sigma", type=float, default=1e-3,
                   help="Surface charge density sigma [C/m^2]")
    p.add_argument("--zmin", type=float, default=-3e-7,
                   help="Minimum z [m]")
    p.add_argument("--zmax", type=float, default=+3e-7,
                   help="Maximum z [m]")
    p.add_argument("--n", type=int, default=501,
                   help="Number of sample points along z")
    p.add_argument("--outdir", type=str, default="results/griffiths_2p7_sphere",
                   help="Output directory for CSV")

    return p.parse_args()


def main():
    args = parse_args()

    R = args.R
    sigma = args.sigma
    zmin = args.zmin
    zmax = args.zmax
    n = args.n

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Total charge on the shell
    q = sigma * 4.0 * np.pi * R**2

    print("=== Griffiths 2.7: spherical shell ===")
    print(f"R      = {R:.3e} m")
    print(f"sigma  = {sigma:.3e} C/m^2")
    print(f"q      = {q:.6e} C")
    print(f"z in   = [{zmin:.3e}, {zmax:.3e}] m, n = {n}")

    # Sample z points
    z = np.linspace(zmin, zmax, n)
    r = np.abs(z)

    # Electric field E_z(z)
    # Inside (r < R): E = 0
    # Outside (r > R): E = (1/(4*pi*eps0)) * q / r^2 * sign(z)
    E_z = np.zeros_like(z)

    mask_out = r > R
    # Avoid divide-by-zero by restricting to mask_out
    E_z[mask_out] = (1.0 / (4.0 * np.pi * EPS0)) * q / (r[mask_out]**2) * np.sign(z[mask_out])

    # Potential phi(z) with phi(infinity) = 0
    # Inside (r < R): phi = (1/(4*pi*eps0)) * q / R
    # Outside (r > R): phi = (1/(4*pi*eps0)) * q / r
    phi = np.zeros_like(z)
    prefac = 1.0 / (4.0 * np.pi * EPS0)

    mask_in = r < R

    phi[mask_in] = prefac * q / R
    phi[mask_out] = prefac * q / r[mask_out]

    # Stack and save as CSV: columns z, E_z, phi
    data = np.column_stack([z, E_z, phi])

    csv_path = outdir / "griffiths_2p7_sphere.csv"
    header = "z [m], E_z [V/m], phi [V]"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
