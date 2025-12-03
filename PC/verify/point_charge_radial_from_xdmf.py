#!/usr/bin/env python3
"""
Radial E-field verification for a SINGLE point charge in a uniform dielectric,
using an existing XDMF output from PC/multi_point_charge_box.py (or point_charge_box.py).

We assume the domain is a cube and there is ONE charge at position x0.
We sample phi along the +x direction starting at the charge (y,z fixed),
estimate E_r(r) from finite differences, and compare to the analytic Coulomb field:

    |E_analytic(r)| = q / (4 pi eps0 epsr r^2)

This is meant as a local check (for r small compared to box size).

Example usage (after running multi_point_charge_box with a single charge):

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 PC/verify/point_charge_radial_from_xdmf.py \
      --xdmf results/multi_point_charge_box/run_YYYYMMDD-HHMMSS_dirichlet/multi_point_charge_box.xdmf \
      --q 1.602176634e-19 \
      --epsr 1.0 \
      --x0 "0.0,0.0,0.0" \
      --r-min 1.0e-8 \
      --r-max 4.0e-8 \
      --nr 200 \
      --delta 1.0e-9 \
      --outcsv results/multi_point_charge_box/point_charge_radial_check.csv \
      --outpng results/multi_point_charge_box/point_charge_radial_check.png
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, geometry, mesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # SI


def parse_args():
    p = argparse.ArgumentParser(description="Single point charge: radial E-field verification from XDMF")

    p.add_argument("--xdmf", type=str, required=True,
                   help="Path to XDMF with mesh + phi (from multi_point_charge_box or point_charge_box)")
    p.add_argument("--deg", type=int, default=1,
                   help="CG degree of phi space (must match solver; default 1)")
    p.add_argument("--q", type=float, default=1.602176634e-19,
                   help="Charge q [C]")
    p.add_argument("--epsr", type=float, default=1.0,
                   help="Relative permittivity eps_r (uniform)")
    p.add_argument("--x0", type=str, default="0.0,0.0,0.0",
                   help="Charge position 'x,y,z' in meters (used for choosing radial line)")
    p.add_argument("--r-min", type=float, default=1.0e-8,
                   help="Minimum radial distance r [m] from charge for sampling")
    p.add_argument("--r-max", type=float, default=4.0e-8,
                   help="Maximum radial distance r [m] from charge for sampling")
    p.add_argument("--nr", type=int, default=200,
                   help="Number of radial sample points")
    p.add_argument("--delta", type=float, default=1.0e-9,
                   help="Finite-difference step [m] for dphi/dr (must be small vs h, but not too small)")
    p.add_argument("--outcsv", type=str, required=True,
                   help="Output CSV path for r, E_num, E_analytic, rel_err")
    p.add_argument("--outpng", type=str, required=True,
                   help="Output PNG path for E(r) comparison plot")

    return p.parse_args()


def build_line_points(x0, r_vals, delta, domain):
    """
    Build 2N sampling points along +x direction through x0:

      minus: x = x0_x + r - delta/2
      plus:  x = x0_x + r + delta/2

    y,z stay fixed at x0_y, x0_z.

    Returns:
      pts (2N, 3), indices_minus (slice), indices_plus (slice)
    """
    r_vals = np.asarray(r_vals)
    n = r_vals.size

    x0 = np.asarray(x0, dtype=domain.geometry.x.dtype)
    x0_x, x0_y, x0_z = x0

    # Points for minus and plus evaluations
    x_minus = x0_x + r_vals - 0.5 * delta
    x_plus  = x0_x + r_vals + 0.5 * delta

    pts = np.zeros((2 * n, 3), dtype=domain.geometry.x.dtype)
    # First N: minus
    pts[:n, 0] = x_minus
    pts[:n, 1] = x0_y
    pts[:n, 2] = x0_z
    # Next N: plus
    pts[n:, 0] = x_plus
    pts[n:, 1] = x0_y
    pts[n:, 2] = x0_z

    idx_minus = slice(0, n)
    idx_plus = slice(n, 2 * n)

    return pts, idx_minus, idx_plus


def evaluate_phi_at_points(phi, domain, pts):
    """
    Evaluate scalar phi at arbitrary points using bb_tree + collisions.
    Returns an array values[i] = phi(pts[i]).
    Points outside the mesh get value 0 by default.
    """
    pts = pts.astype(domain.geometry.x.dtype, copy=False)
    dim = domain.topology.dim

    bb = geometry.bb_tree(domain, dim)
    candidates = geometry.compute_collisions_points(bb, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    num_pts = pts.shape[0]
    cells = np.full(num_pts, -1, dtype=np.int32)
    for i in range(num_pts):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = links[0]

    vals = phi.eval(pts, cells)
    vals = np.asarray(vals).reshape(-1)
    return vals


def main():
    args = parse_args()

    xdmf_path = Path(args.xdmf)
    outcsv = Path(args.outcsv)
    outpng = Path(args.outpng)

    if RANK == 0:
        outcsv.parent.mkdir(parents=True, exist_ok=True)
        outpng.parent.mkdir(parents=True, exist_ok=True)
        print("=== Radial E-field verification from XDMF (single point charge) ===")
        print(f"XDMF     : {xdmf_path}")
        print(f"q        : {args.q:.6e} C")
        print(f"epsr     : {args.epsr}")
        print(f"x0       : {args.x0}")
        print(f"r in     : [{args.r_min:.3e}, {args.r_max:.3e}] m, nr = {args.nr}")
        print(f"delta    : {args.delta:.3e} m")
        print(f"outcsv   : {outcsv}")
        print(f"outpng   : {outpng}")

    # ---- Read mesh + phi from XDMF ----
    with io.XDMFFile(COMM, xdmf_path, "r") as f:
        domain = f.read_mesh()
        V = fem.functionspace(domain, ("CG", args.deg))
        phi = fem.Function(V, name="phi")
        f.read_function(phi)

    # Ensure connectivity (usually already there, but cheap)
    dim = domain.topology.dim
    domain.topology.create_connectivity(dim, dim)

    # Parse x0
    x0 = np.array([float(s) for s in args.x0.split(",")], dtype=domain.geometry.x.dtype)
    if x0.size != 3:
        raise ValueError("x0 must be 'x,y,z' with three components")

    # ---- Radial sample points ----
    r_vals = np.linspace(args.r_min, args.r_max, args.nr)
    pts, idx_minus, idx_plus = build_line_points(x0, r_vals, args.delta, domain)

    # ---- Evaluate phi at these points and estimate E_r ----
    phi_vals = evaluate_phi_at_points(phi, domain, pts)
    n = r_vals.size
    phi_minus = phi_vals[idx_minus]
    phi_plus = phi_vals[idx_plus]

    # Finite-difference derivative: E_r ≈ - (phi_plus - phi_minus) / delta
    E_num = -(phi_plus - phi_minus) / args.delta  # V/m along +x (radial direction)

    # ---- Analytic Coulomb field ----
    eps = EPS0 * args.epsr
    E_analytic = args.q / (4.0 * np.pi * eps * r_vals**2)  # magnitude

    # Avoid r=0 singularities (we start at r_min > 0 anyway)
    rel_err = np.abs(E_num - E_analytic) / np.abs(E_analytic)

    # ---- Save CSV ----
    if RANK == 0:
        data = np.column_stack([r_vals, E_num, E_analytic, rel_err])
        header = "r_m,E_num_V_per_m,E_analytic_V_per_m,rel_err"
        np.savetxt(outcsv, data, delimiter=",", header=header, comments="")
        print(f"[radial_check] Wrote CSV: {outcsv}")

    # ---- Make plot ----
    if RANK == 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

        ax.loglog(r_vals, np.abs(E_num), label="|E_num(r)| FEM")
        ax.loglog(r_vals, np.abs(E_analytic), "--", label="|E_analytic(r)| = q/(4π ε r²)")

        ax.set_xlabel("r (m)")
        ax.set_ylabel("|E(r)| (V/m)")
        ax.set_title("Radial E-field: FEM vs analytic (single point charge)")
        ax.legend()
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(outpng)
        plt.close(fig)

        print(f"[radial_check] Wrote plot: {outpng}")

        # Also print some summary errors
        print("[radial_check] Sample of relative errors:")
        for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
            idx = int(frac * (len(r_vals) - 1))
            print(f"  r = {r_vals[idx]:.3e} m: rel_err = {rel_err[idx]:.3e}")


if __name__ == "__main__":
    main()
