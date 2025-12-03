#!/usr/bin/env python3
"""
Griffiths Problem 2.7: Analytic spherical shell field on a 3D box mesh.

We put a spherical surface of radius R, centered at the origin, with
uniform surface charge density sigma. Total charge:

    q = sigma * 4*pi*R^2.

Analytic solution (in infinite space):

  - For r < R:
      E = 0
      phi = (1/(4*pi*eps0)) * q / R   (constant)
  - For r > R:
      E(r) = (1/(4*pi*eps0)) * q / r^2 * r_hat
      phi(r) = (1/(4*pi*eps0)) * q / r

We:
  * Build a box mesh [-L/2,L/2]^3 with cell size ~h.
  * Compute analytic phi(x,y,z) and E(x,y,z) on the mesh.
  * Write to XDMF as phi_exact and E_exact.

BC flag (--bc) is used only for labeling the run (dirichlet/periodic),
not for solving a PDE.

Usage example:

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 PC/verify/griffiths_2p7_sphere_xdmf.py \
      --L=1e-7 --h=5e-9 \
      --R=5e-8 --sigma=1e-3 \
      --deg=1 \
      --bc=dirichlet \
      --outdir=results/griffiths_2p7_sphere_box
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI

from dolfinx import mesh, fem, io

# ---------- Constants ----------
COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analytic spherical shell field on box mesh")

    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length (domain is [-L/2, L/2]^3) [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--R", type=float, default=5e-8,
                   help="Sphere radius R [m] (must be < L/2)")
    p.add_argument("--sigma", type=float, default=1e-3,
                   help="Surface charge density sigma [C/m^2]")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for phi_exact")
    p.add_argument("--bc", type=str, default="dirichlet",
                   help="BC label: 'dirichlet' or 'periodic' (label only)")
    p.add_argument("--outdir", type=str, default="results/griffiths_2p7_sphere_box",
                   help="Base output directory")

    return p.parse_args()


# ---------- Geometry & Mesh ----------

def build_box(L: float, h: float) -> mesh.Mesh:
    """
    Build a 3D box mesh over [-L/2, L/2]^3 with approx cell size ~ h.
    """
    nx = max(2, int(np.round(L / h)))
    ny = nx
    nz = nx

    p0 = np.array([-L / 2, -L / 2, -L / 2], dtype=np.double)
    p1 = np.array([+L / 2, +L / 2, +L / 2], dtype=np.double)

    domain = mesh.create_box(
        COMM,
        points=[p0, p1],
        n=[nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


# ---------- Analytic φ and E ----------

def make_phi_E_analytic(domain: mesh.Mesh,
                        R: float,
                        sigma: float,
                        deg: int):
    """
    Build analytic phi_exact and E_exact for the spherical shell:
      - phi_exact in CG(deg)
      - E_exact  in DG0 vector space
    """
    gdim = domain.geometry.dim
    assert gdim == 3

    # Total charge
    q = sigma * 4.0 * np.pi * R**2
    prefac = 1.0 / (4.0 * np.pi * EPS0)

    # Spaces
    V = fem.functionspace(domain, ("Lagrange", deg))
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0, (gdim,)))

    # Analytic phi(x)
    def phi_fun(x):
        # x: shape (gdim, npoints)
        r = np.sqrt(np.sum(x * x, axis=0))
        phi_vals = np.zeros_like(r)

        mask_in = r < R
        mask_out = r >= R

        # Inside: constant
        phi_vals[mask_in] = prefac * q / R

        # Outside: q / r
        # avoid division by zero via mask
        phi_vals[mask_out] = prefac * q / r[mask_out]

        return phi_vals

    # Analytic E(x)
    def E_fun(x):
        # x: shape (gdim, npoints)
        r = np.sqrt(np.sum(x * x, axis=0))
        E_vals = np.zeros_like(x)

        mask_out = r > R
        if np.any(mask_out):
            # E = prefac * q / r^2 * r_hat
            # = prefac * q / r^3 * x
            r3 = r[mask_out]**3
            coeff = prefac * q / r3  # shape (n_out,)
            # Broadcast coeff to x-components
            E_vals[:, mask_out] = x[:, mask_out] * coeff

        # Inside sphere: E = 0 (already set)
        return E_vals

    phi_exact = fem.Function(V, name="phi_exact")
    phi_exact.interpolate(phi_fun)

    E_exact = fem.Function(W, name="E_exact")
    E_exact.interpolate(E_fun)

    return phi_exact, E_exact, q


# ---------- Main ----------

def main():
    args = parse_args()
    bc_label = args.bc.lower()
    if bc_label not in ("dirichlet", "periodic"):
        raise ValueError("bc must be 'dirichlet' or 'periodic' (label only here)")

    if args.R >= 0.5 * args.L and RANK == 0:
        print(f"[WARN] R = {args.R:.3e} is not < L/2 = {0.5*args.L:.3e}; "
              "sphere may intersect box boundary.")

    domain = build_box(args.L, args.h)
    phi_exact, E_exact, q = make_phi_E_analytic(domain, args.R, args.sigma, args.deg)

    if RANK == 0:
        print("=== Griffiths 2.7: spherical shell on box mesh ===")
        print(f"L      = {args.L:.3e} m")
        print(f"h      ~ {args.h:.3e} m")
        print(f"R      = {args.R:.3e} m")
        print(f"sigma  = {args.sigma:.3e} C/m^2")
        print(f"q      = {q:.6e} C")
        print(f"deg    = {args.deg}")
        print(f"BC label (for run naming): {bc_label}")
        print(f"Base outdir = {args.outdir}")

    # Prepare output dirs
    base_outdir = Path(args.outdir)
    if RANK == 0:
        base_outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_outdir / f"run_{timestamp}_{bc_label}"
    if RANK == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    # Save params
    if RANK == 0:
        params = {
            "L": args.L,
            "h": args.h,
            "R": args.R,
            "sigma": args.sigma,
            "q": q,
            "deg": args.deg,
            "bc_label": bc_label,
        }
        with (run_dir / "params.json").open("w") as f:
            json.dump(params, f, indent=2)
        print(f"Wrote params.json in {run_dir}")

    # Write XDMF
    xdmf_path = run_dir / "griffiths_2p7_sphere_box.xdmf"
    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi_exact)
        xdmf.write_function(E_exact)

    if RANK == 0:
        print(f"Wrote {xdmf_path}")


if __name__ == "__main__":
    main()
