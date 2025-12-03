#!/usr/bin/env python3
"""
Read phi from an XDMF file (Jackson dielectric interface run) and
make an x–z equipotential contour plot at y = 0, similar to the
analytic sketch.

Usage example:

  docker run --rm -v "\$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 PC/verify/jackson_dielectric_interface_contours_from_xdmf.py \
      --xdmf results/jackson_dielectric_interface_d10nm/jackson_dielectric_interface.xdmf \
      --deg 1 \
      --extent-nm 80 \
      --nslice 401 \
      --outpng results/jackson_dielectric_interface_d10nm/phi_xz_contours_from_xdmf.png
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


def parse_args():
    p = argparse.ArgumentParser(description="Equipotential contours from XDMF (xz-plane)")
    p.add_argument(
        "--xdmf",
        type=str,
        required=True,
        help="Path to XDMF file with mesh + phi",
    )
    p.add_argument(
        "--deg",
        type=int,
        default=1,
        help="CG degree for phi space (must match solver)",
    )
    p.add_argument(
        "--extent-nm",
        type=float,
        default=80.0,
        help="Plot extent in nm (square window: [-L/2, L/2] in x,z)",
    )
    p.add_argument(
        "--nslice",
        type=int,
        default=401,
        help="Number of sample points per axis for the xz slice",
    )
    p.add_argument(
        "--outpng",
        type=str,
        required=True,
        help="Output PNG path for the contour plot",
    )
    return p.parse_args()


def sample_phi_xz(phi: fem.Function, domain: mesh.Mesh, extent_m: float, ns: int):
    """
    Sample phi(x,z) on the plane y = 0, for
      x,z in [-extent_m/2, extent_m/2].

    Returns X_nm, Z_nm (nm units) and Phi (scalar values).
    """
    half = extent_m / 2.0
    grid = np.linspace(-half, half, ns)

    X, Z = np.meshgrid(grid, grid, indexing="ij")
    Y = np.zeros_like(X)

    # pts shape = (N,3)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0).T
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
    Phi = vals.reshape(X.shape)

    # convert coordinates to nm for plotting
    X_nm = X * 1e9
    Z_nm = Z * 1e9
    return X_nm, Z_nm, Phi


def make_contour_plot(X_nm, Z_nm, Phi, outpng: Path):
    """
    Make an equipotential contour plot similar to the textbook sketch.
    We normalize phi so the contours are dimensionless (shape-focused).
    """
    phi0 = np.max(np.abs(Phi))
    Phi_norm = Phi / phi0 if phi0 != 0 else Phi

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Nice ring-ish levels
    levels = np.linspace(0.0, 1.0, 16)[1:]  # skip exactly zero
    cs = ax.contour(X_nm, Z_nm, Phi_norm, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.3f")

    # Interface at z = 0
    ax.axhline(0.0, color="orange", linestyle="--", linewidth=1.5)

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Equipotentials for q near dielectric interface\n(from FEM solution)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(outpng)
    plt.close(fig)

    if RANK == 0:
        print(f"[contours_from_xdmf] Wrote {outpng}")


def main():
    args = parse_args()
    xdmf_path = Path(args.xdmf)
    outpng = Path(args.outpng)

    extent_m = args.extent_nm * 1e-9

    if RANK == 0:
        outpng.parent.mkdir(parents=True, exist_ok=True)

    # --- Read mesh + phi from XDMF ---
    with io.XDMFFile(COMM, xdmf_path, "r") as f:
        # FIX: don't pass name=None; let it use the default "mesh"
        domain = f.read_mesh()  # equivalent to read_mesh(name="mesh")
        V = fem.functionspace(domain, ("CG", args.deg))
        phi = fem.Function(V, name="phi")
        f.read_function(phi)  # reads dataset named "phi"

    # --- Sample and plot ---
    X_nm, Z_nm, Phi = sample_phi_xz(phi, domain, extent_m, args.nslice)
    make_contour_plot(X_nm, Z_nm, Phi, outpng)


if __name__ == "__main__":
    main()
