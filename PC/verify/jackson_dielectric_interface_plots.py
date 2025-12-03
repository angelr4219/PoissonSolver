#!/usr/bin/env python3
"""
Jackson dielectric interface: single point charge above ε1/ε2 interface,
solve -∇·(ε(r) ∇φ) = ρ, then make xz/yz/xy slice plots of φ.

Layers:
  z < split_z  -> epsr_bot
  z >= split_z -> epsr_top
"""

import argparse
from pathlib import Path
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMM = MPI.COMM_WORLD
RANK = COMM.rank
EPS0 = 8.8541878128e-12  # SI


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Jackson dielectric interface: solve + slice plots"
    )
    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length, domain = [-L/2,L/2]^3 [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--epsr-bot", type=float, default=11.7,
                   help="Relative permittivity for z < split_z")
    p.add_argument("--epsr-top", type=float, default=3.9,
                   help="Relative permittivity for z >= split_z")
    p.add_argument("--split-z", type=float, default=0.0,
                   help="Interface plane z = split_z [m]")
    p.add_argument("--q", type=float, default=1.602176634e-19,
                   help="Total charge [C]")
    p.add_argument("--d", type=float, default=1e-8,
                   help="Charge height above interface [m], so z0 = split_z + d")
    p.add_argument("--sigma", type=float, default=5e-9,
                   help="Gaussian width sigma [m]")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for φ")
    p.add_argument("--nslice", type=int, default=201,
                   help="Number of sample points per axis for slice plots")
    p.add_argument("--outdir", type=str,
                   default="results/jackson_dielectric_interface",
                   help="Output directory")

    return p.parse_args()


# ------------- Mesh + ε(r) + ρ ---------------

def build_box(L: float, h: float) -> mesh.Mesh:
    nx = max(2, int(np.round(L / h)))
    ny = nx
    nz = nx
    p0 = np.array([-L/2, -L/2, -L/2], dtype=np.double)
    p1 = np.array([+L/2, +L/2, +L/2], dtype=np.double)
    domain = mesh.create_box(
        COMM,
        [p0, p1],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


def build_eps_r(domain: mesh.Mesh,
                epsr_bot: float,
                epsr_top: float,
                split_z: float):
    """
    Two-layer ε(r):

      region 0 (z < split_z):    epsr_bot
      region 1 (z >= split_z):   epsr_top

    Encoded as:
      - cell_tags: meshtags with values 0 or 1
      - eps_r: DG0 scalar function with those region values
    """
    dim = domain.topology.dim
    gdim = domain.geometry.dim
    assert dim == 3 and gdim == 3

    # Ensure connectivity exists
    domain.topology.create_connectivity(dim, dim)

    tol = 1e-12 * max(1.0, abs(split_z))

    def in_bot(x):
        return x[2] < split_z - tol

    def in_top(x):
        return x[2] >= split_z - tol

    cells_bot = mesh.locate_entities(domain, dim, in_bot)
    cells_top = mesh.locate_entities(domain, dim, in_top)

    if RANK == 0:
        print(f"Cells bot: {cells_bot.size}, cells top: {cells_top.size}")

    indices = np.concatenate([cells_bot, cells_top])
    values = np.concatenate([
        np.full(cells_bot.size, 0, dtype=np.int32),
        np.full(cells_top.size, 1, dtype=np.int32),
    ])

    cell_tags = mesh.meshtags(domain, dim, indices, values)

    V_eps = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps_r = fem.Function(V_eps, name="epsr")

    # Default bottom value
    eps_r.x.array[:] = epsr_bot

    # Overwrite top-region cells
    mask_top = (cell_tags.values == 1)
    cells_top_only = cell_tags.indices[mask_top]
    eps_r.x.array[cells_top_only] = epsr_top

    return cell_tags, eps_r


def make_rho_gaussian(domain: mesh.Mesh,
                      q: float,
                      x0: np.ndarray,
                      sigma: float,
                      V: fem.FunctionSpace) -> fem.Function:
    """
    ρ(x) = q / ((σ √(2π))^3) * exp(-|x - x0|^2 / (2σ^2))
    """
    x0 = np.asarray(x0, dtype=np.double)
    sigma = float(sigma)
    q = float(q)
    prefac = q / ((sigma * np.sqrt(2.0 * np.pi)) ** 3)

    def rho_fun(x):
        dx = x.T - x0
        r2 = np.sum(dx*dx, axis=1)
        return prefac * np.exp(-0.5 * r2 / sigma**2)

    rho = fem.Function(V, name="rho")
    rho.interpolate(rho_fun)
    return rho


# ------------- Sampling helpers (new geometry API) --------------

def sample_phi_on_plane(phi: fem.Function,
                        domain: mesh.Mesh,
                        plane: str,
                        L: float,
                        ns: int):
    """
    Sample scalar phi on planes:

      plane = "xz": y=0
      plane = "yz": x=0
      plane = "xy": z=0
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        X, Z = np.meshgrid(grid, grid, indexing="ij")
        Y = np.zeros_like(X)
    elif plane == "yz":
        Y, Z = np.meshgrid(grid, grid, indexing="ij")
        X = np.zeros_like(Y)
    elif plane == "xy":
        X, Y = np.meshgrid(grid, grid, indexing="ij")
        Z = np.zeros_like(X)
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    # Points: shape (num_points, 3)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0).T

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

    # Evaluate phi at these points
    vals = phi.eval(pts, cells)
    vals = np.asarray(vals).reshape(-1)
    Phi = vals.reshape(X.shape)

    return X, Y, Z, Phi


# ------------- Plotting ---------------

def make_slice_plots(phi: fem.Function,
                     domain: mesh.Mesh,
                     L: float,
                     ns: int,
                     outdir: Path):
    """
    Make PNGs for φ on xz / yz / xy planes.
    """
    planes = ["xz", "yz", "xy"]
    for plane in planes:
        if RANK == 0:
            print(f"Sampling phi on {plane}-plane ...")
        X, Y, Z, Phi = sample_phi_on_plane(phi, domain, plane, L, ns)

        if plane == "xz":
            xlab, ylab = "x [m]", "z [m]"
            coords_x = X
            coords_y = Z
        elif plane == "yz":
            xlab, ylab = "y [m]", "z [m]"
            coords_x = Y
            coords_y = Z
        else:  # "xy"
            xlab, ylab = "x [m]", "y [m]"
            coords_x = X
            coords_y = Y

        if RANK == 0:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
            pcm = ax.pcolormesh(coords_x, coords_y, Phi, shading="auto")
            plt.colorbar(pcm, ax=ax, label="φ [V] (up to constant)")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(f"φ on {plane}-plane")
            fig.tight_layout()
            png_path = outdir / f"phi_slice_{plane}.png"
            fig.savefig(png_path)
            plt.close(fig)
            print(f"Wrote {png_path}")


# ------------- Main ---------------

def main():
    args = parse_args()
    L = args.L
    h = args.h
    epsr_bot = args.epsr_bot
    epsr_top = args.epsr_top
    split_z = args.split_z
    q = args.q
    d = args.d
    sigma = args.sigma
    deg = args.deg
    ns = args.nslice
    outdir = Path(args.outdir)

    if RANK == 0:
        print("=== Jackson dielectric interface solve + plots (v3, scalar-only) ===")
        print(f"L = {L:.3e} m, h ~ {h:.3e} m")
        print(f"epsr_bot (z<split) = {epsr_bot}")
        print(f"epsr_top (z>=split)= {epsr_top}")
        print(f"split_z            = {split_z:.3e} m")
        print(f"q = {q:.6e} C, d = {d:.3e} m, sigma = {sigma:.3e} m")
        print(f"deg = {deg}, nslice = {ns}")
        print(f"outdir = {outdir}")

    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    t0 = time.perf_counter()

    # Build mesh
    domain = build_box(L, h)
    dim = domain.topology.dim
    dx = ufl.dx(domain)

    # Function space for phi and rho
    V = fem.functionspace(domain, ("CG", deg))

    # ε(r)
    cell_tags, eps_r = build_eps_r(domain, epsr_bot, epsr_top, split_z)
    eps = EPS0 * eps_r

    # ρ: single Gaussian at (0,0,split_z+d)
    x0 = np.array([0.0, 0.0, split_z + d], dtype=np.double)
    rho = make_rho_gaussian(domain, q, x0, sigma, V)

    # Weak form: -∇·(ε∇φ) = ρ
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * dx
    Lform = rho * v * dx

    # Dirichlet φ = 0 on boundary
    facet_indices = mesh.locate_entities_boundary(
        domain, dim - 1,
        lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, dim - 1, facet_indices)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    problem = LinearProblem(
        a, Lform,
        bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
        },
    )

    phi = problem.solve()
    phi.name = "phi"

    if RANK == 0:
        its = problem.solver.getIterationNumber()
        print(f"KSP iterations: {its}")

    t1 = time.perf_counter()
    if RANK == 0:
        print(f"Solve time (problem + solve): {t1 - t0:.4f} s")

    # Write XDMF
    xdmf_path = outdir / "jackson_dielectric_interface.xdmf"
    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(eps_r)

    if RANK == 0:
        print(f"Wrote XDMF: {xdmf_path}")

    # Slice plots of φ
    make_slice_plots(phi, domain, L, ns, outdir)

    t2 = time.perf_counter()
    if RANK == 0:
        print(f"Total runtime: {t2 - t0:.4f} s")


if __name__ == "__main__":
    main()
