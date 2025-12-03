#!/usr/bin/env python3
"""
Jackson/Griffiths dielectric interface test:

- Interface at z = 0
- Region 1 (z > 0): epsr_top
- Region 2 (z < 0): epsr_bot
- Single Gaussian-smeared point charge at (0,0,d)

Solves: -div(eps(z) grad phi) = rho
with Dirichlet phi = 0 on outer box, then:

- writes XDMF (mesh, epsr, phi, E, rho)
- creates PNGs of phi and rho on:
    * x-z slice at y = 0
    * y-z slice at x = 0
    * x-y slice at z = d

Run example (your case, d = 10 nm):

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \\
    /dolfinx-env/bin/python3 PC/verify/jackson_dielectric_interface_plots.py \\
      --L=1e-7 --h=5e-9 \\
      --epsr-bot=11.7 --epsr-top=3.9 --split-z=0.0 \\
      --q=1.602176634e-19 \\
      --d=1e-8 \\
      --sigma=5e-9 \\
      --deg=1 \\
      --nslice=201 \\
      --outdir=results/jackson_dielectric_interface_d10nm
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
RANK = COMM.rank
EPS0 = 8.8541878128e-12


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Jackson dielectric interface + slice plots")

    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length (domain is [-L/2,L/2]^3) [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for phi")

    # permittivities: bottom (z<split_z) and top (z>=split_z)
    p.add_argument("--epsr-bot", type=float, default=11.7,
                   help="Relative permittivity for z < split_z")
    p.add_argument("--epsr-top", type=float, default=3.9,
                   help="Relative permittivity for z >= split_z")
    p.add_argument("--split-z", type=float, default=0.0,
                   help="Interface plane z coordinate [m]")

    # charge
    p.add_argument("--q", type=float, default=1.602176634e-19,
                   help="Total charge [C]")
    p.add_argument("--d", type=float, default=1e-8,
                   help="Charge height above interface, z = d [m]")
    p.add_argument("--sigma", type=float, default=5e-9,
                   help="Gaussian width sigma [m]")

    # slice resolution
    p.add_argument("--nslice", type=int, default=201,
                   help="Number of points per axis in slices")

    p.add_argument("--outdir", type=str,
                   default="results/jackson_dielectric_interface_d10nm",
                   help="Output directory (XDMF + PNGs)")

    return p.parse_args()


# ---------- Mesh & materials ----------

def build_box(L, h):
    nx = max(2, int(round(L / h)))
    ny = nx
    nz = nx

    p0 = np.array([-L/2, -L/2, -L/2], dtype=np.double)
    p1 = np.array([+L/2, +L/2, +L/2], dtype=np.double)

    domain = mesh.create_box(
        COMM,
        points=[p0, p1],
        n=[nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


def make_epsr(domain, epsr_bot, epsr_top, split_z):
    """Return (eps_expr, epsr_dg0) for 2-layer epsr(z)."""
    V_eps = fem.functionspace(domain, ("Discontinuous Lagrange", 0))

    x = ufl.SpatialCoordinate(domain)
    z = x[2]

    epsr_expr = ufl.conditional(ufl.lt(z, split_z), epsr_bot, epsr_top)
    eps_expr = EPS0 * epsr_expr

    epsr_dg0 = fem.Function(V_eps, name="epsr")
    expr_epsr = fem.Expression(epsr_expr, V_eps.element.interpolation_points)
    epsr_dg0.interpolate(expr_epsr)

    return eps_expr, epsr_dg0


# ---------- Charge ρ ----------

def make_rho(domain, V, q, d, sigma):
    """Single Gaussian charge at (0,0,d)."""
    x0 = np.array([0.0, 0.0, d], dtype=np.double)
    sigma = float(sigma)
    prefac = q / ((sigma * np.sqrt(2.0 * np.pi)) ** 3)

    def rho_fun(x):
        dx = x.T - x0
        r2 = np.sum(dx * dx, axis=1)
        return prefac * np.exp(-0.5 * r2 / sigma**2)

    rho = fem.Function(V, name="rho")
    rho.interpolate(rho_fun)
    return rho


# ---------- Slice sampling (using bb_tree API) ----------

def sample_phi_on_plane(field, domain, plane, L, ns):
    """
    Sample a scalar FEM field on a vertical plane.

    plane: "xz"  -> slice at y = 0   (x-z plane)
           "yz"  -> slice at x = 0   (y-z plane)

    Returns: (X, Y, Z, Field_grid)
    """
    xs = np.linspace(-L/2, L/2, ns)
    ys = np.linspace(-L/2, L/2, ns)
    zs = np.linspace(-L/2, L/2, ns)

    if plane == "xz":
        X, Z = np.meshgrid(xs, zs)
        Y = np.zeros_like(X)
    elif plane == "yz":
        Y, Z = np.meshgrid(ys, zs)
        X = np.zeros_like(Y)
    else:
        raise ValueError("plane must be 'xz' or 'yz'")

    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])  # (3, npoints)

    dim = domain.topology.dim
    tree = geometry.bb_tree(domain, dim)
    collisions = geometry.compute_collisions(tree, pts)
    cells = geometry.compute_colliding_cells(domain, collisions)

    values = np.full(pts.shape[1], np.nan, dtype=np.float64)
    inside = cells >= 0
    if np.any(inside):
        vals_inside = field.eval(pts[:, inside], cells[inside])
        vals_inside = np.asarray(vals_inside).reshape(-1)
        values[inside] = vals_inside

    Field_grid = values.reshape(X.shape)
    return X, Y, Z, Field_grid


def sample_phi_on_xy_plane(field, domain, L, ns, z0):
    """
    Sample a scalar FEM field on a horizontal plane z = z0.

    Returns: (X, Y, Z, Field_grid)
    """
    xs = np.linspace(-L/2, L/2, ns)
    ys = np.linspace(-L/2, L/2, ns)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, z0)

    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])  # (3, npoints)

    dim = domain.topology.dim
    tree = geometry.bb_tree(domain, dim)
    collisions = geometry.compute_collisions(tree, pts)
    cells = geometry.compute_colliding_cells(domain, collisions)

    values = np.full(pts.shape[1], np.nan, dtype=np.float64)
    inside = cells >= 0
    if np.any(inside):
        vals_inside = field.eval(pts[:, inside], cells[inside])
        vals_inside = np.asarray(vals_inside).reshape(-1)
        values[inside] = vals_inside

    Field_grid = values.reshape(X.shape)
    return X, Y, Z, Field_grid


# ---------- Main ----------

def main():
    args = parse_args()
    t0 = perf_counter()

    if RANK == 0:
        print("=== Jackson dielectric interface solve + plots ===")
        print(f"L = {args.L:.3e} m, h ~ {args.h:.3e} m")
        print(f"epsr_bot (z<split) = {args.epsr_bot:.3g}")
        print(f"epsr_top (z>=split)= {args.epsr_top:.3g}")
        print(f"split_z            = {args.split_z:.3e} m")
        print(f"q = {args.q:.6e} C, d = {args.d:.3e} m, sigma = {args.sigma:.3e} m")
        print(f"deg = {args.deg}, nslice = {args.nslice}")
        print(f"outdir = {args.outdir}")

    # Build mesh, spaces
    domain = build_box(args.L, args.h)
    dim = domain.topology.dim
    gdim = domain.geometry.dim
    assert dim == 3 and gdim == 3

    V = fem.functionspace(domain, ("Lagrange", args.deg))
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0, (gdim,)))
    dx = ufl.dx(domain)

    # eps(z)
    eps_expr, epsr_dg0 = make_epsr(domain, args.epsr_bot, args.epsr_top, args.split_z)

    # rho
    rho = make_rho(domain, V, args.q, args.d, args.sigma)
    total_q = fem.assemble_scalar(fem.form(rho * dx))
    total_q = COMM.allreduce(total_q, op=MPI.SUM)
    if RANK == 0:
        print(f"Total charge from rho: {total_q:.6e} C")

    # weak form: -div(eps grad phi) = rho
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = eps_expr * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_form = rho * v * dx

    # Dirichlet phi=0 on all faces
    facet_indices = mesh.locate_entities_boundary(
        domain, dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, dim - 1, facet_indices)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
    }

    problem = LinearProblem(
        a, L_form,
        bcs=[bc],
        petsc_options=petsc_opts,
        petsc_options_prefix="jackson_eps_",
    )

    t1 = perf_counter()
    phi = problem.solve()
    t2 = perf_counter()
    phi.name = "phi"

    if RANK == 0:
        try:
            its = problem.solver.getIterationNumber()
            print(f"KSP iterations: {its}")
        except Exception:
            pass
        print(f"Solve time (problem + solve): {t2 - t1:.4f} s")

    # E = -grad phi in DG0
    E_expr = -ufl.grad(phi)
    E = fem.Function(W, name="E")
    E_interp = fem.Expression(E_expr, W.element.interpolation_points)
    E.interpolate(E_interp)

    # Output directory
    outdir = Path(args.outdir)
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    # XDMF
    xdmf_path = outdir / "jackson_dielectric_interface.xdmf"
    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(epsr_dg0)
        xdmf.write_function(phi)
        xdmf.write_function(E)
        xdmf.write_function(rho)

    if RANK == 0:
        print(f"Wrote XDMF: {xdmf_path}")

    # ---------- Make slice plots (rank 0 only) ----------
    if RANK == 0:
        ns = args.nslice
        Lbox = args.L

        # ===== phi slices =====

        # x-z slice (y=0)
        X, Y, Z, Phi_xz = sample_phi_on_plane(phi, domain, "xz", Lbox, ns)
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(X * 1e9, Z * 1e9, Phi_xz, levels=30)
        plt.colorbar(c, ax=ax, label="phi (V, up to additive const.)")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.plot(0.0, args.d * 1e9, "wo", markersize=4)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("z (nm)")
        ax.set_title("phi(x,z) slice at y=0")
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(outdir / "phi_xz_slice.png", dpi=200)
        plt.close(fig)

        # y-z slice (x=0)
        X2, Y2, Z2, Phi_yz = sample_phi_on_plane(phi, domain, "yz", Lbox, ns)
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(Y2 * 1e9, Z2 * 1e9, Phi_yz, levels=30)
        plt.colorbar(c, ax=ax, label="phi (V, up to additive const.)")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.plot(0.0, args.d * 1e9, "wo", markersize=4)
        ax.set_xlabel("y (nm)")
        ax.set_ylabel("z (nm)")
        ax.set_title("phi(y,z) slice at x=0")
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(outdir / "phi_yz_slice.png", dpi=200)
        plt.close(fig)

        # x-y slice at z=d (through the charge)
        X3, Y3, Z3, Phi_xy = sample_phi_on_xy_plane(phi, domain, Lbox, ns, args.d)
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(X3 * 1e9, Y3 * 1e9, Phi_xy, levels=30)
        plt.colorbar(c, ax=ax, label="phi (V, up to additive const.)")
        ax.plot(0.0, 0.0, "wo", markersize=4)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"phi(x,y) slice at z = {args.d*1e9:.1f} nm")
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(outdir / "phi_xy_at_zd_slice.png", dpi=200)
        plt.close(fig)

        # ===== rho slices (same planes) =====

        # x-z slice (y=0)
        Xr, Yr, Zr, Rho_xz = sample_phi_on_plane(rho, domain, "xz", Lbox, ns)
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(Xr * 1e9, Zr * 1e9, Rho_xz, levels=30)
        plt.colorbar(c, ax=ax, label=r"rho (C/m$^3$)")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.plot(0.0, args.d * 1e9, "wo", markersize=4)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("z (nm)")
        ax.set_title("rho(x,z) slice at y=0")
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(outdir / "rho_xz_slice.png", dpi=200)
        plt.close(fig)

        # y-z slice (x=0)
        Xr2, Yr2, Zr2, Rho_yz = sample_phi_on_plane(rho, domain, "yz", Lbox, ns)
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(Yr2 * 1e9, Zr2 * 1e9, Rho_yz, levels=30)
        plt.colorbar(c, ax=ax, label=r"rho (C/m$^3$)")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.plot(0.0, args.d * 1e9, "wo", markersize=4)
        ax.set_xlabel("y (nm)")
        ax.set_ylabel("z (nm)")
        ax.set_title("rho(y,z) slice at x=0")
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(outdir / "rho_yz_slice.png", dpi=200)
        plt.close(fig)

        # x-y slice at z=d (through the charge)
        Xr3, Yr3, Zr3, Rho_xy = sample_phi_on_xy_plane(rho, domain, Lbox, ns, args.d)
        fig, ax = plt.subplots(figsize=(6, 5))
        c = ax.contourf(Xr3 * 1e9, Yr3 * 1e9, Rho_xy, levels=30)
        plt.colorbar(c, ax=ax, label=r"rho (C/m$^3$)")
        ax.plot(0.0, 0.0, "wo", markersize=4)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"rho(x,y) slice at z = {args.d*1e9:.1f} nm")
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(outdir / "rho_xy_at_zd_slice.png", dpi=200)
        plt.close(fig)

        print("Wrote slice PNGs (phi and rho) in", outdir)

    t3 = perf_counter()
    if RANK == 0:
        print(f"Total runtime: {t3 - t0:.4f} s")


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------
# Sampling helpers compatible with current dolfinx geometry API
# -------------------------------------------------------------------
from dolfinx import geometry as _geometry


def _find_cells_for_points(domain: mesh.Mesh, pts: np.ndarray) -> np.ndarray:
    """
    Given an (N, 3) array of points in physical coordinates, return an
    (N,) int32 array of cell indices, using the first colliding cell
    for each point (or -1 if none).
    """
    dim = domain.topology.dim
    assert dim == 3
    assert pts.shape[1] == domain.geometry.dim

    # Build bounding-box tree on cells
    tree = _geometry.bb_tree(domain, dim)

    # Candidate collisions and actual colliding cells (AdjacencyList)
    candidates = _geometry.compute_collisions_points(tree, pts)
    colliding = _geometry.compute_colliding_cells(domain, candidates, pts)

    num_points = pts.shape[0]
    cells = np.full(num_points, -1, dtype=np.int32)

    # For each point, take the first colliding cell (if any)
    for i in range(colliding.num_nodes):
        links_i = colliding.links(i)
        if len(links_i) > 0:
            cells[i] = links_i[0]

    return cells


def sample_phi_on_plane(phi: fem.Function,
                        domain: mesh.Mesh,
                        plane: str,
                        L: float,
                        ns: int):
    """
    Sample scalar phi on a symmetric plane:

      plane = "xz": vary x,z, y=0
      plane = "yz": vary y,z, x=0
      plane = "xy": vary x,y, z=0

    Returns X, Y, Z, Phi with shape (ns, ns).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # Evaluate phi at points (Function.eval: x has shape (N,3))
    vals_flat = np.zeros(pts.shape[0], dtype=phi.x.array.dtype)
    phi.eval(pts, cells, vals_flat)

    Phi = vals_flat.reshape(xg.shape)
    return xg, yg, zg, Phi


def sample_E_on_plane(E: fem.Function,
                      domain: mesh.Mesh,
                      plane: str,
                      L: float,
                      ns: int):
    """
    Sample vector E on a symmetric plane.
    Returns X, Y, Z, Ex, Ey, Ez (all shape (ns, ns)).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # E is vector-valued, so eval returns (gdim, N)
    gdim = domain.geometry.dim
    vals = np.zeros((gdim, pts.shape[0]), dtype=E.x.array.dtype)
    E.eval(pts, cells, vals)

    Ex = vals[0, :].reshape(xg.shape)
    Ey = vals[1, :].reshape(xg.shape)
    Ez = vals[2, :].reshape(xg.shape)

    return xg, yg, zg, Ex, Ey, Ez


# -------------------------------------------------------------------
# Sampling helpers compatible with current dolfinx geometry API
# -------------------------------------------------------------------
from dolfinx import geometry as _geometry


def _find_cells_for_points(domain: mesh.Mesh, pts: np.ndarray) -> np.ndarray:
    """
    Given an (N, 3) array of points in physical coordinates, return an
    (N,) int32 array of cell indices, using the first colliding cell
    for each point (or -1 if none).
    """
    dim = domain.topology.dim
    assert dim == 3
    assert pts.shape[1] == domain.geometry.dim

    # Build bounding-box tree on cells
    tree = _geometry.bb_tree(domain, dim)

    # Candidate collisions and actual colliding cells (AdjacencyList)
    candidates = _geometry.compute_collisions_points(tree, pts)
    colliding = _geometry.compute_colliding_cells(domain, candidates, pts)

    num_points = pts.shape[0]
    cells = np.full(num_points, -1, dtype=np.int32)

    # For each point, take the first colliding cell (if any)
    for i in range(colliding.num_nodes):
        links_i = colliding.links(i)
        if len(links_i) > 0:
            cells[i] = links_i[0]

    return cells


def sample_phi_on_plane(phi: fem.Function,
                        domain: mesh.Mesh,
                        plane: str,
                        L: float,
                        ns: int):
    """
    Sample scalar phi on a symmetric plane:

      plane = "xz": vary x,z, y=0
      plane = "yz": vary y,z, x=0
      plane = "xy": vary x,y, z=0

    Returns X, Y, Z, Phi with shape (ns, ns).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # Evaluate phi at points (Function.eval: x has shape (N,3))
    vals_flat = np.zeros(pts.shape[0], dtype=phi.x.array.dtype)
    phi.eval(pts, cells, vals_flat)

    Phi = vals_flat.reshape(xg.shape)
    return xg, yg, zg, Phi


def sample_E_on_plane(E: fem.Function,
                      domain: mesh.Mesh,
                      plane: str,
                      L: float,
                      ns: int):
    """
    Sample vector E on a symmetric plane.
    Returns X, Y, Z, Ex, Ey, Ez (all shape (ns, ns)).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # E is vector-valued, so eval returns (gdim, N)
    gdim = domain.geometry.dim
    vals = np.zeros((gdim, pts.shape[0]), dtype=E.x.array.dtype)
    E.eval(pts, cells, vals)

    Ex = vals[0, :].reshape(xg.shape)
    Ey = vals[1, :].reshape(xg.shape)
    Ez = vals[2, :].reshape(xg.shape)

    return xg, yg, zg, Ex, Ey, Ez


# -------------------------------------------------------------------
# Sampling helpers compatible with current dolfinx geometry API
# -------------------------------------------------------------------
from dolfinx import geometry as _geometry


def _find_cells_for_points(domain: mesh.Mesh, pts: np.ndarray) -> np.ndarray:
    """
    Given an (N, 3) array of points in physical coordinates, return an
    (N,) int32 array of cell indices, using the first colliding cell
    for each point (or -1 if none).
    """
    dim = domain.topology.dim
    assert dim == 3
    assert pts.shape[1] == domain.geometry.dim

    # Build bounding-box tree on cells
    tree = _geometry.bb_tree(domain, dim)

    # Candidate collisions and actual colliding cells (AdjacencyList)
    candidates = _geometry.compute_collisions_points(tree, pts)
    colliding = _geometry.compute_colliding_cells(domain, candidates, pts)

    num_points = pts.shape[0]
    cells = np.full(num_points, -1, dtype=np.int32)

    # For each point, take the first colliding cell (if any)
    for i in range(colliding.num_nodes):
        links_i = colliding.links(i)
        if len(links_i) > 0:
            cells[i] = links_i[0]

    return cells


def sample_phi_on_plane(phi: fem.Function,
                        domain: mesh.Mesh,
                        plane: str,
                        L: float,
                        ns: int):
    """
    Sample scalar phi on a symmetric plane:

      plane = "xz": vary x,z, y=0
      plane = "yz": vary y,z, x=0
      plane = "xy": vary x,y, z=0

    Returns X, Y, Z, Phi with shape (ns, ns).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # Evaluate phi at points (Function.eval: x has shape (N,3))
    vals_flat = np.zeros(pts.shape[0], dtype=phi.x.array.dtype)
    phi.eval(pts, cells, vals_flat)

    Phi = vals_flat.reshape(xg.shape)
    return xg, yg, zg, Phi


def sample_E_on_plane(E: fem.Function,
                      domain: mesh.Mesh,
                      plane: str,
                      L: float,
                      ns: int):
    """
    Sample vector E on a symmetric plane.
    Returns X, Y, Z, Ex, Ey, Ez (all shape (ns, ns)).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # E is vector-valued, so eval returns (gdim, N)
    gdim = domain.geometry.dim
    vals = np.zeros((gdim, pts.shape[0]), dtype=E.x.array.dtype)
    E.eval(pts, cells, vals)

    Ex = vals[0, :].reshape(xg.shape)
    Ey = vals[1, :].reshape(xg.shape)
    Ez = vals[2, :].reshape(xg.shape)

    return xg, yg, zg, Ex, Ey, Ez


# -------------------------------------------------------------------
# Sampling helpers compatible with current dolfinx geometry API
# -------------------------------------------------------------------
from dolfinx import geometry as _geometry


def _find_cells_for_points(domain: mesh.Mesh, pts: np.ndarray) -> np.ndarray:
    """
    Given an (N, 3) array of points in physical coordinates, return an
    (N,) int32 array of cell indices, using the first colliding cell
    for each point (or -1 if none).
    """
    dim = domain.topology.dim
    assert dim == 3
    assert pts.shape[1] == domain.geometry.dim

    # Build bounding-box tree on cells
    tree = _geometry.bb_tree(domain, dim)

    # Candidate collisions and actual colliding cells (AdjacencyList)
    candidates = _geometry.compute_collisions_points(tree, pts)
    colliding = _geometry.compute_colliding_cells(domain, candidates, pts)

    num_points = pts.shape[0]
    cells = np.full(num_points, -1, dtype=np.int32)

    # For each point, take the first colliding cell (if any)
    for i in range(colliding.num_nodes):
        links_i = colliding.links(i)
        if len(links_i) > 0:
            cells[i] = links_i[0]

    return cells


def sample_phi_on_plane(phi: fem.Function,
                        domain: mesh.Mesh,
                        plane: str,
                        L: float,
                        ns: int):
    """
    Sample scalar phi on a symmetric plane:

      plane = "xz": vary x,z, y=0
      plane = "yz": vary y,z, x=0
      plane = "xy": vary x,y, z=0

    Returns X, Y, Z, Phi with shape (ns, ns).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # Evaluate phi at points (Function.eval: x has shape (N,3))
    vals_flat = np.zeros(pts.shape[0], dtype=phi.x.array.dtype)
    phi.eval(pts, cells, vals_flat)

    Phi = vals_flat.reshape(xg.shape)
    return xg, yg, zg, Phi


def sample_E_on_plane(E: fem.Function,
                      domain: mesh.Mesh,
                      plane: str,
                      L: float,
                      ns: int):
    """
    Sample vector E on a symmetric plane.
    Returns X, Y, Z, Ex, Ey, Ez (all shape (ns, ns)).
    """
    half = L / 2.0
    grid = np.linspace(-half, half, ns)

    if plane == "xz":
        xg, zg = np.meshgrid(grid, grid, indexing="ij")
        yg = np.zeros_like(xg)
    elif plane == "yz":
        yg, zg = np.meshgrid(grid, grid, indexing="ij")
        xg = np.zeros_like(yg)
    elif plane == "xy":
        xg, yg = np.meshgrid(grid, grid, indexing="ij")
        zg = np.zeros_like(xg)
    else:
        raise ValueError(f"Unknown plane '{plane}', expected 'xz', 'yz', or 'xy'.")

    pts = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    pts = pts.astype(domain.geometry.x.dtype, copy=False)

    cells = _find_cells_for_points(domain, pts)

    # E is vector-valued, so eval returns (gdim, N)
    gdim = domain.geometry.dim
    vals = np.zeros((gdim, pts.shape[0]), dtype=E.x.array.dtype)
    E.eval(pts, cells, vals)

    Ex = vals[0, :].reshape(xg.shape)
    Ey = vals[1, :].reshape(xg.shape)
    Ez = vals[2, :].reshape(xg.shape)

    return xg, yg, zg, Ex, Ey, Ez

