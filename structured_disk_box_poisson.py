#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, io, mesh, geometry
from dolfinx.mesh import CellType
from dolfinx.fem.petsc import LinearProblem


def make_mesh(comm, Lx, Ly, Lz, Nx, Ny, Nz, cell_type_name):
    if cell_type_name == "hex":
        cell_type = CellType.hexahedron
    elif cell_type_name == "tet":
        cell_type = CellType.tetrahedron
    else:
        raise ValueError(f"Unknown cell type: {cell_type_name}")

    msh = mesh.create_box(
        comm,
        [[-0.5 * Lx, -0.5 * Ly, -0.5 * Lz],
         [ 0.5 * Lx,  0.5 * Ly,  0.5 * Lz]],
        [Nx, Ny, Nz],
        cell_type=cell_type,
    )
    return msh


def mark_disk_cells(msh, R, t, z0):
    """
    Mark cells whose CELL MIDPOINT lies inside the thin cylinder:
        x^2 + y^2 <= R^2
        |z - z0| <= t/2
    """
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)

    mids = mesh.compute_midpoints(msh, tdim, cells)
    x = mids[:, 0]
    y = mids[:, 1]
    z = mids[:, 2]

    mask = (x * x + y * y <= R * R) & (np.abs(z - z0) <= 0.5 * t)
    return cells[mask]


def build_bcs(V, msh, Lx, Ly, Lz):
    fdim = msh.topology.dim - 1
    xh = 0.5 * Lx
    yh = 0.5 * Ly
    zh = 0.5 * Lz

    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim,
        lambda x: (
            np.isclose(x[0], -xh) | np.isclose(x[0], xh)
            | np.isclose(x[1], -yh) | np.isclose(x[1], yh)
            | np.isclose(x[2], -zh) | np.isclose(x[2], zh)
        )
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    return [bc]


def solve_poisson(comm, msh, R, t, z0, Q, epsr, Lx, Ly, Lz):
    eps0 = 8.8541878128e-12
    eps = eps0 * epsr

    V = fem.functionspace(msh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    Q0 = fem.functionspace(msh, ("DG", 0))
    rho = fem.Function(Q0)
    rho.name = "rho"
    rho.x.array[:] = 0.0

    disk_cells = mark_disk_cells(msh, R, t, z0)
    if disk_cells.size == 0:
        raise RuntimeError(
            "No disk cells found. Increase mesh resolution or thickness."
        )

    rho_disk = Q / (math.pi * R * R * t)
    rho.x.array[disk_cells] = rho_disk

    bcs = build_bcs(V, msh, Lx, Ly, Lz)

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    uh = problem.solve()
    uh.name = "phi"

    local_min = float(np.min(uh.x.array))
    local_max = float(np.max(uh.x.array))
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print("=== Solve summary ===")
        print(f"disk cells  = {disk_cells.size}")
        print(f"rho_disk    = {rho_disk:.6e} C/m^3")
        print(f"phi min/max = {gmin:.6e}, {gmax:.6e}")

    return uh, rho


def sample_to_regular_grid(msh, uh, Lx, Ly, Lz, nx, ny, nz, outdir: Path):
    """
    Sample phi onto a separate regular Cartesian grid.
    Saves x, y, z arrays and phi_grid as .npz.
    """
    comm = msh.comm

    xs = np.linspace(-0.5 * Lx, 0.5 * Lx, nx)
    ys = np.linspace(-0.5 * Ly, 0.5 * Ly, ny)
    zs = np.linspace(-0.5 * Lz, 0.5 * Lz, nz)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]).astype(np.float64)

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    inside = np.zeros(pts.shape[0], dtype=bool)
    cells = np.full(pts.shape[0], -1, dtype=np.int32)

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            inside[i] = True
            cells[i] = links[0]

    phi_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if np.any(inside):
        phi_vals[inside] = np.asarray(uh.eval(pts[inside], cells[inside])).reshape(-1)

    phi_grid = phi_vals.reshape((nx, ny, nz))

    if comm.rank == 0:
        np.savez(
            outdir / "sampled_regular_grid.npz",
            x=xs, y=ys, z=zs, phi=phi_grid
        )

        iy = ny // 2
        iz = nz // 2
        ix = nx // 2

        np.savetxt(
            outdir / "probe_x_centerline.csv",
            np.column_stack([xs, phi_grid[:, iy, iz]]),
            delimiter=",",
            header="x,phi",
            comments=""
        )
        np.savetxt(
            outdir / "probe_z_centerline.csv",
            np.column_stack([zs, phi_grid[ix, iy, :]]),
            delimiter=",",
            header="z,phi",
            comments=""
        )

        print("Saved sampled_regular_grid.npz")
        print("Saved probe_x_centerline.csv")
        print("Saved probe_z_centerline.csv")


def main():
    ap = argparse.ArgumentParser(
        description="Structured box Poisson solve for a charged thin disk, with separate regular-grid sampling."
    )

    ap.add_argument("--R", type=float, default=None, help="Disk radius in meters")
    ap.add_argument("--Rnm", type=float, default=None, help="Disk radius in nm")
    ap.add_argument("--t", type=float, default=4e-9, help="Disk thickness in meters")
    ap.add_argument("--z0", type=float, default=0.0, help="Disk center z in meters")
    ap.add_argument("--Q", type=float, default=1.602176634e-19, help="Total charge in Coulombs")
    ap.add_argument("--epsr", type=float, default=12.0, help="Relative permittivity")

    ap.add_argument("--Lx", type=float, default=500e-9, help="Box size x in meters")
    ap.add_argument("--Ly", type=float, default=500e-9, help="Box size y in meters")
    ap.add_argument("--Lz", type=float, default=260e-9, help="Box size z in meters")

    ap.add_argument("--Nx", type=int, default=48, help="Number of mesh cells in x")
    ap.add_argument("--Ny", type=int, default=48, help="Number of mesh cells in y")
    ap.add_argument("--Nz", type=int, default=24, help="Number of mesh cells in z")
    ap.add_argument("--celltype", choices=["hex", "tet"], default="hex",
                    help="Structured box cell type")

    ap.add_argument("--sx", type=int, default=81, help="Sample-grid points in x")
    ap.add_argument("--sy", type=int, default=81, help="Sample-grid points in y")
    ap.add_argument("--sz", type=int, default=41, help="Sample-grid points in z")

    ap.add_argument("--out", type=str, default="Results/structured_disk_box", help="Output folder")

    args = ap.parse_args()

    if args.R is None and args.Rnm is None:
        raise ValueError("Provide either --R or --Rnm.")
    if args.R is not None and args.Rnm is not None:
        raise ValueError("Provide only one of --R or --Rnm.")

    if args.R is not None:
        R = args.R
        Rnm = 1e9 * R
    else:
        R = 1e-9 * args.Rnm
        Rnm = args.Rnm

    comm = MPI.COMM_WORLD
    outdir = Path(args.out)
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    msh = make_mesh(
        comm,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        Nx=args.Nx, Ny=args.Ny, Nz=args.Nz,
        cell_type_name=args.celltype,
    )

    if comm.rank == 0:
        print("=== Mesh summary ===")
        print(f"celltype = {args.celltype}")
        print(f"R = {Rnm:.3f} nm")
        print(f"box = ({args.Lx*1e9:.1f}, {args.Ly*1e9:.1f}, {args.Lz*1e9:.1f}) nm")
        print(f"mesh cells = ({args.Nx}, {args.Ny}, {args.Nz})")
        print(f"sample grid = ({args.sx}, {args.sy}, {args.sz})")

    uh, rho = solve_poisson(
        comm, msh,
        R=R, t=args.t, z0=args.z0,
        Q=args.Q, epsr=args.epsr,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz
    )

    with io.XDMFFile(comm, str(outdir / "mesh_phi.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(uh)

    with io.XDMFFile(comm, str(outdir / "mesh_rho.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(rho)

    sample_to_regular_grid(
        msh, uh,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        nx=args.sx, ny=args.sy, nz=args.sz,
        outdir=outdir
    )


if __name__ == "__main__":
    main()
