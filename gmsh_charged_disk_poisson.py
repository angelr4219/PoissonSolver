#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
import ufl

from dolfinx import fem, io, mesh

try:
    from dolfinx.io import gmsh as gmshio
except ImportError:
    from dolfinx.io import gmshio

try:
    import dolfinx_mpc
except ImportError:
    dolfinx_mpc = None


def build_box_mesh_with_disk_refinement(
    comm: MPI.Comm,
    Lx: float, Ly: float, Lz: float,
    R: float, t: float, z0: float,
    n_diam: int,
    refine_band_R: float,
    far_h_factor: float,
    msh_path: Path,
):
    """
    Build only the outer box mesh.
    Refine near the intended thin-disk region using a Box size field.
    No OCC fragment/boolean is used.
    """
    if comm.rank != 0:
        return None

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("charged_disk_box_only")
    occ = gmsh.model.occ

    x0 = -0.5 * Lx
    y0 = -0.5 * Ly
    zmin = -0.5 * Lz
    box = occ.addBox(x0, y0, zmin, Lx, Ly, Lz)
    occ.synchronize()

    # Tag the single bulk volume
    vols = gmsh.model.getEntities(dim=3)
    if len(vols) != 1:
        gmsh.finalize()
        raise RuntimeError(f"Expected one volume, got {vols}")

    bulk_tag = vols[0][1]
    pg_bulk = 1
    gmsh.model.addPhysicalGroup(3, [bulk_tag], pg_bulk)
    gmsh.model.setPhysicalName(3, pg_bulk, "bulk")

    # Mesh sizing
    h_disk = (2.0 * R) / float(n_diam)
    h_far = far_h_factor * h_disk

    # Use a Box refinement field around the intended disk region
    # Slightly larger than the actual disk so the source region is resolved well.
    f_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_box, "VIn", h_disk)
    gmsh.model.mesh.field.setNumber(f_box, "VOut", h_far)

    band_xy = refine_band_R * R
    band_z = max(3.0 * t, 0.5 * refine_band_R * R)

    gmsh.model.mesh.field.setNumber(f_box, "XMin", -band_xy)
    gmsh.model.mesh.field.setNumber(f_box, "XMax",  band_xy)
    gmsh.model.mesh.field.setNumber(f_box, "YMin", -band_xy)
    gmsh.model.mesh.field.setNumber(f_box, "YMax",  band_xy)
    gmsh.model.mesh.field.setNumber(f_box, "ZMin", z0 - 0.5 * band_z)
    gmsh.model.mesh.field.setNumber(f_box, "ZMax", z0 + 0.5 * band_z)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_box)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h_disk)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)

    gmsh.model.mesh.generate(3)
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))

    info = dict(
        h_disk=h_disk,
        h_far=h_far,
        pg_bulk=pg_bulk,
    )
    gmsh.finalize()
    return info


def collect_outer_boundary_facets(msh, Lx, Ly, Lz, periodic):
    fdim = msh.topology.dim - 1
    facet_sets = []

    xh = 0.5 * Lx
    yh = 0.5 * Ly
    zh = 0.5 * Lz

    if "x" not in periodic:
        facet_sets.append(
            mesh.locate_entities_boundary(
                msh, fdim, lambda x: np.isclose(x[0], -xh) | np.isclose(x[0], xh)
            )
        )

    if "y" not in periodic:
        facet_sets.append(
            mesh.locate_entities_boundary(
                msh, fdim, lambda x: np.isclose(x[1], -yh) | np.isclose(x[1], yh)
            )
        )

    facet_sets.append(
        mesh.locate_entities_boundary(
            msh, fdim, lambda x: np.isclose(x[2], -zh) | np.isclose(x[2], zh)
        )
    )

    if len(facet_sets) == 0:
        return np.array([], dtype=np.int32)

    return np.unique(np.hstack(facet_sets)).astype(np.int32)


def build_dirichlet_bcs(V, msh, Lx, Ly, Lz, periodic):
    fdim = msh.topology.dim - 1
    boundary_facets = collect_outer_boundary_facets(msh, Lx, Ly, Lz, periodic)
    if boundary_facets.size == 0:
        return []

    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)
    return [bc]


def build_periodic_mpc(V, bcs, Lx, Ly, periodic):
    if periodic == "none":
        return None

    if dolfinx_mpc is None:
        raise RuntimeError(
            "Periodic BCs requested, but dolfinx_mpc is not installed in this environment."
        )

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    if periodic in ("x", "xy"):
        def indicator_x(x):
            mask = np.isclose(x[0], 0.5 * Lx)
            if periodic == "xy":
                mask = np.logical_and(mask, ~np.isclose(x[1], 0.5 * Ly))
            return mask

        def relation_x(x):
            out = x.copy()
            out[0] -= Lx
            return out

        mpc.create_periodic_constraint_geometrical(V, indicator_x, relation_x, bcs)

    if periodic in ("y", "xy"):
        def indicator_y(x):
            mask = np.isclose(x[1], 0.5 * Ly)
            if periodic == "xy":
                mask = np.logical_and(mask, ~np.isclose(x[0], 0.5 * Lx))
            return mask

        def relation_y(x):
            out = x.copy()
            out[1] -= Ly
            return out

        mpc.create_periodic_constraint_geometrical(V, indicator_y, relation_y, bcs)

    mpc.finalize()
    return mpc


def mark_disk_cells_geometrically(msh, R, t, z0):
    """
    Mark cells whose midpoint lies inside the thin cylinder:
        x^2 + y^2 <= R^2
        |z - z0| <= t/2
    """
    tdim = msh.topology.dim

    disk_cells = mesh.locate_entities(
        msh, tdim,
        lambda x: (
            (x[0] * x[0] + x[1] * x[1] <= R * R)
            & (np.abs(x[2] - z0) <= 0.5 * t)
        )
    )
    return disk_cells.astype(np.int32)


def solve_poisson(
    comm: MPI.Comm,
    msh_path: Path,
    Q: float,
    R: float,
    t: float,
    z0: float,
    epsr: float,
    Lx: float,
    Ly: float,
    Lz: float,
    periodic: str,
    outdir: Path,
):
    msh, cell_tags, facet_tags = gmshio.read_from_msh(str(msh_path), comm, gdim=3)

    eps0 = 8.8541878128e-12
    eps = eps0 * epsr

    V = fem.functionspace(msh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Constant rho inside geometric disk region
    rho_disk = Q / (math.pi * R * R * t)

    Q0 = fem.functionspace(msh, ("DG", 0))
    rho = fem.Function(Q0)
    rho.x.array[:] = 0.0

    disk_cells = mark_disk_cells_geometrically(msh, R, t, z0)
    if disk_cells.size == 0:
        raise RuntimeError(
            "No disk cells were found geometrically. "
            "Increase refinement near the disk or increase n_diam."
        )

    rho.x.array[disk_cells] = rho_disk

    bcs = build_dirichlet_bcs(V, msh, Lx, Ly, Lz, periodic)
    mpc = build_periodic_mpc(V, bcs, Lx, Ly, periodic)

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
        "ksp_max_it": 2000,
    }

    if mpc is None:
        problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
        uh = problem.solve()
    else:
        problem = dolfinx_mpc.LinearProblem(a, L, mpc, bcs=bcs, petsc_options=petsc_options)
        uh = problem.solve()
        mpc.backsubstitution(uh)

    local_min = float(np.min(uh.x.array))
    local_max = float(np.max(uh.x.array))
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print("=== Solve summary ===")
        print(f"periodic = {periodic}")
        print(f"phi min/max = {gmin:.6e}, {gmax:.6e}")
        print(f"rho_disk    = {rho_disk:.6e} C/m^3")
        print(f"disk cells  = {disk_cells.size}")

    outdir.mkdir(parents=True, exist_ok=True)
    uh.name = "phi"
    rho.name = "rho"

    with io.XDMFFile(comm, str(outdir / "mesh_phi.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(uh)

    with io.XDMFFile(comm, str(outdir / "mesh_rho.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(rho)

    return uh


def main():
    ap = argparse.ArgumentParser(
        description="Charged disk Poisson solve in a box with optional x/y periodic MPC."
    )

    ap.add_argument("--R", type=float, default=None, help="Disk radius in meters")
    ap.add_argument("--Rnm", type=float, default=None, help="Disk radius in nm")

    ap.add_argument("--t", type=float, default=4e-9, help="Disk thickness in meters")
    ap.add_argument("--Q", type=float, default=1.602176634e-19, help="Total charge in Coulombs")
    ap.add_argument("--z0", type=float, default=0.0, help="Disk center z location in meters")
    ap.add_argument("--Lx", type=float, default=600e-9, help="Domain size x in meters")
    ap.add_argument("--Ly", type=float, default=600e-9, help="Domain size y in meters")
    ap.add_argument("--Lz", type=float, default=400e-9, help="Domain size z in meters")
    ap.add_argument("--epsr", type=float, default=12.0, help="Relative permittivity")

    ap.add_argument("--n_diam", type=int, default=24,
                    help="Target elements across disk diameter")
    ap.add_argument("--refine_band_R", type=float, default=6.0,
                    help="Refine band thickness in units of R")
    ap.add_argument("--far_h_factor", type=float, default=20.0,
                    help="Far-field size factor relative to h_disk")

    ap.add_argument("--periodic", choices=["none", "x", "y", "xy"], default="none",
                    help="Optional periodic BCs in x and/or y via dolfinx_mpc")
    ap.add_argument("--out", type=str, default="out_disk_refined", help="Output folder")

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
    msh_path = outdir / "mesh.msh"

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    info = build_box_mesh_with_disk_refinement(
        comm=comm,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        R=R, t=args.t, z0=args.z0,
        n_diam=args.n_diam,
        refine_band_R=args.refine_band_R,
        far_h_factor=args.far_h_factor,
        msh_path=msh_path,
    )

    if comm.rank == 0:
        print("=== Mesh sizing summary ===")
        print(f"R = {Rnm:.3f} nm")
        print(f"t = {args.t:.3e} m")
        print(f"n_diam = {args.n_diam} -> h_disk = {info['h_disk']:.3e} m ({info['h_disk']*1e9:.3f} nm)")
        print(f"h_far  = {info['h_far']:.3e} m ({info['h_far']*1e9:.3f} nm)")
        print(f"periodic = {args.periodic}")

    solve_poisson(
        comm=comm,
        msh_path=msh_path,
        Q=args.Q,
        R=R,
        t=args.t,
        z0=args.z0,
        epsr=args.epsr,
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        periodic=args.periodic,
        outdir=outdir,
    )


if __name__ == "__main__":
    main()
