#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, io, mesh
from dolfinx.mesh import CellType
from dolfinx.fem.petsc import LinearProblem


def make_mesh(comm, Lx, Ly, Lz, Nx, Ny, Nz, cell_type_name):
    if cell_type_name == "hex":
        cell_type = CellType.hexahedron
    elif cell_type_name == "tet":
        cell_type = CellType.tetrahedron
    else:
        raise ValueError(f"Unknown cell type: {cell_type_name}")

    return mesh.create_box(
        comm,
        [[-0.5 * Lx, -0.5 * Ly, -0.5 * Lz],
         [ 0.5 * Lx,  0.5 * Ly,  0.5 * Lz]],
        [Nx, Ny, Nz],
        cell_type=cell_type,
    )


def locate_disk_gate_facets(msh, R, Lz):
    fdim = msh.topology.dim - 1
    zh = 0.5 * Lz
    facets = mesh.locate_entities_boundary(
        msh,
        fdim,
        lambda x: np.isclose(x[2], zh) & ((x[0] * x[0] + x[1] * x[1]) <= R * R),
    )
    return np.asarray(facets, dtype=np.int32)


def locate_other_outer_facets(msh, R, Lx, Ly, Lz):
    fdim = msh.topology.dim - 1
    xh = 0.5 * Lx
    yh = 0.5 * Ly
    zh = 0.5 * Lz

    facets = mesh.locate_entities_boundary(
        msh,
        fdim,
        lambda x: (
            np.isclose(x[0], -xh) | np.isclose(x[0], xh)
            | np.isclose(x[1], -yh) | np.isclose(x[1], yh)
            | np.isclose(x[2], -zh)
            | (np.isclose(x[2], zh) & ((x[0] * x[0] + x[1] * x[1]) > R * R))
        ),
    )
    return np.asarray(facets, dtype=np.int32)


def build_bcs(V, msh, R, Vdisk, Lx, Ly, Lz):
    fdim = msh.topology.dim - 1

    disk_facets = locate_disk_gate_facets(msh, R, Lz)
    other_facets = locate_other_outer_facets(msh, R, Lx, Ly, Lz)

    if disk_facets.size == 0:
        raise RuntimeError(
            "No disk gate facets found. Increase mesh resolution or enlarge the gate radius."
        )

    disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)
    other_dofs = fem.locate_dofs_topological(V, fdim, other_facets)

    bc_disk = fem.dirichletbc(PETSc.ScalarType(Vdisk), disk_dofs, V)
    bc_outer = fem.dirichletbc(PETSc.ScalarType(0.0), other_dofs, V)

    return [bc_disk, bc_outer], disk_facets, other_facets


def build_material_fields(msh, epsr_top, epsr_bottom, z_interface):
    """
    DG0 fields on cells:
      eps_r : relative permittivity
      rho   : charge density, zero for this Laplace gate case
    """
    Q0 = fem.functionspace(msh, ("DG", 0))

    eps_r = fem.Function(Q0)
    eps_r.name = "eps_r"

    rho = fem.Function(Q0)
    rho.name = "rho"
    rho.x.array[:] = 0.0

    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    mids = mesh.compute_midpoints(msh, tdim, cells)

    zmid = mids[:, 2]
    top_mask = zmid > z_interface
    bot_mask = ~top_mask

    vals = np.empty(num_cells, dtype=np.float64)
    vals[top_mask] = epsr_top
    vals[bot_mask] = epsr_bottom
    eps_r.x.array[:] = vals

    return eps_r, rho


def solve_laplace_gate(comm, msh, R, Vdisk, epsr_top, epsr_bottom, z_interface, Lx, Ly, Lz):
    eps0 = 8.8541878128e-12

    V = fem.functionspace(msh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    bcs, disk_facets, other_facets = build_bcs(V, msh, R, Vdisk, Lx, Ly, Lz)
    eps_r, rho = build_material_fields(msh, epsr_top, epsr_bottom, z_interface)

    a = ufl.inner(eps0 * eps_r * ufl.grad(u), ufl.grad(v)) * ufl.dx

    zero = fem.Constant(msh, PETSc.ScalarType(0.0))
    L = zero * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="disk_gate_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
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
        print(f"disk gate voltage  = {Vdisk:.6f} V")
        print(f"disk facets        = {disk_facets.size}")
        print(f"other outer facets = {other_facets.size}")
        print(f"phi min/max        = {gmin:.6e}, {gmax:.6e}")
        print(f"epsr_top           = {epsr_top}")
        print(f"epsr_bottom        = {epsr_bottom}")
        print(f"z_interface        = {z_interface:.6e} m")

    return uh, rho, eps_r


def write_single_output_file(comm, msh, phi, rho, eps_r, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    with io.XDMFFile(comm, str(outdir / "fields.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(eps_r)


def main():
    ap = argparse.ArgumentParser(
        description="Structured box Laplace solve with a circular top-surface disk gate and one XDMF output containing phi, rho, eps_r."
    )

    ap.add_argument("--R", type=float, default=None, help="Disk gate radius in meters")
    ap.add_argument("--Rnm", type=float, default=None, help="Disk gate radius in nm")
    ap.add_argument("--Vdisk", type=float, required=True, help="Disk gate voltage in volts")

    ap.add_argument("--Lx", type=float, default=500e-9, help="Box size x in meters")
    ap.add_argument("--Ly", type=float, default=500e-9, help="Box size y in meters")
    ap.add_argument("--Lz", type=float, default=260e-9, help="Box size z in meters")

    ap.add_argument("--Nx", type=int, default=120, help="Number of mesh cells in x")
    ap.add_argument("--Ny", type=int, default=120, help="Number of mesh cells in y")
    ap.add_argument("--Nz", type=int, default=80, help="Number of mesh cells in z")
    ap.add_argument("--celltype", choices=["hex", "tet"], default="hex",
                    help="Structured box cell type")

    ap.add_argument("--epsr_top", type=float, default=3.9,
                    help="Relative permittivity for z > z_interface")
    ap.add_argument("--epsr_bottom", type=float, default=11.7,
                    help="Relative permittivity for z <= z_interface")
    ap.add_argument("--z_interface", type=float, default=0.0,
                    help="Dielectric interface location in meters")

    ap.add_argument("--out", type=str, required=True, help="Output folder")

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

    msh = make_mesh(
        comm,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        Nx=args.Nx, Ny=args.Ny, Nz=args.Nz,
        cell_type_name=args.celltype,
    )

    if comm.rank == 0:
        print("=== Mesh summary ===")
        print(f"celltype     = {args.celltype}")
        print(f"R            = {Rnm:.3f} nm")
        print(f"Vdisk        = {args.Vdisk:.6f} V")
        print(f"box          = ({args.Lx*1e9:.1f}, {args.Ly*1e9:.1f}, {args.Lz*1e9:.1f}) nm")
        print(f"mesh cells   = ({args.Nx}, {args.Ny}, {args.Nz})")
        print(f"epsr_top     = {args.epsr_top}")
        print(f"epsr_bottom  = {args.epsr_bottom}")
        print(f"z_interface  = {args.z_interface:.6e} m")

    phi, rho, eps_r = solve_laplace_gate(
        comm, msh,
        R=R,
        Vdisk=args.Vdisk,
        epsr_top=args.epsr_top,
        epsr_bottom=args.epsr_bottom,
        z_interface=args.z_interface,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
    )

    write_single_output_file(comm, msh, phi, rho, eps_r, outdir)


if __name__ == "__main__":
    main()
