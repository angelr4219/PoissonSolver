#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem


def create_box_mesh(comm, Lx, Ly, Lz, Nx, Ny, Nz, celltype_str):
    if celltype_str == "hex":
        celltype = mesh.CellType.hexahedron
    elif celltype_str == "tet":
        celltype = mesh.CellType.tetrahedron
    else:
        raise ValueError(f"Unknown celltype: {celltype_str}")

    return mesh.create_box(
        comm,
        points=[np.array([-Lx / 2, -Ly / 2, 0.0]), np.array([Lx / 2, Ly / 2, Lz])],
        n=[Nx, Ny, Nz],
        cell_type=celltype,
    )


def build_cell_fields(msh, z_sige_top, z_si_top, epsr_sige, epsr_si):
    """
    Build DG0 cellwise fields:
      - eps_r
      - mat_id
      - rho
    Layer model:
      0 <= z < z_sige_top         : SiGe
      z_sige_top <= z < z_si_top  : Si
      z_si_top <= z <= Lz         : SiGe
    """
    tdim = msh.topology.dim
    num_cells_local = msh.topology.index_map(tdim).size_local

    Q0 = fem.functionspace(msh, ("DG", 0))

    eps_r = fem.Function(Q0)
    eps_r.name = "eps_r"

    mat_id = fem.Function(Q0)
    mat_id.name = "mat_id"

    rho = fem.Function(Q0)
    rho.name = "rho"

    # Cell centers from DG0 dof coordinates
    cell_centers = Q0.tabulate_dof_coordinates().reshape(num_cells_local, msh.geometry.dim)
    zc = cell_centers[:, 2]

    sige_mask = (zc < z_sige_top) | (zc >= z_si_top)
    si_mask = (zc >= z_sige_top) & (zc < z_si_top)

    eps_vals = np.empty(num_cells_local, dtype=np.float64)
    eps_vals[sige_mask] = epsr_sige
    eps_vals[si_mask] = epsr_si

    mat_vals = np.empty(num_cells_local, dtype=np.float64)
    mat_vals[sige_mask] = 1.0  # SiGe
    mat_vals[si_mask] = 2.0    # Si

    rho_vals = np.zeros(num_cells_local, dtype=np.float64)

    eps_r.x.array[:] = eps_vals
    mat_id.x.array[:] = mat_vals
    rho.x.array[:] = rho_vals

    return Q0, eps_r, mat_id, rho


def build_gate_mask(msh, R, Lz):
    """
    Build a DG0 volume indicator that marks cells near the top surface and
    under the circular disk-gate footprint.
    This is not the exact boundary facet tag, but it is very useful in ParaView.
    """
    Q0 = fem.functionspace(msh, ("DG", 0))
    gate_mask = fem.Function(Q0)
    gate_mask.name = "gate_mask"

    num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
    cell_centers = Q0.tabulate_dof_coordinates().reshape(num_cells_local, msh.geometry.dim)

    xc = cell_centers[:, 0]
    yc = cell_centers[:, 1]
    zc = cell_centers[:, 2]

    # mark cells whose centers are near the top and inside disk projection
    r2 = xc**2 + yc**2
    top_band = zc > (Lz - 0.10 * Lz)   # top 10% of box
    disk_xy = r2 <= R**2

    vals = np.zeros(num_cells_local, dtype=np.float64)
    vals[top_band & disk_xy] = 1.0
    gate_mask.x.array[:] = vals
    return gate_mask


def locate_boundary_sets(msh, R, Lz):
    """
    Return facet sets:
      - disk_facets: top boundary facets inside disk
      - top_free_facets: remaining top boundary facets
      - bottom_facets
      - side_facets
    """
    tdim = msh.topology.dim
    fdim = tdim - 1

    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)

    all_bfacets = mesh.exterior_facet_indices(msh.topology)
    x = mesh.compute_midpoints(msh, fdim, all_bfacets)

    z = x[:, 2]
    r2 = x[:, 0] ** 2 + x[:, 1] ** 2

    tol_z = 1e-12 * max(1.0, Lz)

    top_mask = np.isclose(z, Lz, atol=tol_z)
    bottom_mask = np.isclose(z, 0.0, atol=tol_z)
    side_mask = ~(top_mask | bottom_mask)

    disk_mask = top_mask & (r2 <= R**2)
    top_free_mask = top_mask & (~disk_mask)

    disk_facets = all_bfacets[disk_mask]
    top_free_facets = all_bfacets[top_free_mask]
    bottom_facets = all_bfacets[bottom_mask]
    side_facets = all_bfacets[side_mask]

    if disk_facets.size == 0:
        raise RuntimeError("No top disk-gate facets found. Increase lateral mesh resolution or enlarge R.")

    return disk_facets, top_free_facets, bottom_facets, side_facets


def solve_case(
    msh,
    R,
    Vgate,
    Vbottom,
    z_sige_top,
    z_si_top,
    epsr_sige,
    epsr_si,
    sigma_top_cm2,
):
    """
    Solve:
      div(eps_r * grad(phi)) = 0
    with:
      phi = Vgate on top disk patch
      phi = Vbottom on bottom
    and a surrogate free-top surface charge applied as a boundary term on the rest of top.
    """
    comm = msh.comm
    tdim = msh.topology.dim
    fdim = tdim - 1
    Lz = msh.geometry.x[:, 2].max()

    # Cell fields
    Q0, eps_r, mat_id, rho = build_cell_fields(msh, z_sige_top, z_si_top, epsr_sige, epsr_si)
    gate_mask = build_gate_mask(msh, R, Lz)

    # Facets
    disk_facets, top_free_facets, bottom_facets, side_facets = locate_boundary_sets(msh, R, Lz)

    # Function space for phi
    V = fem.functionspace(msh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Permittivity
    eps0 = 8.8541878128e-12
    eps = eps0 * eps_r

    # Measures
    facet_values = np.hstack([
        np.full(len(disk_facets), 1, dtype=np.int32),
        np.full(len(top_free_facets), 2, dtype=np.int32),
        np.full(len(bottom_facets), 3, dtype=np.int32),
        np.full(len(side_facets), 4, dtype=np.int32),
    ])
    facet_indices = np.hstack([disk_facets, top_free_facets, bottom_facets, side_facets])

    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(msh, fdim, facet_indices[order], facet_values[order])

    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=msh)

    # Surface charge density on free top
    sigma_top = sigma_top_cm2 * 1.0e4 * 1.602176634e-19  # cm^-2 -> m^-2, then e -> C
    # sign carried by user input

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * dx
    L = rho * v * dx + sigma_top * v * ds(2)

    # Dirichlet BCs
    bc_disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)
    bc_bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)

    bc_disk = fem.dirichletbc(PETSc.ScalarType(Vgate), bc_disk_dofs, V)
    bc_bottom = fem.dirichletbc(PETSc.ScalarType(Vbottom), bc_bottom_dofs, V)

    print("Starting linear solve...")
    problem = LinearProblem(
        a,
        L,
        bcs=[bc_disk, bc_bottom],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-10,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 2000,
            "ksp_error_if_not_converged": True,
        },
    )
    phi = problem.solve()
    phi.name = "phi"

    # diagnostics
    local_min = np.min(phi.x.array)
    local_max = np.max(phi.x.array)
    gmin = comm.allreduce(float(local_min), op=MPI.MIN)
    gmax = comm.allreduce(float(local_max), op=MPI.MAX)

    if comm.rank == 0:
        print("=== Solve summary ===")
        print(f"Vgate           = {Vgate:.6f} V")
        print(f"Vbottom         = {Vbottom:.6f} V")
        print(f"sigma_top_cm2   = {sigma_top_cm2:.6e} cm^-2")
        print(f"disk facets     = {len(disk_facets)}")
        print(f"top free facets = {len(top_free_facets)}")
        print(f"bottom facets   = {len(bottom_facets)}")
        print(f"side facets     = {len(side_facets)}")
        print(f"phi min/max     = {gmin:.6e}, {gmax:.6e}")

    return phi, rho, eps_r, mat_id, gate_mask, facet_tags


def write_all_fields(outdir, msh, phi, rho, eps_r, mat_id, gate_mask):
    """
    Write one robust XDMF/H5 pair:
      - mesh
      - phi
      - rho
      - eps_r
      - mat_id
      - gate_mask
    """
    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / "fields.xdmf"

    print(f"Writing XDMF output to {outdir} ...")
    with io.XDMFFile(msh.comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(eps_r)
        xdmf.write_function(mat_id)
        xdmf.write_function(gate_mask)

    if msh.comm.rank == 0:
        print("Finished writing:")
        print(f"  {outdir / 'fields.xdmf'}")
        print(f"  {outdir / 'fields.h5'}")


def main():
    ap = argparse.ArgumentParser(description="Leah-like disk-gate benchmark with robust XDMF writing.")
    ap.add_argument("--R", type=float, default=None, help="Disk radius in meters")
    ap.add_argument("--Rnm", type=float, default=None, help="Disk radius in nm")
    ap.add_argument("--Vgate", type=float, required=True)
    ap.add_argument("--Vbottom", type=float, default=12.0)

    ap.add_argument("--Lx", type=float, default=500e-9)
    ap.add_argument("--Ly", type=float, default=500e-9)
    ap.add_argument("--Lz", type=float, default=255e-9)

    ap.add_argument("--Nx", type=int, default=60)
    ap.add_argument("--Ny", type=int, default=60)
    ap.add_argument("--Nz", type=int, default=48)

    ap.add_argument("--celltype", choices=["hex", "tet"], default="hex")

    ap.add_argument("--z_sige_top", type=float, default=50e-9)
    ap.add_argument("--z_si_top", type=float, default=55e-9)

    ap.add_argument("--epsr_sige", type=float, default=12.0)
    ap.add_argument("--epsr_si", type=float, default=11.7)

    ap.add_argument("--sigma_top_cm2", type=float, default=-2.0e11)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.R is None and args.Rnm is None:
        raise ValueError("Provide either --R (meters) or --Rnm (nm).")
    if args.R is not None and args.Rnm is not None:
        raise ValueError("Provide only one of --R or --Rnm.")

    R = args.R if args.R is not None else args.Rnm * 1.0e-9

    comm = MPI.COMM_WORLD
    outdir = Path(args.out)

    msh = create_box_mesh(
        comm,
        args.Lx, args.Ly, args.Lz,
        args.Nx, args.Ny, args.Nz,
        args.celltype,
    )

    if comm.rank == 0:
        print("=== Mesh / case summary ===")
        print(f"celltype        = {args.celltype}")
        print(f"R               = {R * 1e9:.3f} nm")
        print(f"Vgate           = {args.Vgate:.6f}")
        print(f"Vbottom         = {args.Vbottom:.6f}")
        print(f"box             = ({args.Lx*1e9:.1f}, {args.Ly*1e9:.1f}, {args.Lz*1e9:.1f}) nm")
        print(f"mesh cells      = ({args.Nx}, {args.Ny}, {args.Nz})")
        print(f"sigma_top_cm2   = {args.sigma_top_cm2:.6e}")
        print(f"outdir          = {outdir}")

    phi, rho, eps_r, mat_id, gate_mask, facet_tags = solve_case(
        msh=msh,
        R=R,
        Vgate=args.Vgate,
        Vbottom=args.Vbottom,
        z_sige_top=args.z_sige_top,
        z_si_top=args.z_si_top,
        epsr_sige=args.epsr_sige,
        epsr_si=args.epsr_si,
        sigma_top_cm2=args.sigma_top_cm2,
    )

    write_all_fields(outdir, msh, phi, rho, eps_r, mat_id, gate_mask)


if __name__ == "__main__":
    main()
