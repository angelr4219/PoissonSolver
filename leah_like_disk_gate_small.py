#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, io, mesh
from dolfinx.mesh import CellType, meshtags
from dolfinx.fem.petsc import LinearProblem


# -----------------------------
# Fixed small-case parameters
# -----------------------------
R = 10e-9
Vgate = 1.0
Vbottom = 12.0

Lx = 500e-9
Ly = 500e-9
Lz = 255e-9

Nx = 60
Ny = 60
Nz = 48

celltype_name = "hex"

z_sige_top = 50e-9
z_si_top = 55e-9
epsr_sige = 12.0
epsr_si = 11.7

sigma_top_cm2 = -2.0e11

outdir = Path("Results/disk_1v_12v_small")


def make_mesh(comm):
    if celltype_name == "hex":
        celltype = CellType.hexahedron
    elif celltype_name == "tet":
        celltype = CellType.tetrahedron
    else:
        raise ValueError(f"Unknown cell type {celltype_name}")

    return mesh.create_box(
        comm,
        [[-0.5 * Lx, -0.5 * Ly, 0.0],
         [ 0.5 * Lx,  0.5 * Ly, Lz]],
        [Nx, Ny, Nz],
        cell_type=celltype,
    )


def build_cell_fields_dg0(msh):
    Q0 = fem.functionspace(msh, ("DG", 0))

    eps_r = fem.Function(Q0)
    eps_r.name = "eps_r_dg0"

    rho = fem.Function(Q0)
    rho.name = "rho_dg0"
    rho.x.array[:] = 0.0

    mat_id = fem.Function(Q0)
    mat_id.name = "mat_id_dg0"

    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    mids = mesh.compute_midpoints(msh, tdim, cells)
    zmid = mids[:, 2]

    is_si = (zmid > z_sige_top) & (zmid <= z_si_top)
    is_sige = ~is_si

    eps_vals = np.empty(num_cells, dtype=np.float64)
    eps_vals[is_sige] = epsr_sige
    eps_vals[is_si] = epsr_si
    eps_r.x.array[:] = eps_vals

    mat_vals = np.empty(num_cells, dtype=np.float64)
    mat_vals[is_sige] = 1.0
    mat_vals[is_si] = 2.0
    mat_id.x.array[:] = mat_vals

    return eps_r, rho, mat_id


def locate_boundary_sets(msh):
    tdim = msh.topology.dim
    fdim = tdim - 1

    msh.topology.create_connectivity(fdim, tdim)
    all_bfacets = mesh.exterior_facet_indices(msh.topology)
    mids = mesh.compute_midpoints(msh, fdim, all_bfacets)

    x = mids[:, 0]
    y = mids[:, 1]
    z = mids[:, 2]
    r2 = x * x + y * y

    top_mask = np.isclose(z, 0.0)
    bot_mask = np.isclose(z, Lz)

    disk_mask = top_mask & (r2 <= R * R)
    top_free_mask = top_mask & (r2 > R * R)
    bottom_mask = bot_mask
    side_mask = ~(disk_mask | top_free_mask | bottom_mask)

    disk_facets = all_bfacets[disk_mask].astype(np.int32)
    top_free_facets = all_bfacets[top_free_mask].astype(np.int32)
    bottom_facets = all_bfacets[bottom_mask].astype(np.int32)
    side_facets = all_bfacets[side_mask].astype(np.int32)

    if disk_facets.size == 0:
        raise RuntimeError("No disk gate facets found.")

    facet_indices = np.hstack([disk_facets, top_free_facets, bottom_facets, side_facets])
    facet_values = np.hstack([
        np.full(disk_facets.size, 1, dtype=np.int32),
        np.full(top_free_facets.size, 2, dtype=np.int32),
        np.full(bottom_facets.size, 3, dtype=np.int32),
        np.full(side_facets.size, 4, dtype=np.int32),
    ])

    order = np.argsort(facet_indices)
    facet_tags = meshtags(msh, fdim, facet_indices[order], facet_values[order])

    return facet_tags, disk_facets, top_free_facets, bottom_facets, side_facets


def build_bcs(V, facet_tags):
    fdim = V.mesh.topology.dim - 1

    disk_facets = facet_tags.find(1)
    bottom_facets = facet_tags.find(3)

    disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)

    bc_gate = fem.dirichletbc(PETSc.ScalarType(Vgate), disk_dofs, V)
    bc_bottom = fem.dirichletbc(PETSc.ScalarType(Vbottom), bottom_dofs, V)
    return [bc_gate, bc_bottom]


def interpolate_to_cg1(msh, f_in, name):
    V1 = fem.functionspace(msh, ("CG", 1))
    f_out = fem.Function(V1)
    f_out.name = name
    f_out.interpolate(f_in)
    return f_out


def solve_case(comm, msh):
    eps0 = 8.8541878128e-12
    sigma_top = sigma_top_cm2 * 1.0e4 * 1.602176634e-19

    V = fem.functionspace(msh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_r_dg0, rho_dg0, mat_id_dg0 = build_cell_fields_dg0(msh)

    facet_tags, disk_facets, top_free_facets, bottom_facets, side_facets = locate_boundary_sets(msh)
    bcs = build_bcs(V, facet_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    a = ufl.inner(eps0 * eps_r_dg0 * ufl.grad(u), ufl.grad(v)) * ufl.dx
    zero = fem.Constant(msh, PETSc.ScalarType(0.0))
    L = zero * v * ufl.dx + PETSc.ScalarType(sigma_top) * v * ds(2)

    if comm.rank == 0:
        print("Starting linear solve...")

    phi = fem.Function(V)
    phi.name = "phi"

    problem = LinearProblem(
        a,
        L,
        u=phi,
        bcs=bcs,
        petsc_options_prefix="leah_like_small_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
            "ksp_error_if_not_converged": True,
        },
    )
    problem.solve()

    local_min = float(np.min(phi.x.array))
    local_max = float(np.max(phi.x.array))
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print("=== Solve summary ===")
        print(f"disk facets     = {disk_facets.size}")
        print(f"top free facets = {top_free_facets.size}")
        print(f"bottom facets   = {bottom_facets.size}")
        print(f"side facets     = {side_facets.size}")
        print(f"phi min/max     = {gmin:.6e}, {gmax:.6e}")

    rho = interpolate_to_cg1(msh, rho_dg0, "rho")
    eps_r = interpolate_to_cg1(msh, eps_r_dg0, "eps_r")
    mat_id = interpolate_to_cg1(msh, mat_id_dg0, "mat_id")

    return phi, rho, eps_r, mat_id


def write_fields_xdmf(comm, msh, phi, rho, eps_r, mat_id):
    if comm.rank == 0:
        print(f"Writing XDMF output to {outdir} ...")
        outdir.mkdir(parents=True, exist_ok=True)

    comm.barrier()

    with io.XDMFFile(comm, str(outdir / "fields.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(eps_r)
        xdmf.write_function(mat_id)

    comm.barrier()

    if comm.rank == 0:
        print("Finished writing:")
        print(f"  {outdir / 'fields.xdmf'}")
        print(f"  {outdir / 'fields.h5'}")


def main():
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        print("=== Mesh / case summary ===")
        print(f"celltype        = {celltype_name}")
        print(f"R               = {R*1e9:.3f} nm")
        print(f"Vgate           = {Vgate:.6f}")
        print(f"Vbottom         = {Vbottom:.6f}")
        print(f"box             = ({Lx*1e9:.1f}, {Ly*1e9:.1f}, {Lz*1e9:.1f}) nm")
        print(f"mesh cells      = ({Nx}, {Ny}, {Nz})")
        print(f"sigma_top_cm2   = {sigma_top_cm2:.6e}")
        print(f"outdir          = {outdir}")

    msh = make_mesh(comm)
    phi, rho, eps_r, mat_id = solve_case(comm, msh)
    write_fields_xdmf(comm, msh, phi, rho, eps_r, mat_id)


if __name__ == "__main__":
    main()
