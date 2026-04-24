#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

from poisson_output_layer import XDMFRunWriter, print_written_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Demo of canonical DOLFINx output layer for phi, eps, rho, and tags."
    )
    p.add_argument("--outdir", type=str, default="Results/demo_output_layer")
    p.add_argument("--basename", type=str, default="demo_box")
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=12)
    p.add_argument("--nz", type=int, default=10)
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--eps_r", type=float, default=11.7)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--phi_left", type=float, default=1.0)
    p.add_argument("--phi_right", type=float, default=0.0)
    p.add_argument("--pc_type", type=str, default="hypre")
    p.add_argument("--ksp_type", type=str, default="cg")
    return p.parse_args()


def make_facet_tags(domain: mesh.Mesh):
    """
    Build simple facet tags for x_min, x_max, y_min, y_max, z_min, z_max.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(fdim, tdim)

    Lx = 1.0
    Ly = 1.0
    Lz = 1.0
    tol = 1e-12

    def x_min(x):
        return np.isclose(x[0], 0.0, atol=tol)

    def x_max(x):
        return np.isclose(x[0], Lx, atol=tol)

    def y_min(x):
        return np.isclose(x[1], 0.0, atol=tol)

    def y_max(x):
        return np.isclose(x[1], Ly, atol=tol)

    def z_min(x):
        return np.isclose(x[2], 0.0, atol=tol)

    def z_max(x):
        return np.isclose(x[2], Lz, atol=tol)

    tag_map = {
        "x_min": 1,
        "x_max": 2,
        "y_min": 3,
        "y_max": 4,
        "z_min": 5,
        "z_max": 6,
    }

    facet_sets = [
        (tag_map["x_min"], mesh.locate_entities_boundary(domain, fdim, x_min)),
        (tag_map["x_max"], mesh.locate_entities_boundary(domain, fdim, x_max)),
        (tag_map["y_min"], mesh.locate_entities_boundary(domain, fdim, y_min)),
        (tag_map["y_max"], mesh.locate_entities_boundary(domain, fdim, y_max)),
        (tag_map["z_min"], mesh.locate_entities_boundary(domain, fdim, z_min)),
        (tag_map["z_max"], mesh.locate_entities_boundary(domain, fdim, z_max)),
    ]

    facets = np.hstack([f for _, f in facet_sets]).astype(np.int32)
    values = np.hstack([np.full(len(f), tag, dtype=np.int32) for tag, f in facet_sets])

    order = np.argsort(facets)
    facet_tags = mesh.meshtags(domain, fdim, facets[order], values[order])
    facet_tags.name = "facet_tags"
    return facet_tags, tag_map


def make_cell_tags(domain: mesh.Mesh):
    """
    Single-region cell tag for demo purposes.
    """
    tdim = domain.topology.dim
    num_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_local, dtype=np.int32)
    values = np.full(num_local, 1, dtype=np.int32)
    cell_tags = mesh.meshtags(domain, tdim, cells, values)
    cell_tags.name = "cell_tags"
    tag_map = {"bulk": 1}
    return cell_tags, tag_map


def build_problem(args: argparse.Namespace):
    comm = MPI.COMM_WORLD
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [args.nx, args.ny, args.nz],
        cell_type=mesh.CellType.tetrahedron,
    )

    V = fem.functionspace(domain, ("Lagrange", args.deg))

    facet_tags, facet_tag_map = make_facet_tags(domain)
    cell_tags, cell_tag_map = make_cell_tags(domain)

    fdim = domain.topology.dim - 1
    left_facets = facet_tags.find(facet_tag_map["x_min"])
    right_facets = facet_tags.find(facet_tag_map["x_max"])

    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)

    bc_left = fem.dirichletbc(PETSc.ScalarType(args.phi_left), left_dofs, V)
    bc_right = fem.dirichletbc(PETSc.ScalarType(args.phi_right), right_dofs, V)
    bcs = [bc_left, bc_right]

    V0 = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(V0, name="eps")
    eps.x.array[:] = args.eps_r
    eps.x.scatter_forward()

    rho = fem.Function(V0, name="rho")
    rho.x.array[:] = args.rho
    rho.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    # This stable DOLFINx image requires petsc_options_prefix as a keyword-only argument.
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"{args.basename}_",
        petsc_options={
            "ksp_type": args.ksp_type,
            "pc_type": args.pc_type,
            "ksp_rtol": 1.0e-10,
            "ksp_atol": 1.0e-12,
            "ksp_max_it": 5000,
        },
    )
    phi = problem.solve()
    phi.name = "phi"

    return domain, phi, eps, rho, cell_tags, facet_tags, cell_tag_map, facet_tag_map


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD

    domain, phi, eps, rho, cell_tags, facet_tags, cell_tag_map, facet_tag_map = build_problem(args)

    phi_local = phi.x.array
    eps_local = eps.x.array
    rho_local = rho.x.array

    phi_min = comm.allreduce(float(np.min(phi_local)) if phi_local.size else np.inf, op=MPI.MIN)
    phi_max = comm.allreduce(float(np.max(phi_local)) if phi_local.size else -np.inf, op=MPI.MAX)
    eps_min = comm.allreduce(float(np.min(eps_local)) if eps_local.size else np.inf, op=MPI.MIN)
    eps_max = comm.allreduce(float(np.max(eps_local)) if eps_local.size else -np.inf, op=MPI.MAX)
    rho_min = comm.allreduce(float(np.min(rho_local)) if rho_local.size else np.inf, op=MPI.MIN)
    rho_max = comm.allreduce(float(np.max(rho_local)) if rho_local.size else -np.inf, op=MPI.MAX)

    if comm.rank == 0:
        print("\n=== Solver diagnostics ===")
        print(f"phi min/max : {phi_min:.12e}  {phi_max:.12e}")
        print(f"eps min/max : {eps_min:.12e}  {eps_max:.12e}")
        print(f"rho min/max : {rho_min:.12e}  {rho_max:.12e}")
        print("BCs:")
        print(f"  x_min -> phi = {args.phi_left}")
        print(f"  x_max -> phi = {args.phi_right}")
        print("Facet tag map:")
        for k, v in facet_tag_map.items():
            print(f"  {k}: {v}")
        print("Cell tag map:")
        for k, v in cell_tag_map.items():
            print(f"  {k}: {v}")

    writer = XDMFRunWriter(args.outdir, args.basename)
    written = writer.write_fields(phi=phi, eps=eps, rho=rho, force_phi_cg1=True)
    writer.write_meshtags(domain, cell_tags=cell_tags, facet_tags=facet_tags)

    cfg = {
        "case_name": args.basename,
        "geometry": {
            "type": "unit_box",
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
        },
        "solver": {
            "degree": args.deg,
            "ksp_type": args.ksp_type,
            "pc_type": args.pc_type,
            "petsc_options_prefix": f"{args.basename}_",
        },
        "materials": {
            "eps_r": args.eps_r,
            "rho": args.rho,
        },
        "bcs": {
            "x_min": args.phi_left,
            "x_max": args.phi_right,
        },
        "tag_maps": {
            "cell_tag_map": cell_tag_map,
            "facet_tag_map": facet_tag_map,
        },
    }
    writer.write_config(cfg)

    writer.write_summary(
        phi=written["phi"],
        eps=written["eps"],
        rho=written["rho"],
        extra={
            "degree": args.deg,
            "ksp_type": args.ksp_type,
            "pc_type": args.pc_type,
            "petsc_options_prefix": f"{args.basename}_",
            "bc_x_min": args.phi_left,
            "bc_x_max": args.phi_right,
        },
    )
    print_written_files(writer, comm)


if __name__ == "__main__":
    main()
