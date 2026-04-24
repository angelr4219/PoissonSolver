#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem as DOLFINxLinearProblem
from dolfinx_mpc import MultiPointConstraint, LinearProblem as MPCLinearProblem

from poisson_output_layer import XDMFRunWriter, print_written_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D Laplace box with optional periodic x/y constraints using dolfinx_mpc."
    )
    p.add_argument("--outdir", type=str, default="Results/periodic_box_demo")
    p.add_argument("--basename", type=str, default="periodic_box_demo")

    p.add_argument("--Lx_nm", type=float, default=300.0)
    p.add_argument("--Ly_nm", type=float, default=300.0)
    p.add_argument("--Lz_nm", type=float, default=120.0)

    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--nz", type=int, default=12)

    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--eps_r", type=float, default=11.7)
    p.add_argument("--rho", type=float, default=0.0)

    p.add_argument(
        "--periodic",
        type=str,
        default="xy",
        choices=["none", "x", "y", "xy"],
        help="Periodic directions implemented with dolfinx_mpc."
    )

    p.add_argument("--phi_zmin", type=float, default=0.0)
    p.add_argument("--phi_zmax", type=float, default=1.0)

    p.add_argument("--ksp_type", type=str, default="cg")
    p.add_argument("--pc_type", type=str, default="hypre")

    return p.parse_args()


def make_facet_tags(domain: mesh.Mesh, Lx: float, Ly: float, Lz: float):
    tdim = domain.topology.dim
    fdim = tdim - 1
    tol = 1.0e-12

    domain.topology.create_connectivity(fdim, tdim)

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

    facet_tag_map = {
        "x_min": 1,
        "x_max": 2,
        "y_min": 3,
        "y_max": 4,
        "z_min": 5,
        "z_max": 6,
    }

    facet_sets = [
        (facet_tag_map["x_min"], mesh.locate_entities_boundary(domain, fdim, x_min)),
        (facet_tag_map["x_max"], mesh.locate_entities_boundary(domain, fdim, x_max)),
        (facet_tag_map["y_min"], mesh.locate_entities_boundary(domain, fdim, y_min)),
        (facet_tag_map["y_max"], mesh.locate_entities_boundary(domain, fdim, y_max)),
        (facet_tag_map["z_min"], mesh.locate_entities_boundary(domain, fdim, z_min)),
        (facet_tag_map["z_max"], mesh.locate_entities_boundary(domain, fdim, z_max)),
    ]

    facets = np.hstack([f for _, f in facet_sets]).astype(np.int32)
    values = np.hstack([np.full(len(f), tag, dtype=np.int32) for tag, f in facet_sets])

    order = np.argsort(facets)
    facet_tags = mesh.meshtags(domain, fdim, facets[order], values[order])
    facet_tags.name = "facet_tags"
    return facet_tags, facet_tag_map


def make_cell_tags(domain: mesh.Mesh):
    tdim = domain.topology.dim
    num_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_local, dtype=np.int32)
    values = np.full(num_local, 1, dtype=np.int32)
    cell_tags = mesh.meshtags(domain, tdim, cells, values)
    cell_tags.name = "cell_tags"
    cell_tag_map = {"bulk": 1}
    return cell_tags, cell_tag_map


def build_periodic_mpc(
    V,
    bcs,
    periodic_mode: str,
    Lx: float,
    Ly: float,
) -> MultiPointConstraint:
    tol = 1.0e-12
    mpc = MultiPointConstraint(V)

    if periodic_mode in ("x", "xy"):
        def indicator_x(x):
            return np.isclose(x[0], Lx, atol=tol)

        def relation_x(x):
            out = x.copy()
            out[0] = out[0] - Lx
            return out

        mpc.create_periodic_constraint_geometrical(V, indicator_x, relation_x, bcs)

    if periodic_mode in ("y", "xy"):
        def indicator_y(x):
            mask = np.isclose(x[1], Ly, atol=tol)
            if periodic_mode == "xy":
                mask = np.logical_and(mask, np.logical_not(np.isclose(x[0], Lx, atol=tol)))
            return mask

        def relation_y(x):
            out = x.copy()
            out[1] = out[1] - Ly
            return out

        mpc.create_periodic_constraint_geometrical(V, indicator_y, relation_y, bcs)

    mpc.finalize()
    return mpc


def compute_error_metrics(phi: fem.Function, phi_exact: fem.Function) -> tuple[float, float, float]:
    comm = phi.function_space.mesh.comm

    local_linf = 0.0
    if phi.x.array.size > 0:
        local_linf = float(np.max(np.abs(phi.x.array - phi_exact.x.array)))
    err_linf = comm.allreduce(local_linf, op=MPI.MAX)

    e = phi - phi_exact

    l2_form = fem.form((e * e) * ufl.dx)
    h1_form = fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx)

    l2_sq_local = fem.assemble_scalar(l2_form)
    h1_sq_local = fem.assemble_scalar(h1_form)

    l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
    h1_sq = comm.allreduce(h1_sq_local, op=MPI.SUM)

    err_l2 = math.sqrt(max(l2_sq, 0.0))
    err_h1_semi = math.sqrt(max(h1_sq, 0.0))
    return err_linf, err_l2, err_h1_semi


def seam_mismatch_serial(V, phi: fem.Function, axis: int, L: float) -> tuple[float, int]:
    coords = V.tabulate_dof_coordinates()
    vals = phi.x.array
    tol = 1.0e-12

    left_mask = np.isclose(coords[:, axis], 0.0, atol=tol)
    right_mask = np.isclose(coords[:, axis], L, atol=tol)

    other_axes = [i for i in range(coords.shape[1]) if i != axis]

    left_map = {}
    for c, v in zip(coords[left_mask], vals[left_mask]):
        key = tuple(np.round(c[other_axes], 12))
        left_map[key] = float(v)

    diffs = []
    for c, v in zip(coords[right_mask], vals[right_mask]):
        key = tuple(np.round(c[other_axes], 12))
        if key in left_map:
            diffs.append(abs(float(v) - left_map[key]))

    if len(diffs) == 0:
        return float("nan"), 0
    return float(max(diffs)), len(diffs)


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank

    nm = 1.0e-9
    Lx = args.Lx_nm * nm
    Ly = args.Ly_nm * nm
    Lz = args.Lz_nm * nm

    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [args.nx, args.ny, args.nz],
        cell_type=mesh.CellType.tetrahedron,
    )

    V = fem.functionspace(domain, ("Lagrange", args.deg))
    V0 = fem.functionspace(domain, ("DG", 0))

    facet_tags, facet_tag_map = make_facet_tags(domain, Lx, Ly, Lz)
    cell_tags, cell_tag_map = make_cell_tags(domain)

    eps = fem.Function(V0, name="eps")
    eps.x.array[:] = args.eps_r
    eps.x.scatter_forward()

    rho = fem.Function(V0, name="rho")
    rho.x.array[:] = args.rho
    rho.x.scatter_forward()

    fdim = domain.topology.dim - 1
    zmin_facets = facet_tags.find(facet_tag_map["z_min"])
    zmax_facets = facet_tags.find(facet_tag_map["z_max"])

    dofs_zmin = fem.locate_dofs_topological(V, fdim, zmin_facets)
    dofs_zmax = fem.locate_dofs_topological(V, fdim, zmax_facets)

    bc_zmin = fem.dirichletbc(PETSc.ScalarType(args.phi_zmin), dofs_zmin, V)
    bc_zmax = fem.dirichletbc(PETSc.ScalarType(args.phi_zmax), dofs_zmax, V)
    bcs = [bc_zmin, bc_zmax]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    Lform = rho * v * ufl.dx

    phi = None
    num_local_slaves = 0
    num_global_slaves = 0

    if args.periodic == "none":
        problem = DOLFINxLinearProblem(
            a,
            Lform,
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
    else:
        mpc = build_periodic_mpc(V, bcs, args.periodic, Lx, Ly)

        num_local_slaves = int(mpc.num_local_slaves)
        num_global_slaves = int(comm.allreduce(num_local_slaves, op=MPI.SUM))

        problem = MPCLinearProblem(
            a,
            Lform,
            mpc,
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

        # Critical fix for periodic MPC output and postprocessing:
        # recover slave dof values in the full function after solving the reduced system.
        mpc.backsubstitution(phi)
        phi.x.scatter_forward()

    phi.name = "phi"

    phi_exact = fem.Function(V, name="phi_exact")
    phi_exact.interpolate(
        lambda x: args.phi_zmin + (args.phi_zmax - args.phi_zmin) * (x[2] / Lz)
    )
    phi_exact.x.scatter_forward()

    err_linf, err_l2, err_h1_semi = compute_error_metrics(phi, phi_exact)

    x_seam = float("nan")
    y_seam = float("nan")
    x_pairs = 0
    y_pairs = 0

    if comm.size == 1:
        if args.periodic in ("x", "xy"):
            x_seam, x_pairs = seam_mismatch_serial(V, phi, axis=0, L=Lx)
        if args.periodic in ("y", "xy"):
            y_seam, y_pairs = seam_mismatch_serial(V, phi, axis=1, L=Ly)

    phi_min_local = float(np.min(phi.x.array)) if phi.x.array.size else np.inf
    phi_max_local = float(np.max(phi.x.array)) if phi.x.array.size else -np.inf
    phi_min = comm.allreduce(phi_min_local, op=MPI.MIN)
    phi_max = comm.allreduce(phi_max_local, op=MPI.MAX)

    if rank == 0:
        print("\n=== Periodic box scaffold ===")
        print(f"periodic mode      : {args.periodic}")
        print(f"mesh cells         : {args.nx} x {args.ny} x {args.nz}")
        print(f"degree             : {args.deg}")
        print(f"ksp_type / pc_type : {args.ksp_type} / {args.pc_type}")
        print(f"phi min/max        : {phi_min:.12e}  {phi_max:.12e}")
        print(f"eps_r              : {args.eps_r:.12e}")
        print(f"rho                : {args.rho:.12e}")
        print(f"BC z_min / z_max   : {args.phi_zmin:.6e} / {args.phi_zmax:.6e}")
        if args.periodic != "none":
            print(f"MPC global slaves  : {num_global_slaves}")

        print("\n=== Error metrics vs exact phi(z) ===")
        print(f"Linf dof error     : {err_linf:.12e}")
        print(f"L2 error           : {err_l2:.12e}")
        print(f"H1 seminorm error  : {err_h1_semi:.12e}")

        print("\n=== Periodic seam checks ===")
        if args.periodic in ("x", "xy"):
            print(f"x seam max mismatch: {x_seam:.12e}  over {x_pairs} matched dof pairs")
        if args.periodic in ("y", "xy"):
            print(f"y seam max mismatch: {y_seam:.12e}  over {y_pairs} matched dof pairs")
        if args.periodic == "none":
            print("no periodic seams requested")

    writer = XDMFRunWriter(args.outdir, args.basename)
    written = writer.write_fields(phi=phi, eps=eps, rho=rho, force_phi_cg1=True)
    writer.write_meshtags(domain, cell_tags=cell_tags, facet_tags=facet_tags)

    cfg = {
        "case_name": args.basename,
        "geometry": {
            "type": "box",
            "Lx_nm": args.Lx_nm,
            "Ly_nm": args.Ly_nm,
            "Lz_nm": args.Lz_nm,
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
            "phi_zmin": args.phi_zmin,
            "phi_zmax": args.phi_zmax,
        },
        "periodic": {
            "mode": args.periodic,
            "global_slaves": num_global_slaves,
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
            "periodic_mode": args.periodic,
            "global_slaves": num_global_slaves,
            "linf_error": err_linf,
            "l2_error": err_l2,
            "h1_seminorm_error": err_h1_semi,
            "x_seam_max_mismatch": x_seam,
            "y_seam_max_mismatch": y_seam,
            "bc_zmin": args.phi_zmin,
            "bc_zmax": args.phi_zmax,
        },
    )
    print_written_files(writer, comm)


if __name__ == "__main__":
    main()
