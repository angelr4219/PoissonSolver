from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh

from ..common import COMM, RANK, global_minmax, make_function_space
from ..materials import build_multi_disk_box_fields
from ..mesh_builders import gmsh_build_box_with_refinement_field, read_gmsh_mesh
from ..output import function_for_xdmf, write_mesh_and_functions, write_pre_fields
from ..solver import solve_scalar_problem


def _flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


def _parse_layers(layer_args):
    if not layer_args:
        return []
    layers = []
    for raw in layer_args:
        parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid --layer entry '{raw}'. Expected zmin,zmax,epsr")
        zmin, zmax, epsr = map(float, parts)
        layers.append((zmin, zmax, epsr))
    return layers


def _parse_disks(args):
    """
    New format:
      --disk xc,yc,zc,R,t,Q
    Old fallback:
      --R, --t, --Q, --z0  -> one centered disk at (0,0,z0)
    """
    if args.disk:
        disks = []
        for raw in args.disk:
            parts = [p.strip() for p in str(raw).split(",")]
            if len(parts) != 6:
                raise ValueError(
                    f"Invalid --disk entry '{raw}'. Expected xc,yc,zc,R,t,Q"
                )
            xc, yc, zc, R, t, Q = map(float, parts)
            disks.append((xc, yc, zc, R, t, Q))
        return disks

    # backward-compatible single-disk mode
    return [(0.0, 0.0, float(args.z0), float(args.R), float(args.t), float(args.Q))]


def register_parser(subparsers) -> None:
    ap = subparsers.add_parser(
        "charged_disk_box",
        help="3D box with one or more charged disks and distance-based mesh refinement.",
    )
    ap.set_defaults(func=run)

    # old single-disk args kept for compatibility
    ap.add_argument("--R", type=float, default=None, help="Single-disk radius [m]")
    ap.add_argument("--t", type=float, default=4e-9, help="Single-disk thickness [m]")
    ap.add_argument("--Q", type=float, default=1.602176634e-19, help="Single-disk total charge [C]")
    ap.add_argument("--z0", type=float, default=0.0, help="Single-disk center z [m]")

    # new general disk syntax
    ap.add_argument(
        "--disk",
        action="append",
        default=[],
        help="Charged disk as xc,yc,zc,R,t,Q . Repeatable.",
    )

    ap.add_argument("--out", type=str, default="disk_Q1e", help="Output directory")

    ap.add_argument("--p", type=int, default=2, help="Polynomial degree for solve")
    ap.add_argument("--epsr", type=float, default=12.0, help="Default relative permittivity")

    ap.add_argument("--Lx", type=float, default=600e-9)
    ap.add_argument("--Ly", type=float, default=600e-9)
    ap.add_argument("--Lz", type=float, default=400e-9)

    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)

    ap.add_argument("--n_diam", type=int, default=28, help="Elements across reference disk diameter")
    ap.add_argument("--refine_band_R", type=float, default=8.0, help="Refinement band in units of reference R")
    ap.add_argument("--far_h_factor", type=float, default=25.0, help="h_far = far_h_factor * h_disk")

    ap.add_argument(
        "--layer",
        action="append",
        default=[],
        help="Optional dielectric layer as zmin,zmax,epsr . Repeatable.",
    )

    ap.add_argument("--gate_scaffold", action="store_true")
    ap.add_argument("--hard_exit", action="store_true")
    ap.add_argument("--skip_write", action="store_true")


def run(args) -> None:
    outdir = Path(args.out)
    msh_path = outdir / "mesh.msh"

    disks = _parse_disks(args)
    layers = _parse_layers(args.layer)

    # use the largest disk as the refinement reference
    ref_R = max(d[3] for d in disks)

    # use z of first disk only for the current helper refinement field
    ref_z0 = disks[0][2]

    _flush_print("[stage] build mesh file")
    info = gmsh_build_box_with_refinement_field(
        comm=COMM,
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        R=ref_R,
        z0=ref_z0,
        n_diam=args.n_diam,
        refine_band_R=args.refine_band_R,
        far_h_factor=args.far_h_factor,
        msh_path=msh_path,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        zmin=args.zmin,
        zmax=args.zmax,
    )

    if RANK == 0:
        _flush_print("=== CASE: charged_disk_box ===")
        _flush_print(f"number of disks = {len(disks)}")
        for i, (xc, yc, zc, R, t, Q) in enumerate(disks, start=1):
            _flush_print(
                f"disk[{i}] = xc={xc:.3e}, yc={yc:.3e}, zc={zc:.3e}, "
                f"R={R:.3e}, t={t:.3e}, Q={Q:.3e}"
            )
        _flush_print(f"n_diam = {args.n_diam} -> h_disk = {info['h_disk']:.3e} m")
        _flush_print(f"h_far  = {info['h_far']:.3e} m")
        _flush_print(
            f"box = x:[{info['xmin']:.3e},{info['xmax']:.3e}] "
            f"y:[{info['ymin']:.3e},{info['ymax']:.3e}] "
            f"z:[{info['zmin']:.3e},{info['zmax']:.3e}]"
        )
        if layers:
            _flush_print(f"dielectric layers = {layers}")
        else:
            _flush_print(f"uniform epsr = {args.epsr}")
        _flush_print(f"Wrote mesh: {msh_path}")

    _flush_print("[stage] read mesh")
    msh, cell_tags, facet_tags = read_gmsh_mesh(msh_path, COMM, gdim=3)

    _flush_print("[stage] build fields")
    rho, epsilon, region_id, inside, rho_total_assigned = build_multi_disk_box_fields(
        msh=msh,
        disks=disks,
        epsr_bulk=args.epsr,
        layers=layers if layers else None,
    )

    if not args.skip_write:
        _flush_print("[stage] write pre_fields")
        write_pre_fields(COMM, msh, outdir, rho, epsilon, region_id)

    V = make_function_space(msh, ("CG", args.p))
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.full(X.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    _flush_print("[stage] solve")
    phi, _ksp = solve_scalar_problem(
        V,
        epsilon,
        bcs=[bc],
        rhs_expr=rho,
        name="phi_p",
        petsc_options_prefix="charged_disk_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 4000,
        },
    )

    gmin, gmax = global_minmax(phi)
    n_local = int(np.count_nonzero(inside))
    n_global = COMM.allreduce(n_local, op=MPI.SUM)

    if RANK == 0:
        _flush_print("=== Solve summary ===")
        _flush_print(f"p_solve = {args.p}")
        _flush_print(f"charged DG0 cells = {n_global}")
        _flush_print(f"rho accumulator = {rho_total_assigned:.6e}")
        _flush_print(f"phi(min/max) = {gmin:.6e}, {gmax:.6e}")

    if not args.skip_write:
        _flush_print("[stage] function_for_xdmf")
        phi_out = function_for_xdmf(phi, msh)

        _flush_print("[stage] write mesh_fields")
        write_mesh_and_functions(
            COMM,
            msh,
            outdir / "mesh_fields.xdmf",
            [phi_out, rho, epsilon, region_id],
        )

    if args.gate_scaffold and RANK == 0:
        has_facets = facet_tags is not None
        _flush_print(f"[gate_id scaffold] facet_tags present: {has_facets}")

    if RANK == 0:
        _flush_print("[stage] finished main body")

    if args.hard_exit:
        _flush_print("[stage] barrier before hard exit")
        try:
            COMM.Barrier()
        except Exception:
            pass
        os._exit(0)
