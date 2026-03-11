from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from petsc4py import PETSc
from dolfinx import fem, mesh

from ..common import COMM, RANK, global_minmax, make_function_space
from ..materials import build_disk_box_fields
from ..mesh_builders import gmsh_build_box_with_refinement_field, read_gmsh_mesh
from ..output import function_for_xdmf, write_mesh_and_functions, write_pre_fields
from ..solver import solve_scalar_problem


def register_parser(subparsers) -> None:
    ap = subparsers.add_parser(
        "charged_disk_box",
        help="3D box with a charged disk volume indicator and distance-based mesh refinement.",
    )
    ap.set_defaults(func=run)

    ap.add_argument("--R", type=float, required=True, help="Disk radius in meters, e.g. 80e-9")
    ap.add_argument("--t", type=float, default=4e-9, help="Disk thickness in meters")
    ap.add_argument("--Q", type=float, default=1.602176634e-19, help="Total charge in Coulombs")
    ap.add_argument("--out", type=str, default="disk_Q1e", help="Output directory")

    ap.add_argument("--p", type=int, default=2, help="Polynomial degree for solve")
    ap.add_argument("--epsr", type=float, default=12.0, help="Bulk relative permittivity")
    ap.add_argument("--z0", type=float, default=0.0, help="Disk center z location [m]")

    ap.add_argument("--Lx", type=float, default=600e-9)
    ap.add_argument("--Ly", type=float, default=600e-9)
    ap.add_argument("--Lz", type=float, default=400e-9)

    ap.add_argument("--n_diam", type=int, default=28, help="Elements across disk diameter")
    ap.add_argument("--refine_band_R", type=float, default=8.0, help="Refinement band in units of R")
    ap.add_argument("--far_h_factor", type=float, default=25.0, help="h_far = far_h_factor * h_disk")

    ap.add_argument(
        "--gate_scaffold",
        action="store_true",
        help="Print facet_tags presence as a placeholder for future gate exports.",
    )


def run(args) -> None:
    outdir = Path(args.out)
    msh_path = outdir / "mesh.msh"

    info = gmsh_build_box_with_refinement_field(
        comm=COMM,
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        R=args.R,
        z0=args.z0,
        n_diam=args.n_diam,
        refine_band_R=args.refine_band_R,
        far_h_factor=args.far_h_factor,
        msh_path=msh_path,
    )

    if RANK == 0:
        print("=== CASE: charged_disk_box ===")
        print(f"R = {args.R:.3e} m ({args.R * 1e9:.1f} nm), t = {args.t:.3e} m")
        print(f"n_diam = {args.n_diam} -> h_disk = {info['h_disk']:.3e} m ({info['h_disk'] * 1e9:.3f} nm)")
        print(f"h_far  = {info['h_far']:.3e} m ({info['h_far'] * 1e9:.3f} nm)")
        print(f"refine_band_R = {args.refine_band_R}R -> DistMax = {args.refine_band_R * args.R:.3e} m")
        print(f"Wrote mesh: {msh_path}")

    msh, cell_tags, facet_tags = read_gmsh_mesh(msh_path, COMM, gdim=3)

    rho, epsilon, region_id, inside, rho_disk = build_disk_box_fields(
        msh=msh,
        R=args.R,
        t=args.t,
        Q=args.Q,
        z0=args.z0,
        epsr_bulk=args.epsr,
    )

    write_pre_fields(COMM, msh, outdir, rho, epsilon, region_id)

    V = make_function_space(msh, ("CG", args.p))
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.full(X.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    phi, ksp = solve_scalar_problem(
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
    n_global = COMM.allreduce(n_local, op=COMM.SUM)

    if RANK == 0:
        print("=== Solve summary ===")
        print(f"p_solve = {args.p}")
        print(f"rho_disk = {rho_disk:.6e} C/m^3")
        print(f"charged DG0 cells = {n_global}")
        print(f"phi(min/max) = {gmin:.6e}, {gmax:.6e}")
        if ksp is not None:
            try:
                print(
                    f"[KSP] its={ksp.getIterationNumber()} "
                    f"reason={ksp.getConvergedReason()} "
                    f"rnorm={ksp.getResidualNorm():.3e}"
                )
            except Exception:
                pass

    phi_out = function_for_xdmf(phi, msh)
    write_mesh_and_functions(
        COMM,
        msh,
        outdir / "mesh_fields.xdmf",
        [phi_out, rho, epsilon, region_id],
    )

    if args.gate_scaffold and RANK == 0:
        has_facets = facet_tags is not None
        print(f"[gate_id scaffold] facet_tags present: {has_facets}")
