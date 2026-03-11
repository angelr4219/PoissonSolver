from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from petsc4py import PETSc
from dolfinx import fem, mesh

from ..common import COMM, RANK, global_minmax, make_function_space
from ..materials import (
    build_eps_from_cell_tags,
    build_region_id_from_cell_tags,
    gaussian_charge_expr,
    tag_rod_layers,
)
from ..mesh_builders import build_rod_with_bore
from ..output import function_for_xdmf, write_mesh_and_functions, write_mesh_and_meshtags
from ..solver import solve_scalar_problem


def register_parser(subparsers) -> None:
    ap = subparsers.add_parser(
        "rod_bore",
        help="Rectangular prism with cylindrical bore and layered dielectric tagging.",
    )
    ap.set_defaults(func=run)

    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--Lz", type=float, default=3.0)
    ap.add_argument("--R", type=float, default=0.15)
    ap.add_argument("--h", type=float, default=0.05)

    ap.add_argument("--t_si_bot", type=float, default=1.0)
    ap.add_argument("--t_sige", type=float, default=1.0)
    ap.add_argument("--t_si_top", type=float, default=1.0)

    ap.add_argument("--epsr_si", type=float, default=11.7)
    ap.add_argument("--epsr_sige", type=float, default=13.0)
    ap.add_argument("--epsr_air", type=float, default=1.0)

    ap.add_argument("--q", type=float, default=1e-12)
    ap.add_argument("--x0", type=float, default=0.5)
    ap.add_argument("--y0", type=float, default=0.5)
    ap.add_argument("--z0", type=float, default=1.5)
    ap.add_argument("--sigma", type=float, default=0.05)

    ap.add_argument("--outfile", type=str, default="results/phi_rod_bore_3d")


def run(args) -> None:
    assert abs(args.t_si_bot + args.t_sige + args.t_si_top - args.Lz) < 1e-12, (
        "Layer thicknesses must sum to Lz."
    )

    domain, ct_raw, ft = build_rod_with_bore(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        R=args.R,
        h=args.h,
    )

    ct = tag_rod_layers(
        domain,
        ct_raw,
        Lz=args.Lz,
        t_si_bot=args.t_si_bot,
        t_sige=args.t_sige,
        t_si_top=args.t_si_top,
        air_tag=101,
    )

    eps = build_eps_from_cell_tags(
        domain,
        ct,
        epsr_by_tag={
            101: args.epsr_air,
            201: args.epsr_si,
            202: args.epsr_sige,
            203: args.epsr_si,
        },
        name="eps_abs",
    )
    region_id = build_region_id_from_cell_tags(domain, ct, name="region_id")

    rho_expr = gaussian_charge_expr(
        domain,
        q_value=args.q,
        center=(args.x0, args.y0, args.z0),
        sigma=args.sigma,
    )

    V = make_function_space(domain, ("CG", 1))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdry_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc0 = fem.dirichletbc(PETSc.ScalarType(0.0), bdry_dofs, V)

    phi, ksp = solve_scalar_problem(
        V,
        eps,
        bcs=[bc0],
        rhs_expr=rho_expr,
        name="phi",
        petsc_options_prefix="rod_bore_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-10,
        },
    )

    gmin, gmax = global_minmax(phi)
    if RANK == 0:
        print("=== CASE: rod_bore ===")
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

    xdmf_path = Path(args.outfile).with_suffix(".xdmf")
    xdmf_path.parent.mkdir(parents=True, exist_ok=True)

    phi_out = function_for_xdmf(phi, domain)
    write_mesh_and_functions(COMM, domain, xdmf_path, [phi_out, eps, region_id])
    write_mesh_and_meshtags(COMM, domain, xdmf_path.with_name(xdmf_path.stem + "_cell_tags.xdmf"), ct)

    if RANK == 0:
        print(f"[done] Wrote {xdmf_path}")
        print("Tip: In ParaView, add 'Gradient' of phi and multiply by -1 to visualize E.")
