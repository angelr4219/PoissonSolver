from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from petsc4py import PETSc
from dolfinx import fem

from ..common import COMM, RANK, Phys, global_minmax, make_function_space
from ..materials import make_eps_cellwise_from_ct
from ..mesh_builders import build_mesh_topdisk_3d
from ..output import function_for_xdmf, write_mesh_and_functions, write_mesh_and_meshtags
from ..solver import solve_scalar_problem


def register_parser(subparsers) -> None:
    ap = subparsers.add_parser(
        "topdisk3d",
        help="3D box with a top circular electrode patch and optional dielectric split.",
    )
    ap.set_defaults(func=run)

    ap.add_argument("--outdir", default="Results/run", type=str)
    ap.add_argument("--basename", default="run", type=str)

    ap.add_argument("--deg", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1e-9)
    ap.add_argument("--h", type=float, default=10.0)

    ap.add_argument("--Lx", type=float, default=300.0)
    ap.add_argument("--Ly", type=float, default=300.0)
    ap.add_argument("--z_top", type=float, default=0.0)
    ap.add_argument("--Lz", type=float, default=200.0)

    ap.add_argument("--eps_r0", type=float, default=11.7)
    ap.add_argument("--eps_r1", type=float, default=None)
    ap.add_argument("--split_z", type=float, default=None)

    ap.add_argument("--bc_top", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")

    ap.add_argument("--disk_xc", type=float, default=0.0)
    ap.add_argument("--disk_yc", type=float, default=0.0)
    ap.add_argument("--disk_R", type=float, default=50.0)
    ap.add_argument("--Vdisk", type=float, default=1.0)

    ap.add_argument(
        "--hard_exit",
        action="store_true",
        help="Use os._exit(0) after completion to avoid rare PETSc teardown segfaults.",
    )


def run(args) -> None:
    phys = Phys()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if RANK == 0:
        print("\n=== CASE: topdisk3d (CAD units before scale) ===")
        print(f"Domain: Lx={args.Lx} Ly={args.Ly} z in [{args.z_top - args.Lz}, {args.z_top}]")
        print(f"Disk: center=({args.disk_xc},{args.disk_yc}) R={args.disk_R} V={args.Vdisk}")
        print(f"h={args.h} deg={args.deg} split_z={args.split_z} eps_r0={args.eps_r0} eps_r1={args.eps_r1}")
        print(f"BCs: top={args.bc_top} bottom={args.bc_bottom} sides={args.bc_sides}")
        print(f"scale={args.scale:g}")

    msh, ct, ft = build_mesh_topdisk_3d(
        Lx=args.Lx,
        Ly=args.Ly,
        z_top=args.z_top,
        Lz=args.Lz,
        disk_xc=args.disk_xc,
        disk_yc=args.disk_yc,
        disk_R=args.disk_R,
        h=args.h,
        phys=phys,
        split_z=args.split_z,
    )

    if args.scale != 1.0:
        msh.geometry.x[:] *= args.scale

    eps_cell = make_eps_cellwise_from_ct(
        msh, ct, phys, eps_r0=args.eps_r0, eps_r1=args.eps_r1
    )

    V = make_function_space(msh, ("CG", args.deg))
    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    def add_tag_bc(tag: int, val: float, label: str):
        facets = ft.find(tag)
        if RANK == 0:
            print(f"BC {label}: tag={tag} facets={facets.size} val={val}")
        if facets.size == 0:
            return
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(val), dofs, V))

    add_tag_bc(phys.TOPDISK, args.Vdisk, "top_disk")
    if args.bc_top == "dirichlet0":
        add_tag_bc(phys.TOP, 0.0, "top_ground")
    if args.bc_bottom == "dirichlet0":
        add_tag_bc(phys.BOTTOM, 0.0, "bottom")
    if args.bc_sides == "dirichlet0":
        add_tag_bc(phys.SIDES, 0.0, "sides")

    phi, ksp = solve_scalar_problem(
        V,
        eps_cell,
        bcs=bcs,
        rhs_expr=None,
        name="phi",
        petsc_options_prefix="laplace3d_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_max_it": 20000,
        },
    )

    gmin, gmax = global_minmax(phi)
    if RANK == 0:
        print(f"\nphi min/max = [{gmin:.6e}, {gmax:.6e}]")
        if ksp is not None:
            try:
                print(
                    f"[KSP] its={ksp.getIterationNumber()} "
                    f"reason={ksp.getConvergedReason()} "
                    f"rnorm={ksp.getResidualNorm():.3e}"
                )
            except Exception:
                pass

    phi_w = function_for_xdmf(phi, msh)

    x_phi = outdir / f"{args.basename}_phi.xdmf"
    x_eps = outdir / f"{args.basename}_eps_abs.xdmf"
    x_ft = outdir / f"{args.basename}_facet_tags.xdmf"
    x_ct = outdir / f"{args.basename}_cell_tags.xdmf"

    write_mesh_and_functions(COMM, msh, x_phi, [phi_w])
    write_mesh_and_functions(COMM, msh, x_eps, [eps_cell])
    write_mesh_and_meshtags(COMM, msh, x_ft, ft)
    write_mesh_and_meshtags(COMM, msh, x_ct, ct)

    if RANK == 0:
        print("\nParaView:")
        print(f"  open {x_phi} (phi)")
        print(f"  open {x_eps} (eps_abs)")
        print(f"  open {x_ft} and color by gmsh:physical (TOPDISK={phys.TOPDISK}, TOP={phys.TOP})")
        print(f"  open {x_ct} and color by gmsh:physical (VOL0={phys.VOL0}, VOL1={phys.VOL1})")

    if args.hard_exit:
        try:
            COMM.Barrier()
        except Exception:
            pass
        os._exit(0)
