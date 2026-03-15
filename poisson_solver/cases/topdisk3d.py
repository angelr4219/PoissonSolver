from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from petsc4py import PETSc
from dolfinx import fem, io

from ..common import COMM, RANK, Phys, global_minmax, make_function_space
from ..materials import make_eps_cellwise_from_ct
from ..mesh_builders import build_mesh_topdisk_3d
from ..output import function_for_xdmf, write_meshtags_compat
from ..solver import solve_scalar_problem


def _flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


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

    ap.add_argument("--hard_exit", action="store_true")
    ap.add_argument("--skip_write", action="store_true")


def run(args) -> None:
    phys = Phys()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if RANK == 0:
        _flush_print("\n=== CASE: topdisk3d (CAD units before scale) ===")
        _flush_print(f"Domain: Lx={args.Lx} Ly={args.Ly} z in [{args.z_top - args.Lz}, {args.z_top}]")
        _flush_print(f"Disk: center=({args.disk_xc},{args.disk_yc}) R={args.disk_R} V={args.Vdisk}")
        _flush_print(f"h={args.h} deg={args.deg} split_z={args.split_z} eps_r0={args.eps_r0} eps_r1={args.eps_r1}")
        _flush_print(f"BCs: top={args.bc_top} bottom={args.bc_bottom} sides={args.bc_sides}")
        _flush_print(f"scale={args.scale:g}")

    _flush_print("[stage] build_mesh_topdisk_3d")
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

    _flush_print("[stage] epsilon and function space")
    eps_cell = make_eps_cellwise_from_ct(
        msh, ct, phys, eps_r0=args.eps_r0, eps_r1=args.eps_r1
    )

    V = make_function_space(msh, ("CG", args.deg))
    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    def add_tag_bc(tag: int, val: float, label: str):
        facets = ft.find(tag)
        if RANK == 0:
            _flush_print(f"BC {label}: tag={tag} facets={facets.size} val={val}")
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

    _flush_print("[stage] solve")
    phi, _ksp = solve_scalar_problem(
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
        _flush_print(f"phi min/max = [{gmin:.6e}, {gmax:.6e}]")

    if not args.skip_write:
        _flush_print("[stage] function_for_xdmf")
        phi_w = function_for_xdmf(phi, msh)

        x_all = outdir / f"{args.basename}.xdmf"

        _flush_print("[stage] write combined output")
        with io.XDMFFile(COMM, str(x_all), "w", encoding=io.XDMFFile.Encoding.ASCII) as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(phi_w)
            xdmf.write_function(eps_cell)
            write_meshtags_compat(xdmf, ft, msh)
            write_meshtags_compat(xdmf, ct, msh)

        if RANK == 0:
            _flush_print(f"Wrote: {x_all}")

    if RANK == 0:
        _flush_print("[stage] finished main body")

    if args.hard_exit:
        _flush_print("[stage] barrier before hard exit")
        try:
            COMM.Barrier()
        except Exception:
            pass
        os._exit(0)
