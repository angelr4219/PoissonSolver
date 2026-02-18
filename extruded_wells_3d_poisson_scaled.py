#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from mpi4py import MPI

import gmsh
import ufl
from petsc4py import PETSc
from dolfinx import fem, io, mesh as dmesh

try:
    from dolfinx.io import gmshio as gmshio
except Exception:
    from dolfinx.io import gmsh as gmshio

from dolfinx.fem.petsc import LinearProblem

COMM = MPI.COMM_WORLD
RANK = COMM.rank


@dataclass(frozen=True)
class Phys:
    # 3D
    VOL0: int = 1
    VOL1: int = 2
    TOP: int = 10
    BOTTOM: int = 11
    SIDES: int = 12
    TOPDISK: int = 20

    # 2D
    DISK_REGION: int = 1
    DISK_BND: int = 10


def write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


def _degree_compat(deg_or_callable) -> int:
    try:
        return int(deg_or_callable())
    except TypeError:
        return int(deg_or_callable)


def function_for_xdmf(phi: fem.Function, msh: dmesh.Mesh) -> fem.Function:
    mesh_deg = _degree_compat(msh.geometry.cmap.degree)
    sol_deg = _degree_compat(phi.function_space.ufl_element().degree)
    if sol_deg == mesh_deg:
        return phi
    Vout = fem.functionspace(msh, ("CG", mesh_deg))
    phi_out = fem.Function(Vout, name=phi.name)
    try:
        phi_out.interpolate(phi)
    except Exception:
        expr = fem.Expression(phi, Vout.element.interpolation_points())
        phi_out.interpolate(expr)
    return phi_out


def _global_minmax(phi: fem.Function) -> Tuple[float, float]:
    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)
    return gmin, gmax


# -------------------------
# 3D: TOP DISK ELECTRODE
# -------------------------
def build_mesh_topdisk_3d(
    Lx: float,
    Ly: float,
    z_top: float,
    Lz: float,
    disk_xc: float,
    disk_yc: float,
    disk_R: float,
    h: float,
    phys: Phys,
    split_z: Optional[float],
    model_name: str = "topdisk3d",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    if RANK == 0:
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        xmin, xmax = -0.5 * Lx, 0.5 * Lx
        ymin, ymax = -0.5 * Ly, 0.5 * Ly
        z_bot = z_top - Lz

        dev = occ.addBox(xmin, ymin, z_bot, xmax - xmin, ymax - ymin, Lz)
        occ.synchronize()

        # Optional dielectric split plane (fragment with slab)
        if split_z is not None and (0.0 < split_z < Lz):
            z_split = z_top - split_z
            slab = occ.addBox(xmin, ymin, z_split, xmax - xmin, ymax - ymin, z_top - z_split)
            occ.synchronize()
            occ.fragment([(3, dev)], [(3, slab)])
            occ.synchronize()

        # Add disk surface at top
        disk = occ.addDisk(disk_xc, disk_yc, z_top, disk_R, disk_R)
        occ.synchronize()

        # IMPORTANT: fragment ALL current volumes with disk, and capture mapping for disk-generated surfaces
        vols_now = gmsh.model.getEntities(dim=3)
        out_dimtags, out_map = occ.fragment(vols_now, [(2, disk)])
        occ.synchronize()

        # out_map has length len(vols_now) + len([(2,disk)]) = len(vols_now)+1
        # last entry corresponds to the disk input, mapped to resulting entities
        disk_out = out_map[len(vols_now)]
        disk_surface_ids = [tag for (dim, tag) in disk_out if dim == 2]

        # Tag volumes
        vols = gmsh.model.getEntities(dim=3)
        if split_z is None:
            gmsh.model.addPhysicalGroup(3, [t for (_, t) in vols], phys.VOL0)
            gmsh.model.setPhysicalName(3, phys.VOL0, "vol0")
        else:
            z_split = z_top - split_z
            v0, v1 = [], []
            for (d, vid) in vols:
                bx0, by0, bz0, bx1, by1, bz1 = occ.getBoundingBox(d, vid)
                zc = 0.5 * (bz0 + bz1)
                (v0 if zc >= z_split else v1).append(vid)
            if v0:
                gmsh.model.addPhysicalGroup(3, v0, phys.VOL0)
                gmsh.model.setPhysicalName(3, phys.VOL0, "vol_upper")
            if v1:
                gmsh.model.addPhysicalGroup(3, v1, phys.VOL1)
                gmsh.model.setPhysicalName(3, phys.VOL1, "vol_lower")

        # Tag surfaces: use disk_surface_ids directly
        surfs = gmsh.model.getEntities(dim=2)
        top_ids, bot_ids, side_ids = [], [], []

        tolz = max(1e-6 * max(Lx, Ly, Lz, 1.0), 0.25 * h)

        disk_set = set(disk_surface_ids)

        for (dim, sid) in surfs:
            bx0, by0, bz0, bx1, by1, bz1 = occ.getBoundingBox(dim, sid)
            is_top = (abs(bz0 - z_top) < tolz) and (abs(bz1 - z_top) < tolz)
            is_bot = (abs(bz0 - z_bot) < tolz) and (abs(bz1 - z_bot) < tolz)

            if is_top:
                if sid in disk_set:
                    continue  # will be tagged as TOPDISK
                top_ids.append(sid)
            elif is_bot:
                bot_ids.append(sid)
            else:
                side_ids.append(sid)

        if disk_surface_ids:
            gmsh.model.addPhysicalGroup(2, disk_surface_ids, phys.TOPDISK)
            gmsh.model.setPhysicalName(2, phys.TOPDISK, "top_disk")
        else:
            print("[WARN] Disk fragment produced 0 surface ids. (unexpected)")

        if top_ids:
            gmsh.model.addPhysicalGroup(2, top_ids, phys.TOP)
            gmsh.model.setPhysicalName(2, phys.TOP, "top_ground")
        if bot_ids:
            gmsh.model.addPhysicalGroup(2, bot_ids, phys.BOTTOM)
            gmsh.model.setPhysicalName(2, phys.BOTTOM, "bottom")
        if side_ids:
            gmsh.model.addPhysicalGroup(2, side_ids, phys.SIDES)
            gmsh.model.setPhysicalName(2, phys.SIDES, "sides")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]
    gmsh.finalize()
    return msh, ct, ft


def make_eps_cellwise_from_ct(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    phys: Phys,
    eps_r0: float,
    eps_r1: Optional[float],
) -> fem.Function:
    eps0 = 8.8541878128e-12
    V0 = fem.functionspace(msh, ("DG", 0))
    eps = fem.Function(V0, name="eps_abs")
    eps.x.array[:] = eps0 * eps_r0
    if eps_r1 is not None:
        cells1 = ct.find(phys.VOL1)
        if cells1.size > 0:
            eps.x.array[cells1] = eps0 * float(eps_r1)
    return eps


def solve_laplace_3d_sameV(
    V: fem.FunctionSpace,
    eps_cell: fem.Function,
    bcs: List[fem.DirichletBC],
    rho_expr: Optional[ufl.core.expr.Expr] = None,
) -> Tuple[fem.Function, Optional[PETSc.KSP]]:
    msh = V.mesh
    phi = fem.Function(V, name="phi")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(eps_cell * ufl.grad(u), ufl.grad(v)) * ufl.dx
    if rho_expr is None:
        L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx
    else:
        L = rho_expr * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=bcs, u=phi,
        petsc_options_prefix="laplace3d_",
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12, "ksp_atol": 1e-14, "ksp_max_it": 20000},
    )
    phi = problem.solve()
    phi.name = "phi"

    ksp = None
    try:
        ksp = problem.solver
    except Exception:
        pass
    return phi, ksp


def main():
    ap = argparse.ArgumentParser(description="topdisk3d patched: correct disk tagging + same-V BCs.")
    ap.add_argument("--mode", choices=["topdisk3d", "disk2d"], default="topdisk3d")
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
    ap.add_argument("--rho0", type=float, default=0.0, help="Uniform charge density amplitude (disk2d mode).")
    ap.add_argument("--sigma", type=float, default=10.0, help="Gaussian width for charge (disk2d mode).")
    ap.add_argument("--eps_r1", type=float, default=None)
    ap.add_argument("--split_z", type=float, default=None)

    ap.add_argument("--bc_top", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")

    ap.add_argument("--disk_xc", type=float, default=0.0)
    ap.add_argument("--disk_yc", type=float, default=0.0)
    ap.add_argument("--disk_R", type=float, default=50.0)
    ap.add_argument("--Vdisk", type=float, default=1.0)

    args = ap.parse_args()
    phys = Phys()
    os.makedirs(args.outdir, exist_ok=True)

    if RANK == 0:
        print("\n=== MODE: topdisk3d (CAD units) ===")
        print(f"Domain: Lx={args.Lx} Ly={args.Ly} z in [{args.z_top-args.Lz}, {args.z_top}]")
        print(f"Disk: center=({args.disk_xc},{args.disk_yc}) R={args.disk_R} V={args.Vdisk}")
        print(f"h={args.h} deg={args.deg} split_z={args.split_z} eps_r0={args.eps_r0} eps_r1={args.eps_r1}")
        print(f"BCs: top={args.bc_top} bottom={args.bc_bottom} sides={args.bc_sides}")
        print(f"scale={args.scale:g}")

    msh, ct, ft = build_mesh_topdisk_3d(
        Lx=args.Lx, Ly=args.Ly, z_top=args.z_top, Lz=args.Lz,
        disk_xc=args.disk_xc, disk_yc=args.disk_yc, disk_R=args.disk_R,
        h=args.h, phys=phys, split_z=args.split_z
    )
    if args.scale != 1.0:
        msh.geometry.x[:] *= args.scale

    eps_cell = make_eps_cellwise_from_ct(msh, ct, phys, eps_r0=args.eps_r0, eps_r1=args.eps_r1)

    # IMPORTANT: build V once and use it everywhere
    V = fem.functionspace(msh, ("CG", args.deg))
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

    phi, ksp = solve_laplace_3d_sameV(V, eps_cell, bcs=bcs, rho_expr=None)

    gmin, gmax = _global_minmax(phi)
    if RANK == 0:
        print(f"\nphi min/max = [{gmin:.6e}, {gmax:.6e}]")
        if ksp is not None:
            try:
                print(f"[KSP] its={ksp.getIterationNumber()} reason={ksp.getConvergedReason()} rnorm={ksp.getResidualNorm():.3e}")
            except Exception:
                pass

    phi_w = function_for_xdmf(phi, msh)

    x_phi = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    x_eps = os.path.join(args.outdir, f"{args.basename}_eps_abs.xdmf")
    x_ft = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")
    import os
    if os.environ.get("NOWRITE","0")=="1":
        if RANK==0: print("NOTE: NOWRITE=1, skipping output")
        return

    x_ct = os.path.join(args.outdir, f"{args.basename}_cell_tags.xdmf")

    import os
    if os.environ.get("NOWRITE","0")=="1":
        if RANK==0: print("NOTE: NOWRITE=1, skipping output")
        return

    with io.XDMFFile(COMM, x_phi, "w", encoding=io.XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_w)

    with io.XDMFFile(COMM, x_eps, "w", encoding=io.XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(eps_cell)

    with io.XDMFFile(COMM, x_ft, "w", encoding=io.XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ft, msh)

    with io.XDMFFile(COMM, x_ct, "w", encoding=io.XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ct, msh)

    if RANK == 0:
        print("\nParaView:")
        print(f"  open {x_phi} (phi)")
        print(f"  open {x_eps} (eps_abs)")
        print(f"  open {x_ft} and color by gmsh:physical (TOPDISK={phys.TOPDISK}, TOP={phys.TOP})")
        print(f"  open {x_ct} and color by gmsh:physical (VOL0={phys.VOL0}, VOL1={phys.VOL1})")


    # Hard-exit to avoid occasional PETSc teardown segfaults in some builds
    try:
        from mpi4py import MPI as _MPI
        _MPI.COMM_WORLD.Barrier()
    except Exception:
        pass
    os._exit(0)

if __name__ == "__main__":
    main()
