#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

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

EPS0 = 8.8541878128e-12  # F/m


@dataclass(frozen=True)
class Phys:
    VOL: int = 1
    TOP: int = 10
    BOTTOM: int = 11
    SIDES: int = 12
    DISK: int = 20


def write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


def _build_disk_patch_tag(
    occ,
    z_top: float,
    disk_xc: float,
    disk_yc: float,
    disk_R: float,
    phys_disk_tag: int,
    tol: float = 1e-6,
) -> None:
    """
    Tag the subset of top surface facets whose (x,y) lies inside a disk of radius disk_R
    centered at (disk_xc, disk_yc), at z=z_top.

    This is the 'radius function' you asked for. It is purely geometric, robust, and does not
    rely on exact entity ids surviving boolean ops.
    """
    disk_ids = []
    for (dim, sid) in gmsh.model.getEntities(dim=2):
        cx, cy, cz = occ.getCenterOfMass(dim, sid)
        if abs(cz - z_top) > tol:
            continue
        r2 = (cx - disk_xc) ** 2 + (cy - disk_yc) ** 2
        if r2 <= (disk_R + tol) ** 2:
            disk_ids.append(sid)

    if disk_ids:
        gmsh.model.addPhysicalGroup(2, disk_ids, phys_disk_tag)
        gmsh.model.setPhysicalName(2, phys_disk_tag, "disk")
    else:
        print(f"[WARN] No top facets tagged as disk (tag={phys_disk_tag}). "
              f"Check disk center/R and CAD units. z_top={z_top}, R={disk_R}.")


def build_mesh_with_disk_patch(
    Lx: float,
    Ly: float,
    z_top: float,
    Lz: float,
    disk_xc: float,
    disk_yc: float,
    disk_R: float,
    h: float,
    model_name: str = "disk_patch",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags, Phys]:
    """
    Build in CAD units (nm recommended) then scale afterwards (msh.geometry.x *= scale).
    """
    phys = Phys()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    if RANK == 0:
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        xmin, xmax = -0.5 * Lx, 0.5 * Lx
        ymin, ymax = -0.5 * Ly, 0.5 * Ly
        zmin, zmax = z_top - Lz, z_top

        dev = occ.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin)
        occ.synchronize()

        # Tag volume(s)
        vols = gmsh.model.getEntities(dim=3)
        gmsh.model.addPhysicalGroup(3, [t for (_, t) in vols], phys.VOL)
        gmsh.model.setPhysicalName(3, phys.VOL, "device")

        # Tag outer boundaries: TOP/BOTTOM/SIDES
        surfs = gmsh.model.getEntities(dim=2)
        top_ids, bot_ids, side_ids = [], [], []
        for (dim, sid) in surfs:
            cx, cy, cz = occ.getCenterOfMass(dim, sid)
            if abs(cz - z_top) < 1e-6:
                top_ids.append(sid)
            elif abs(cz - (z_top - Lz)) < 1e-6:
                bot_ids.append(sid)
            else:
                side_ids.append(sid)

        if top_ids:
            gmsh.model.addPhysicalGroup(2, top_ids, phys.TOP)
            gmsh.model.setPhysicalName(2, phys.TOP, "top")
        if bot_ids:
            gmsh.model.addPhysicalGroup(2, bot_ids, phys.BOTTOM)
            gmsh.model.setPhysicalName(2, phys.BOTTOM, "bottom")
        if side_ids:
            gmsh.model.addPhysicalGroup(2, side_ids, phys.SIDES)
            gmsh.model.setPhysicalName(2, phys.SIDES, "sides")

        # Tag the disk subset on TOP using the radius function
        _build_disk_patch_tag(
            occ=occ,
            z_top=z_top,
            disk_xc=disk_xc,
            disk_yc=disk_yc,
            disk_R=disk_R,
            phys_disk_tag=phys.DISK,
            tol=1e-6,
        )

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]

    gmsh.finalize()
    return msh, ct, ft, phys


def build_eps_abs(
    msh: dmesh.Mesh,
    eps_uniform: float | None,
    tox_m: float,
    eps_ox: float,
    eps_si: float,
) -> fem.Function:
    """
    Returns cellwise absolute permittivity eps_abs (F/m) in DG0.

    If eps_uniform is not None: eps_abs = eps_uniform*EPS0 everywhere.
    Else: two-layer in z with oxide on top of thickness tox_m (meters), i.e. z in [-tox, 0].
    """
    V0 = fem.functionspace(msh, ("DG", 0))
    eps = fem.Function(V0, name="eps_abs")

    tdim = msh.topology.dim
    nloc = msh.topology.index_map(tdim).size_local
    cells = np.arange(nloc, dtype=np.int32)
    mids = dmesh.compute_midpoints(msh, tdim, cells)
    zc = mids[:, 2]  # meters (after scaling)

    if eps_uniform is not None:
        eps_r = float(eps_uniform) * np.ones_like(zc)
    else:
        # oxide is the top layer: z in [-tox, 0]
        eps_r = np.where(zc > -tox_m, float(eps_ox), float(eps_si))

    eps.x.array[:] = (EPS0 * eps_r).astype(eps.x.array.dtype)
    return eps


def solve_poisson(
    msh: dmesh.Mesh,
    ft: dmesh.MeshTags,
    phys: Phys,
    deg: int,
    eps_abs: fem.Function,
    Vdisk: float,
    bc_top: str,
    bc_bottom: str,
    bc_sides: str,
    rho0: float,
    xq: float,
    yq: float,
    zq: float,
    sigma: float,
) -> fem.Function:
    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # IMPORTANT: tie measures to mesh domain (safe)
    dx = ufl.Measure("dx", domain=msh)

    a = ufl.inner(eps_abs * ufl.grad(u), ufl.grad(v)) * dx

    # IMPORTANT: RHS must remain a valid form even when rho0=0 (dolfinx 0.10.0 behavior)
    rho0_c = fem.Constant(msh, PETSc.ScalarType(rho0))
    X = ufl.SpatialCoordinate(msh)
    r2 = (X[0] - xq) ** 2 + (X[1] - yq) ** 2 + (X[2] - zq) ** 2
    rho = rho0_c * ufl.exp(-r2 / (2.0 * sigma ** 2))
    L = rho * v * dx

    fdim = msh.topology.dim - 1
    bcs = []

    def add_tag_bc(tag: int, val: float, name: str):
        facets = ft.find(tag)
        if facets.size == 0:
            if RANK == 0:
                print(f"[WARN] facets for tag {tag} ({name}) not found in ft.")
            return
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        if RANK == 0:
            print(f"BC '{name}': tag={tag} facets={facets.size} dofs={len(dofs)} val={val}")
        if len(dofs) == 0:
            raise RuntimeError(f"BC '{name}' has 0 dofs (tag={tag}). Tagging failed.")
        bcs.append(fem.dirichletbc(PETSc.ScalarType(val), dofs, V))

    # Outer BCs
    if bc_top == "dirichlet0":
        add_tag_bc(phys.TOP, 0.0, "top")
    if bc_bottom == "dirichlet0":
        add_tag_bc(phys.BOTTOM, 0.0, "bottom")
    if bc_sides == "dirichlet0":
        add_tag_bc(phys.SIDES, 0.0, "sides")

    # Disk BC
    add_tag_bc(phys.DISK, Vdisk, "disk")

    problem = LinearProblem(
        a, L, bcs=bcs, u=phi,
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_max_it": 20000,
        },
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


def main():
    ap = argparse.ArgumentParser(description="Gmsh disk patch on top surface + Poisson/Laplace, scaled CAD units.")
    ap.add_argument("--outdir", default="Results/disk_patch", type=str)
    ap.add_argument("--basename", default="disk_patch", type=str)

    # CAD units (nm recommended)
    ap.add_argument("--Lx", type=float, default=300.0)
    ap.add_argument("--Ly", type=float, default=300.0)
    ap.add_argument("--z_top", type=float, default=0.0)
    ap.add_argument("--Lz", type=float, default=200.0)

    ap.add_argument("--h", type=float, default=10.0)
    ap.add_argument("--deg", type=int, default=1)

    # Disk in CAD units
    ap.add_argument("--disk_xc", type=float, default=0.0)
    ap.add_argument("--disk_yc", type=float, default=0.0)
    ap.add_argument("--disk_R", type=float, default=50.0)
    ap.add_argument("--Vdisk", type=float, default=0.1)

    # BC choices
    ap.add_argument("--bc_top", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")

    # Scale geometry after meshing
    ap.add_argument("--scale", type=float, default=1e-9, help="Multiply mesh coords by this (nm->m = 1e-9).")

    # Dielectrics
    ap.add_argument("--eps_uniform", type=float, default=None, help="If set, uniform relative permittivity.")
    ap.add_argument("--tox", type=float, default=0.0, help="Oxide thickness in meters (used if eps_uniform is None).")
    ap.add_argument("--eps_ox", type=float, default=3.9)
    ap.add_argument("--eps_si", type=float, default=11.7)

    # Charge (Gaussian)
    ap.add_argument("--rho0", type=float, default=0.0, help="Peak C/m^3. 0 => Laplace.")
    ap.add_argument("--xq", type=float, default=0.0, help="Charge center x in meters.")
    ap.add_argument("--yq", type=float, default=0.0, help="Charge center y in meters.")
    ap.add_argument("--zq", type=float, default=-60e-9, help="Charge center z in meters.")
    ap.add_argument("--sigma", type=float, default=15e-9, help="Gaussian sigma in meters.")

    args = ap.parse_args()

    if RANK == 0:
        print("\n=== gmsh_disk_poisson_scaled (CAD units -> scaled to meters) ===")
        print(f"CAD Domain: Lx={args.Lx} Ly={args.Ly} z in [{args.z_top-args.Lz}, {args.z_top}]")
        print(f"CAD Disk: center=({args.disk_xc},{args.disk_yc}) R={args.disk_R}  Vdisk={args.Vdisk}")
        print(f"h={args.h} deg={args.deg} scale={args.scale:g}")
        if args.eps_uniform is not None:
            print(f"Dielectric: uniform eps_r={args.eps_uniform}")
        else:
            print(f"Dielectric: tox={args.tox:.3e} m, eps_ox={args.eps_ox}, eps_si={args.eps_si}")
        print(f"Charge: rho0={args.rho0} C/m^3, center=({args.xq},{args.yq},{args.zq}), sigma={args.sigma}")

    msh, ct, ft, phys = build_mesh_with_disk_patch(
        Lx=args.Lx, Ly=args.Ly, z_top=args.z_top, Lz=args.Lz,
        disk_xc=args.disk_xc, disk_yc=args.disk_yc, disk_R=args.disk_R,
        h=args.h
    )

    # Scale mesh to meters
    if args.scale != 1.0:
        msh.geometry.x[:] *= args.scale

    # eps_abs in F/m
    eps_abs = build_eps_abs(
        msh=msh,
        eps_uniform=args.eps_uniform,
        tox_m=args.tox,
        eps_ox=args.eps_ox,
        eps_si=args.eps_si,
    )

    phi = solve_poisson(
        msh=msh, ft=ft, phys=phys,
        deg=args.deg, eps_abs=eps_abs,
        Vdisk=args.Vdisk,
        bc_top=args.bc_top, bc_bottom=args.bc_bottom, bc_sides=args.bc_sides,
        rho0=args.rho0, xq=args.xq, yq=args.yq, zq=args.zq, sigma=args.sigma
    )

    # Min/max print
    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)

    if RANK == 0:
        print(f"\nphi min/max = [{gmin:.6e}, {gmax:.6e}]")

        # quick tag coverage
        for name, tag in [("TOP", phys.TOP), ("BOTTOM", phys.BOTTOM), ("SIDES", phys.SIDES), ("DISK", phys.DISK)]:
            n = ft.find(tag).size
            print(f"facet tag {name}={tag}: facets={n}")

    # Outputs
    os.makedirs(args.outdir, exist_ok=True)
    x_phi = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    x_ft = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")

    with io.XDMFFile(COMM, x_phi, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(eps_abs)

    with io.XDMFFile(COMM, x_ft, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ft, msh)

    if RANK == 0:
        print("\nParaView:")
        print(f"  open {x_phi} (phi + eps_abs)")
        print(f"  open {x_ft} and color by gmsh:physical (facet tags)")
        print("  confirm disk facets exist (tag DISK=20) and are at phi=Vdisk")

if __name__ == "__main__":
    main()
