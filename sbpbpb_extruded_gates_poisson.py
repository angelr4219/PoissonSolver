#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from mpi4py import MPI

import gmsh
import ufl
from petsc4py import PETSc

from dolfinx import fem, io, mesh as dmesh

# DOLFINx compatibility: gmsh import helper moved across versions
try:
    from dolfinx.io import gmshio as gmshio
except Exception:
    from dolfinx.io import gmsh as gmshio

from dolfinx.fem.petsc import LinearProblem as _LinearProblem

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def normalize_outdir(outdir: str) -> str:
    if outdir == "results":
        return "Results"
    if outdir.startswith("results/"):
        return "Results/" + outdir[len("results/") :]
    return outdir


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
    if RANK == 0:
        print(f"[INFO] XDMF requires degree match: mesh degree={mesh_deg}, phi degree={sol_deg}. Writing interpolated CG{mesh_deg}.")
    Vout = fem.functionspace(msh, ("CG", mesh_deg))
    phi_out = fem.Function(Vout, name=phi.name)
    try:
        phi_out.interpolate(phi)
    except Exception:
        expr = fem.Expression(phi, Vout.element.interpolation_points())
        phi_out.interpolate(expr)
    return phi_out


@dataclass(frozen=True)
class Rect:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass(frozen=True)
class Phys:
    # volumes
    SIGE: int = 1
    GATE0: int = 100  # gate volume tags start here (and also surface tags for electrodes)
    # outer boundary "ground"
    TOP_GROUND: int = 10
    BOTTOM_GROUND: int = 11
    SIDES: int = 12


def ct_to_dg0(msh: dmesh.Mesh, ct: dmesh.MeshTags, name: str = "region_id") -> fem.Function:
    V0 = fem.functionspace(msh, ("DG", 0))
    f = fem.Function(V0, name=name)
    f.x.array[:] = 0
    f.x.array[ct.indices] = ct.values.astype(f.x.array.dtype, copy=False)
    return f


def solve_poisson_with_gate_surfaces(
    msh: dmesh.Mesh,
    ft: dmesh.MeshTags,
    gate_surface_tags: Dict[str, int],
    gateV: Dict[str, float],
    default_gateV: float,
    deg: int,
    eps_r: float,
    rho: float,
    outer_bc_tags: Dict[str, int],
    outer_bc_mode: Dict[str, str],
) -> fem.Function:
    """
    Solve on WHOLE domain (SiGe + gate volumes):

      -div(eps grad phi) = rho

    with Dirichlet:
      phi = V_gate on all gate electrode surfaces (tagged in ft)
      optionally phi=0 on outer boundaries (top_ground/bottom/sides)
    """
    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps0 = 8.8541878128e-12
    eps = fem.Constant(msh, PETSc.ScalarType(eps0 * eps_r))
    rhs = fem.Constant(msh, PETSc.ScalarType(rho))

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rhs * v * ufl.dx

    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    # Gate electrode BCs
    for name, tag in gate_surface_tags.items():
        Vg = float(gateV.get(name, default_gateV))
        facets = ft.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    # Optional outer boundary pinning
    for bname, tag in outer_bc_tags.items():
        mode = outer_bc_mode.get(bname, "none")
        if mode != "dirichlet0":
            continue
        facets = ft.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    problem = _LinearProblem(
        a, L, bcs=bcs, u=phi,
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-12,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 20000,
        },
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


def build_geometry_extruded_gates(
    # physical meters
    Lx_m: float, Ly_m: float, Lz_m: float,
    h_m: float,
    # gates
    gate_thickness_m: float,
    s_width_m: float,
    # BPBPB top pattern
    finger_len_y_m: float,
    finger_wx_m: float,
    finger_gap_m: float,
    # CAD scaling
    occ_scale: float,
    z_top_m: float = 0.0,
    model_name: str = "sbpbpb_extruded",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags, Dict[str, int], Dict[str, int], Phys]:
    """
    Build slab + outward-extruded gate prisms:

      - SiGe slab: z in [z_top-Lz, z_top]
      - S_TOP prism above top surface (z in [z_top, z_top+t])
      - S_BOT prism below bottom surface (z in [z_bot-t, z_bot])
      - BPBPB prisms above top surface (z in [z_top, z_top+t])

    Fragment to get conforming mesh, then tag:
      - cell regions: SiGe=1, gates get ids 100,101,...
      - facet regions: electrode surfaces for each gate prism use SAME ids 100,101,... (easy)
      - outer boundary: top_ground, bottom_ground, sides
    """
    phys = Phys()

    # scale to OCC units
    Lx = Lx_m * occ_scale
    Ly = Ly_m * occ_scale
    Lz = Lz_m * occ_scale
    h = h_m * occ_scale
    tgate = gate_thickness_m * occ_scale
    s_width = s_width_m * occ_scale
    finger_len_y = finger_len_y_m * occ_scale
    finger_wx = finger_wx_m * occ_scale
    finger_gap = finger_gap_m * occ_scale
    z_top = z_top_m * occ_scale

    if tgate <= 0:
        raise ValueError("--gate_thickness must be > 0")

    xmin, xmax = -0.5 * Lx, 0.5 * Lx
    ymin, ymax = -0.5 * Ly, 0.5 * Ly
    zbot, zmax = z_top - Lz, z_top

    # S-gate rectangles (stripes)
    s_top_rect = Rect("S_TOP", xmin, xmax, ymax - s_width, ymax)
    s_bot_rect = Rect("S_BOT", xmin, xmax, ymin, ymin + s_width)

    # BPBPB rectangles centered in y
    names = ["B0", "P1", "B1", "P2", "B2"]
    n = len(names)
    total_w = n * finger_wx + (n - 1) * finger_gap
    x0 = -0.5 * total_w
    y0 = -0.5 * finger_len_y
    y1 = 0.5 * finger_len_y

    finger_rects: List[Rect] = []
    for i, nm in enumerate(names):
        xi0 = x0 + i * (finger_wx + finger_gap)
        xi1 = xi0 + finger_wx
        finger_rects.append(Rect(nm, xi0, xi1, y0, y1))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    gate_surface_tags: Dict[str, int] = {}
    gate_volume_tags: Dict[str, int] = {}

    if RANK == 0:
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        # SiGe slab
        dev = occ.addBox(xmin, ymin, zbot, xmax - xmin, ymax - ymin, zmax - zbot)

        # Gate prisms
        gate_names: List[str] = []
        gate_prisms: List[int] = []

        def add_prism_from_rect(r: Rect, z0: float, dz: float) -> int:
            dx = r.xmax - r.xmin
            dy = r.ymax - r.ymin
            return occ.addBox(r.xmin, r.ymin, z0, dx, dy, dz)

        # Top S prism (above)
        g_stop = add_prism_from_rect(s_top_rect, z_top, tgate)
        gate_names.append("S_TOP")
        gate_prisms.append(g_stop)

        # Bottom S prism (below)
        g_sbot = add_prism_from_rect(s_bot_rect, zbot - tgate, tgate)
        gate_names.append("S_BOT")
        gate_prisms.append(g_sbot)

        # BPBPB prisms above top
        for r in finger_rects:
            g = add_prism_from_rect(r, z_top, tgate)
            gate_names.append(r.name)
            gate_prisms.append(g)

        occ.synchronize()

        # Fragment slab with gate prisms so they conform
        in_dimtags = [(3, dev)] + [(3, g) for g in gate_prisms]
        out_dimtags, out_map = occ.fragment([in_dimtags[0]], in_dimtags[1:])
        occ.synchronize()

        def dim3_tags(dimtags: List[Tuple[int, int]]) -> List[int]:
            tags = [t for (d, t) in dimtags if d == 3]
            seen = set()
            out = []
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out

        # Device (SiGe) volume tags after fragmentation
        dev_out = dim3_tags(out_map[0])
        if not dev_out:
            raise RuntimeError("Fragment produced no device volumes.")
        gmsh.model.addPhysicalGroup(3, dev_out, phys.SIGE)
        gmsh.model.setPhysicalName(3, phys.SIGE, "SiGe")

        # Assign each gate volume a unique physical id starting at phys.GATE0
        next_gid = phys.GATE0
        gate_out_vols_by_name: Dict[str, List[int]] = {}

        for i, name in enumerate(gate_names):
            gate_out = dim3_tags(out_map[1 + i])
            if not gate_out:
                print(f"[WARN] Gate '{name}' produced no volumes after fragment.")
                continue
            gid = next_gid
            next_gid += 1

            gate_out_vols_by_name[name] = gate_out
            gate_volume_tags[name] = gid

            gmsh.model.addPhysicalGroup(3, gate_out, gid)
            gmsh.model.setPhysicalName(3, gid, f"gate_vol_{name}")

        # Now tag ALL boundary surfaces of each gate volume as the electrode surface group
        # using the SAME id as the gate volume id (easy mapping).
        for name, vols in gate_out_vols_by_name.items():
            gid = gate_volume_tags[name]
            surf_tags: List[int] = []
            for vtag in vols:
                bnd = gmsh.model.getBoundary([(3, vtag)], oriented=False, recursive=False)
                for (d, s) in bnd:
                    if d == 2:
                        surf_tags.append(s)
            # Dedup
            surf_tags = sorted(set(surf_tags))
            if not surf_tags:
                print(f"[WARN] Gate '{name}' has no boundary surfaces??")
                continue
            gmsh.model.addPhysicalGroup(2, surf_tags, gid)
            gmsh.model.setPhysicalName(2, gid, f"gate_surf_{name}")
            gate_surface_tags[name] = gid

        # Outer boundaries: classify by COM
        surfs = gmsh.model.getEntities(dim=2)
        top_ground: List[int] = []
        bottom_ground: List[int] = []
        sides: List[int] = []

        # We want "outer top ground" as the portion of the slab top surface not covered by gate prisms.
        # The gate prisms create additional outer surfaces; those are already tagged as gate surfaces.
        # So for outer tags, we only tag surfaces that are NOT already part of any gate surface physical group.
        all_gate_surf_entities = set()
        for name, gid in gate_surface_tags.items():
            # Can't query back by id reliably without bookkeeping; we already built surf lists above,
            # but easiest: mark by checking physical groups later is hard. So: we re-classify and skip those
            # that lie above z_top (they belong to gate outer surfaces). We'll keep it simple:
            pass

        tol = 1e-9 * max(Lx, Ly, Lz, 1.0)

        for (dim, s) in surfs:
            cx, cy, cz = occ.getCenterOfMass(dim, s)

            # Skip gate outer surfaces: those are mostly at cz >= z_top (top of gates) or cz <= zbot - something (bottom of bottom gate)
            # but the slab top ground is exactly at cz=z_top and outside the top S stripe and BPBPB rectangles.
            on_top_plane = abs(cz - z_top) < tol
            on_bottom_plane = abs(cz - zbot) < tol

            # Identify slab top ground:
            if on_top_plane:
                # Exclude areas under top-extruded gates. Since gates are volumes above z_top,
                # the slab top plane still exists everywhere EXCEPT where it is shared with a gate volume.
                # Those shared surfaces are internal and may not appear in 'surfs' as boundary surfaces.
                # So what remains on boundary at z_top is indeed top ground.
                top_ground.append(s)
                continue

            if on_bottom_plane:
                bottom_ground.append(s)
                continue

            # Sides: everything else that is boundary and not the gate surfaces.
            # Many gate outer walls are "sides" too, but that is OK as long as we do not apply BC there.
            sides.append(s)

        if top_ground:
            gmsh.model.addPhysicalGroup(2, sorted(set(top_ground)), phys.TOP_GROUND)
            gmsh.model.setPhysicalName(2, phys.TOP_GROUND, "top_ground")
        if bottom_ground:
            gmsh.model.addPhysicalGroup(2, sorted(set(bottom_ground)), phys.BOTTOM_GROUND)
            gmsh.model.setPhysicalName(2, phys.BOTTOM_GROUND, "bottom_ground")
        if sides:
            gmsh.model.addPhysicalGroup(2, sorted(set(sides)), phys.SIDES)
            gmsh.model.setPhysicalName(2, phys.SIDES, "sides")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]
    gmsh.finalize()

    # Broadcast-like reconstruction of ids (deterministic ordering)
    # Note: actual tags are created only on rank 0, but DOLFINx meshtags are distributed.
    # We'll rebuild name->id mapping the same way as rank 0.
    gate_surface_tags_all: Dict[str, int] = {}
    gate_volume_tags_all: Dict[str, int] = {}
    # Fixed ordering: S_TOP, S_BOT, B0, P1, B1, P2, B2
    all_names = ["S_TOP", "S_BOT", "B0", "P1", "B1", "P2", "B2"]
    gid = phys.GATE0
    for nm in all_names:
        gate_surface_tags_all[nm] = gid
        gate_volume_tags_all[nm] = gid
        gid += 1

    return msh, ct, ft, gate_surface_tags_all, gate_volume_tags_all, phys


def main() -> None:
    ap = argparse.ArgumentParser(description="Outward-extruded S top/bottom gates + BPBPB prisms, with Poisson solve.")
    ap.add_argument("--outdir", type=str, default="Results/sbpbpb_extruded", help="Output directory.")
    ap.add_argument("--basename", type=str, default="sbpbpb_extruded", help="Basename for outputs.")

    ap.add_argument("--Lx", type=float, default=600e-9)
    ap.add_argument("--Ly", type=float, default=400e-9)
    ap.add_argument("--Lz", type=float, default=120e-9)

    ap.add_argument("--gate_thickness", type=float, default=25e-9, help="Extruded gate prism thickness (m).")
    ap.add_argument("--s_width", type=float, default=40e-9, help="S stripe width in y (m).")

    ap.add_argument("--finger_len_y", type=float, default=220e-9)
    ap.add_argument("--finger_wx", type=float, default=40e-9)
    ap.add_argument("--finger_gap", type=float, default=20e-9)

    ap.add_argument("--h", type=float, default=12e-9)
    ap.add_argument("--deg", type=int, default=2)

    ap.add_argument("--occ_scale", type=float, default=1e9,
                    help="Build CAD in scaled units to avoid OCC failures. Default 1e9 (nm).")

    ap.add_argument("--eps_r", type=float, default=11.7)
    ap.add_argument("--rho", type=float, default=0.0)

    # Outer BCs (keep defaults similar to your CIF script)
    ap.add_argument("--bc_top_ground", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom_ground", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")

    # Gate voltages
    ap.add_argument("--V_S_TOP", type=float, default=0.0)
    ap.add_argument("--V_S_BOT", type=float, default=0.0)
    ap.add_argument("--V_B0", type=float, default=-0.05)
    ap.add_argument("--V_P1", type=float, default=0.10)
    ap.add_argument("--V_B1", type=float, default=-0.05)
    ap.add_argument("--V_P2", type=float, default=0.10)
    ap.add_argument("--V_B2", type=float, default=-0.05)
    ap.add_argument("--default_gateV", type=float, default=0.0)

    args = ap.parse_args()
    args.outdir = normalize_outdir(args.outdir)

    msh, ct, ft, gate_surface_tags, gate_volume_tags, phys = build_geometry_extruded_gates(
        Lx_m=args.Lx, Ly_m=args.Ly, Lz_m=args.Lz,
        h_m=args.h,
        gate_thickness_m=args.gate_thickness,
        s_width_m=args.s_width,
        finger_len_y_m=args.finger_len_y,
        finger_wx_m=args.finger_wx,
        finger_gap_m=args.finger_gap,
        occ_scale=args.occ_scale,
        z_top_m=0.0,
        model_name="sbpbpb_extruded",
    )

    # Scale mesh coords back to meters
    msh.geometry.x[:] *= (1.0 / args.occ_scale)

    gateV = {
        "S_TOP": args.V_S_TOP,
        "S_BOT": args.V_S_BOT,
        "B0": args.V_B0,
        "P1": args.V_P1,
        "B1": args.V_B1,
        "P2": args.V_P2,
        "B2": args.V_B2,
    }

    outer_tags = {
        "top_ground": phys.TOP_GROUND,
        "bottom_ground": phys.BOTTOM_GROUND,
        "sides": phys.SIDES,
    }
    outer_modes = {
        "top_ground": args.bc_top_ground,
        "bottom_ground": args.bc_bottom_ground,
        "sides": args.bc_sides,
    }

    if RANK == 0:
        print("\n=== Gate surface tag summary (Dirichlet electrodes) ===")
        for nm, tag in gate_surface_tags.items():
            nfac = int(ft.find(tag).size)
            print(f"  {nm:6s} tag={tag:4d} nfacets={nfac:6d} V={gateV.get(nm, args.default_gateV): .3f}")
        print("\n=== Outer boundary tags ===")
        for nm, tag in outer_tags.items():
            print(f"  {nm:12s} tag={tag:4d} nfacets={int(ft.find(tag).size)} mode={outer_modes[nm]}")

    phi = solve_poisson_with_gate_surfaces(
        msh=msh,
        ft=ft,
        gate_surface_tags=gate_surface_tags,
        gateV=gateV,
        default_gateV=args.default_gateV,
        deg=args.deg,
        eps_r=args.eps_r,
        rho=args.rho,
        outer_bc_tags=outer_tags,
        outer_bc_mode=outer_modes,
    )

    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)
    if RANK == 0:
        print("\n=== Solve summary ===")
        print(f"phi min/max = [{gmin:.6e}, {gmax:.6e}]")

    os.makedirs(args.outdir, exist_ok=True)
    phi_path = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    tags_path = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")
    reg_path = os.path.join(args.outdir, f"{args.basename}_regions.xdmf")

    phi_out = function_for_xdmf(phi, msh)
    region_id = ct_to_dg0(msh, ct, name="region_id")

    with io.XDMFFile(COMM, phi_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_out)

    with io.XDMFFile(COMM, tags_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ft, msh)

    with io.XDMFFile(COMM, reg_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(region_id)

    if RANK == 0:
        print("\n=== Wrote files ===")
        print(f"  phi:        {phi_path}")
        print(f"  facet tags: {tags_path}")
        print(f"  regions:    {reg_path}")
        print("\nParaView:")
        print("  - Open *_regions.xdmf and color by region_id (see gate prisms extruded outward)")
        print("  - Open *_facet_tags.xdmf and color by 'gmsh:physical' (electrode surfaces)")
        print("  - Open *_phi.xdmf and color by phi")


if __name__ == "__main__":
    main()
