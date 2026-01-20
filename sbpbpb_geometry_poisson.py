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


@dataclass(frozen=True)
class Rect2D:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains_xy(self, x: float, y: float, tol: float = 0.0) -> bool:
        return (self.xmin - tol <= x <= self.xmax + tol) and (self.ymin - tol <= y <= self.ymax + tol)


def _write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


def build_box_mesh_nm(
    Lx_nm: float,
    Ly_nm: float,
    Lz_nm: float,
    h_nm: float,
    model_name: str = "sbpbpb",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    """
    Build a simple 3D box in Gmsh units = nm.
    Domain:
      x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2], z in [-Lz, 0]
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)
    gmsh.model.add(model_name)

    occ = gmsh.model.occ

    xmin, ymin, zmin = -0.5 * Lx_nm, -0.5 * Ly_nm, -Lz_nm
    dx, dy, dz = Lx_nm, Ly_nm, Lz_nm
    vol = occ.addBox(xmin, ymin, zmin, dx, dy, dz)
    occ.synchronize()

    # Physical group for volume (not strictly needed, but nice)
    gmsh.model.addPhysicalGroup(3, [vol], 1)
    gmsh.model.setPhysicalName(3, 1, "SiGe")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_nm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_nm)
    gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]
    gmsh.finalize()
    return msh, ct, ft


def tag_facets_by_rectangles(
    msh: dmesh.Mesh,
    z_top: float,
    z_bottom: float,
    rects_top: List[Rect2D],
    rects_bottom: List[Rect2D],
    tol_z: float,
    tol_xy: float,
    gate_tag_start: int = 100,
    tag_top_ground: int = 10,
    tag_bottom_ground: int = 11,
    tag_sides: int = 12,
) -> Tuple[dmesh.MeshTags, Dict[str, int], Dict[int, str]]:
    """
    Create facet tags for:
      - top plane z=z_top: rectangles for gates + background (TOP_GROUND)
      - bottom plane z=z_bottom: rectangles for gates + background (BOTTOM_GROUND)
      - all side walls: SIDES
    Rectangles are in *mesh coordinates* (after any scaling you apply).
    """
    fdim = msh.topology.dim - 1

    # --- locate top, bottom, side facets ---
    def is_top(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[2], z_top, atol=tol_z)

    def is_bottom(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[2], z_bottom, atol=tol_z)

    top_facets = dmesh.locate_entities_boundary(msh, fdim, is_top)
    bot_facets = dmesh.locate_entities_boundary(msh, fdim, is_bottom)

    # sides = boundary facets not on top/bottom
    top_set = set(int(i) for i in top_facets.tolist())
    bot_set = set(int(i) for i in bot_facets.tolist())

    def is_boundary(x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[1], True, dtype=bool)

    all_bfacets = dmesh.locate_entities_boundary(msh, fdim, is_boundary)
    side_facets = np.array([f for f in all_bfacets.tolist() if (int(f) not in top_set) and (int(f) not in bot_set)], dtype=np.int32)

    # --- allocate tags ---
    name_to_tag: Dict[str, int] = {}
    tag_to_name: Dict[int, str] = {}

    name_to_tag["TOP_GROUND"] = tag_top_ground
    tag_to_name[tag_top_ground] = "TOP_GROUND"
    name_to_tag["BOTTOM_GROUND"] = tag_bottom_ground
    tag_to_name[tag_bottom_ground] = "BOTTOM_GROUND"
    name_to_tag["SIDES"] = tag_sides
    tag_to_name[tag_sides] = "SIDES"

    next_gate_tag = gate_tag_start
    for r in rects_top + rects_bottom:
        if r.name not in name_to_tag:
            name_to_tag[r.name] = next_gate_tag
            tag_to_name[next_gate_tag] = r.name
            next_gate_tag += 1

    # --- classify facets ---
    # start with background labels for top/bottom, plus side label for sides
    all_facets = np.concatenate([top_facets, bot_facets, side_facets]).astype(np.int32)
    all_values = np.empty_like(all_facets, dtype=np.int32)

    # initialize
    all_values[: len(top_facets)] = tag_top_ground
    all_values[len(top_facets) : len(top_facets) + len(bot_facets)] = tag_bottom_ground
    all_values[len(top_facets) + len(bot_facets) :] = tag_sides

    # helper: get facet centers
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    X = msh.geometry.x

    def facet_centroid(fid: int) -> Tuple[float, float, float]:
        vs = f2v.links(fid)
        xc = X[vs].mean(axis=0)
        return float(xc[0]), float(xc[1]), float(xc[2])

    # classify top rectangles
    top_idx = {int(f): i for i, f in enumerate(top_facets.tolist())}
    for fid in top_facets.tolist():
        x, y, z = facet_centroid(int(fid))
        # last match wins (so you can layer rectangles if needed)
        for r in rects_top:
            if r.contains_xy(x, y, tol=tol_xy):
                all_values[top_idx[int(fid)]] = name_to_tag[r.name]

    # classify bottom rectangles
    bot_offset = len(top_facets)
    bot_idx = {int(f): bot_offset + i for i, f in enumerate(bot_facets.tolist())}
    for fid in bot_facets.tolist():
        x, y, z = facet_centroid(int(fid))
        for r in rects_bottom:
            if r.contains_xy(x, y, tol=tol_xy):
                all_values[bot_idx[int(fid)]] = name_to_tag[r.name]

    # build MeshTags sorted by facet id
    sort = np.argsort(all_facets)
    mt = dmesh.meshtags(msh, fdim, all_facets[sort], all_values[sort])

    return mt, name_to_tag, tag_to_name


def solve_poisson_dirichlet_patches(
    msh: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
    tag_to_voltage: Dict[int, float],
    deg: int,
    eps_r: float,
    rho: float,
) -> fem.Function:
    """
    Solve: -div(eps*grad(phi)) = rho in volume,
    with Dirichlet conditions on all boundary facets according to tag_to_voltage,
    defaulting to 0 for any tags not present in the dict.
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

    # Create BCs tag-by-tag
    bcs = []
    unique_tags = np.unique(facet_tags.values)
    for t in unique_tags:
        facets_t = facet_tags.indices[facet_tags.values == t]
        if facets_t.size == 0:
            continue
        dofs_t = fem.locate_dofs_topological(V, fdim, facets_t)
        Vt = float(tag_to_voltage.get(int(t), 0.0))
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vt), dofs_t, V))

    problem = _LinearProblem(
        a, L, bcs=bcs, u=phi,
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-12,
            "ksp_atol": 1.0e-14,
        },
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


def interpolate_to_cg1_if_needed(phi: fem.Function) -> fem.Function:
    """
    XDMF often expects output degree to match mesh geometry degree.
    A safe workaround: write CG1 copy when phi degree > 1.
    """
    try:
        deg = int(phi.function_space.ufl_element().degree)
    except Exception:
        # fallback
        deg = 1
    if deg <= 1:
        return phi

    msh = phi.function_space.mesh
    V1 = fem.functionspace(msh, ("CG", 1))
    phi1 = fem.Function(V1, name="phi")
    phi1.interpolate(phi)
    return phi1


def write_top_view_gate_id(
    msh: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
    name_to_tag: Dict[str, int],
    rects_top: List[Rect2D],
    z_top: float,
    tol_z: float,
    tol_xy: float,
    outpath: str,
) -> None:
    """
    Create a 2D submesh of the top boundary facets and write DG0 "gate_id" by centroid classification.
    No EntityMap gymnastics. Just classify on the submesh.
    """
    fdim = msh.topology.dim - 1

    def is_top(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[2], z_top, atol=tol_z)

    top_facets = dmesh.locate_entities_boundary(msh, fdim, is_top)
    sub = dmesh.create_submesh(msh, fdim, top_facets)
    smesh = sub[0]

    tdim = smesh.topology.dim
    smesh.topology.create_connectivity(tdim, 0)
    c2v = smesh.topology.connectivity(tdim, 0)

    V0 = fem.functionspace(smesh, ("DG", 0))
    gate_id = fem.Function(V0, name="gate_id")

    coords = smesh.geometry.x
    im = smesh.topology.index_map(tdim)
    ncells_local = im.size_local + im.num_ghosts
    ncells_local = min(ncells_local, gate_id.x.array.size)

    bg = name_to_tag.get("TOP_GROUND", 0)
    vals = np.full(ncells_local, bg, dtype=gate_id.x.array.dtype)

    for c in range(ncells_local):
        vs = c2v.links(c)
        xc = coords[vs].mean(axis=0)
        x, y = float(xc[0]), float(xc[1])
        tag_here = bg
        for r in rects_top:
            if r.contains_xy(x, y, tol=tol_xy):
                tag_here = name_to_tag[r.name]
        vals[c] = tag_here

    gate_id.x.array[:ncells_local] = vals

    with io.XDMFFile(COMM, outpath, "w") as xdmf:
        xdmf.write_mesh(smesh)
        xdmf.write_function(gate_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="SBPBPB demo: top/bottom S gates + BPBPB fingers (nm-in-Gmsh, post-scale)")
    ap.add_argument("--outdir", type=str, default="Results/sbpbpb_demo")
    ap.add_argument("--basename", type=str, default="sbpbpb_demo")

    # Geometry in nm (inside Gmsh)
    ap.add_argument("--Lx_nm", type=float, default=300.0, help="Domain x-size in nm (Gmsh units).")
    ap.add_argument("--Ly_nm", type=float, default=300.0, help="Domain y-size in nm (Gmsh units).")
    ap.add_argument("--Lz_nm", type=float, default=120.0, help="Domain thickness in nm (Gmsh units).")
    ap.add_argument("--h_nm", type=float, default=12.0, help="Mesh size in nm (Gmsh units).")

    # Post-scale to meters (optional)
    ap.add_argument("--scale_post", type=float, default=1e-9, help="Multiply coordinates after import. 1e-9 converts nm to m.")

    # Gate layout parameters (nm, in mesh coordinates after post-scale is applied)
    ap.add_argument("--S_width_nm", type=float, default=40.0, help="Width of S strips along y, in nm.")
    ap.add_argument("--finger_w_nm", type=float, default=40.0, help="Finger width along x, in nm.")
    ap.add_argument("--finger_h_nm", type=float, default=75.0, help="Finger height along y, in nm.")
    ap.add_argument("--finger_gap_nm", type=float, default=20.0, help="Gap between adjacent fingers along x, in nm.")
    ap.add_argument("--finger_y0_nm", type=float, default=0.0, help="Finger center y location, in nm.")

    # Physics
    ap.add_argument("--deg", type=int, default=2)
    ap.add_argument("--eps_r", type=float, default=11.7)
    ap.add_argument("--rho", type=float, default=0.0)

    # Voltages
    ap.add_argument("--V_STOP", type=float, default=0.0)
    ap.add_argument("--V_SBOT", type=float, default=0.0)
    ap.add_argument("--V_B0", type=float, default=-0.05)
    ap.add_argument("--V_P1", type=float, default=0.10)
    ap.add_argument("--V_B1", type=float, default=-0.05)
    ap.add_argument("--V_P2", type=float, default=0.10)
    ap.add_argument("--V_B2", type=float, default=-0.05)

    ap.add_argument("--side_dirichlet", type=float, default=0.0, help="Dirichlet value on side walls.")
    ap.add_argument("--top_ground", type=float, default=0.0, help="Dirichlet background on top plane outside gates.")
    ap.add_argument("--bottom_ground", type=float, default=0.0, help="Dirichlet background on bottom plane outside gates.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if RANK == 0:
        print("\n=== Build params (Gmsh in nm units) ===")
        print(f"Domain: Lx={args.Lx_nm} nm, Ly={args.Ly_nm} nm, Lz={args.Lz_nm} nm")
        print(f"Mesh h={args.h_nm} nm, deg={args.deg}")
        print(f"Post-scale = {args.scale_post:g} (so coords become meters if 1e-9)")

    # 1) Build mesh in nm units inside gmsh
    msh, ct, ft = build_box_mesh_nm(args.Lx_nm, args.Ly_nm, args.Lz_nm, args.h_nm)

    # 2) Post-scale coordinates
    msh.geometry.x[:] *= args.scale_post

    # Mesh coordinates after scaling
    Lx = args.Lx_nm * args.scale_post
    Ly = args.Ly_nm * args.scale_post
    Lz = args.Lz_nm * args.scale_post
    z_top = 0.0
    z_bottom = -Lz

    # Tolerances in scaled units
    tol_z = 1.0e-12 * max(Lx, Ly, Lz, 1.0)
    tol_xy = 1.0e-12 * max(Lx, Ly, 1.0)

    # 3) Define rectangles (in scaled coordinates)
    S_w = args.S_width_nm * args.scale_post
    fw = args.finger_w_nm * args.scale_post
    fh = args.finger_h_nm * args.scale_post
    gap = args.finger_gap_nm * args.scale_post
    fy0 = args.finger_y0_nm * args.scale_post

    # STOP strip near +Ly/2, SBOT strip near -Ly/2
    rects_top: List[Rect2D] = []
    rects_bottom: List[Rect2D] = []

    y_top0 = 0.5 * Ly - S_w
    y_top1 = 0.5 * Ly
    rects_top.append(Rect2D("STOP", xmin=-0.5 * Lx, xmax=0.5 * Lx, ymin=y_top0, ymax=y_top1))

    y_bot0 = -0.5 * Ly
    y_bot1 = -0.5 * Ly + S_w
    rects_bottom.append(Rect2D("SBOT", xmin=-0.5 * Lx, xmax=0.5 * Lx, ymin=y_bot0, ymax=y_bot1))

    # BPBPB fingers: 5 centered along x, alternating names
    names = ["B0", "P1", "B1", "P2", "B2"]
    total_w = 5 * fw + 4 * gap
    x0 = -0.5 * total_w
    y0 = fy0 - 0.5 * fh
    y1 = fy0 + 0.5 * fh
    for i, nm in enumerate(names):
        xi0 = x0 + i * (fw + gap)
        xi1 = xi0 + fw
        rects_top.append(Rect2D(nm, xmin=xi0, xmax=xi1, ymin=y0, ymax=y1))

    # 4) Tag facets: top/bottom rectangles + backgrounds + sides
    facet_tags, name_to_tag, tag_to_name = tag_facets_by_rectangles(
        msh=msh,
        z_top=z_top,
        z_bottom=z_bottom,
        rects_top=rects_top,
        rects_bottom=rects_bottom,
        tol_z=tol_z,
        tol_xy=tol_xy,
        gate_tag_start=100,
        tag_top_ground=10,
        tag_bottom_ground=11,
        tag_sides=12,
    )

    # Build tag -> voltage map
    tagV: Dict[int, float] = {}
    tagV[name_to_tag["TOP_GROUND"]] = float(args.top_ground)
    tagV[name_to_tag["BOTTOM_GROUND"]] = float(args.bottom_ground)
    tagV[name_to_tag["SIDES"]] = float(args.side_dirichlet)

    tagV[name_to_tag["STOP"]] = float(args.V_STOP)
    tagV[name_to_tag["SBOT"]] = float(args.V_SBOT)
    tagV[name_to_tag["B0"]] = float(args.V_B0)
    tagV[name_to_tag["P1"]] = float(args.V_P1)
    tagV[name_to_tag["B1"]] = float(args.V_B1)
    tagV[name_to_tag["P2"]] = float(args.V_P2)
    tagV[name_to_tag["B2"]] = float(args.V_B2)

    # Print facet summary
    if RANK == 0:
        print("\n=== Facet tag summary ===")
        for t in np.unique(facet_tags.values):
            n = int(np.sum(facet_tags.values == t))
            nm = tag_to_name.get(int(t), f"tag{int(t)}")
            vv = tagV.get(int(t), 0.0)
            print(f"  {nm:12s} tag={int(t):4d}  nfacets={n:6d}  V={vv:g}")

    # 5) Solve
    phi = solve_poisson_dirichlet_patches(
        msh=msh,
        facet_tags=facet_tags,
        tag_to_voltage=tagV,
        deg=args.deg,
        eps_r=args.eps_r,
        rho=args.rho,
    )

    # min/max
    local_min = float(np.min(phi.x.array))
    local_max = float(np.max(phi.x.array))
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)
    if RANK == 0:
        print("\n=== Solve summary ===")
        print(f"phi min/max = [{gmin:.6e}, {gmax:.6e}]")

    # 6) Output
    base = os.path.join(args.outdir, args.basename)
    xdmf_phi = base + "_phi.xdmf"
    xdmf_tags = base + "_facet_tags.xdmf"
    xdmf_top = base + "_top_view_gate_id.xdmf"
    map_txt = base + "_gate_map.txt"

    # XDMF phi (CG1 if needed)
    phi_out = interpolate_to_cg1_if_needed(phi)
    if RANK == 0 and phi_out is not phi:
        try:
            sol_deg = int(phi.function_space.ufl_element().degree)
        except Exception:
            sol_deg = args.deg
        print(f"[INFO] XDMF requires degree match: writing interpolated CG1 (phi degree was {sol_deg}).")

    with io.XDMFFile(COMM, xdmf_phi, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_out)

    with io.XDMFFile(COMM, xdmf_tags, "w") as xdmf:
        xdmf.write_mesh(msh)
        _write_meshtags_compat(xdmf, facet_tags, msh)

    write_top_view_gate_id(
        msh=msh,
        facet_tags=facet_tags,
        name_to_tag=name_to_tag,
        rects_top=rects_top,
        z_top=z_top,
        tol_z=tol_z,
        tol_xy=tol_xy,
        outpath=xdmf_top,
    )
    if RANK == 0:
        print(f"[INFO] Wrote top-view layout (centroid classification): {xdmf_top}")

    # Human-readable mapping file
    if RANK == 0:
        with open(map_txt, "w", encoding="utf-8") as f:
            f.write("Gate map (tags, voltages, rectangles) in mesh coordinates (after post-scale)\n\n")
            for nm in ["TOP_GROUND", "STOP", "B0", "P1", "B1", "P2", "B2", "SBOT", "BOTTOM_GROUND", "SIDES"]:
                if nm not in name_to_tag:
                    continue
                t = name_to_tag[nm]
                vv = tagV.get(t, 0.0)
                f.write(f"{nm:14s} tag={t:4d}  V={vv: .6g}\n")
                # list rectangles if any
                rr = [r for r in (rects_top + rects_bottom) if r.name == nm]
                for r in rr:
                    f.write(f"    rect: x=[{r.xmin:.6e},{r.xmax:.6e}] y=[{r.ymin:.6e},{r.ymax:.6e}]\n")
        print(f"[INFO] Wrote gate map: {map_txt}")

    if RANK == 0:
        print("\n=== Wrote files ===")
        print(f"  phi solution:   {xdmf_phi}")
        print(f"  facet tags:     {xdmf_tags}")
        print(f"  top view tags:  {xdmf_top}")
        print("\nParaView tips:")
        print("  - Open *_top_view_gate_id.xdmf for a clean 2D layout.")
        print("  - Open *_facet_tags.xdmf and color by 'gmsh:physical' to see 3D boundary patches.")
        print("  - To label tags: Color Map Editor -> 'Annotations' -> add tag value/name pairs (use *_gate_map.txt).")


if __name__ == "__main__":
    main()
