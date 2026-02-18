from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import gmsh
from dolfinx import fem, io, mesh as dmesh

# Robust import across dolfinx versions
try:
    import dolfinx.io.gmshio as gmshio  # type: ignore
except Exception:  # noqa: BLE001
    from dolfinx.io import gmsh as gmshio  # type: ignore

import ufl


COMM = MPI.COMM_WORLD


@dataclass
class PhysTags:
    vol_bg: int
    vol_well: int
    vol_inners: List[int]
    surf_top: int
    surf_bottom: int


def _model_to_mesh_robust(model, comm, rank=0):
    out = gmshio.model_to_mesh(model, comm, rank=rank, gdim=3)
    if isinstance(out, tuple) and len(out) == 3:
        msh, ct, ft = out
        phys = {}
    elif isinstance(out, tuple) and len(out) == 4:
        msh, ct, ft, phys = out
    else:
        raise RuntimeError(f"Unexpected gmshio.model_to_mesh output type={type(out)} len={len(out)}")
    return msh, ct, ft, phys


def build_geometry(
    *,
    Lx: float,
    Ly: float,
    Lz: float,
    z0: float,
    z1: float,
    shape: str,
    square_w: float,
    circle_r: float,
    inner_circles: List[Tuple[float, float, float]],
    clearance: float,
    mesh_h: float,
    tag_offset: int = 1,
    occ_disable_check: bool = True,
) -> Tuple[PhysTags, Dict[str, int]]:
    gmsh.model.add("extruded_wells_3d_poisson")
    occ = gmsh.model.occ

    # OCC robustness options
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
    gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)
    if occ_disable_check:
        gmsh.option.setNumber("Geometry.OCCDisableBooleanCheck", 1)

    # Background box
    x0 = -0.5 * Lx
    y0 = -0.5 * Ly
    box = occ.addBox(x0, y0, 0.0, Lx, Ly, Lz)

    dz = z1 - z0
    if dz <= 0:
        raise ValueError("Need z1 > z0 for well extrusion thickness")

    # If main is circular and we have inner circles, avoid tangency/overlap:
    circle_r_eff = circle_r
    if shape in ("circle", "multi_circle") and len(inner_circles) > 0:
        worst = -np.inf
        for (cx, cy, rr) in inner_circles:
            dist = float(np.hypot(cx, cy))
            worst = max(worst, dist + rr + clearance)
        if worst >= circle_r_eff:
            circle_r_eff = worst + clearance
            if COMM.rank == 0:
                print(f"[GMSH] Adjusting main circle radius: {circle_r:.3e} -> {circle_r_eff:.3e} (avoid tangency/overlap)")

    # Main well volume
    if shape == "square":
        w = square_w
        well = occ.addBox(-0.5 * w, -0.5 * w, z0, w, w, dz)
    elif shape in ("circle", "multi_circle"):
        well = occ.addCylinder(0.0, 0.0, z0, 0.0, 0.0, dz, circle_r_eff)
    else:
        raise ValueError(f"Unknown shape={shape}")

    # Inner cylinders
    inner_vols = [occ.addCylinder(cx, cy, z0, 0.0, 0.0, dz, rr) for (cx, cy, rr) in inner_circles]

    # --- Sequential fragmentation (much more stable than one giant fragment) ---
    occ.synchronize()
    current_objects = [(3, box)]

    # First fragment with the main well
    try:
        occ.fragment(current_objects, [(3, well)], removeObject=False, removeTool=False)
    except Exception as e:
        raise RuntimeError(f"OCC fragment(box, well) failed. Try larger --clearance. Original error: {e}") from e
    occ.synchronize()

    # Then fragment with each inner cylinder one-by-one
    for i, v in enumerate(inner_vols):
        try:
            occ.fragment(current_objects, [(3, v)], removeObject=False, removeTool=False)
        except Exception as e:
            raise RuntimeError(
                f"OCC fragment with inner cylinder #{i} failed. "
                f"Try increasing --clearance (e.g. 5e-9) or slightly shifting positions. "
                f"Original error: {e}"
            ) from e
        occ.synchronize()

    # Now proceed with tagging by searching entities
    vol_entities = gmsh.model.getEntities(dim=3)

    def bbox_vol(tag):
        xb = gmsh.model.getBoundingBox(3, tag)
        return (xb[3] - xb[0]) * (xb[4] - xb[1]) * (xb[5] - xb[2])

    def vol_in_box(xmin, ymin, zmin, xmax, ymax, zmax, tol=1e-12):
        hits = []
        for (dim, tag) in vol_entities:
            xb = gmsh.model.getBoundingBox(dim, tag)
            if (xb[0] >= xmin - tol and xb[1] >= ymin - tol and xb[2] >= zmin - tol and
                xb[3] <= xmax + tol and xb[4] <= ymax + tol and xb[5] <= zmax + tol):
                hits.append(tag)
        return hits

    if shape == "square":
        w = square_w
        xmin, ymin, zmin = -0.5 * w, -0.5 * w, z0
        xmax, ymax, zmax =  0.5 * w,  0.5 * w, z1
    else:
        r = circle_r_eff
        xmin, ymin, zmin = -r, -r, z0
        xmax, ymax, zmax =  r,  r, z1

    well_hits = vol_in_box(xmin, ymin, zmin, xmax, ymax, zmax)
    if len(well_hits) == 0:
        raise RuntimeError("Failed to identify main well volume after fragment")
    well_tag = max(well_hits, key=bbox_vol)

    inner_tags = []
    for (cx, cy, rr) in inner_circles:
        hits = vol_in_box(cx - rr, cy - rr, z0, cx + rr, cy + rr, z1)
        if len(hits) == 0:
            raise RuntimeError(f"Failed to identify inner cylinder at ({cx},{cy}) r={rr}")
        inner_tags.append(max(hits, key=bbox_vol))

    bg_tag = max([t for (_, t) in vol_entities], key=bbox_vol)

    # Tag top/bottom surfaces
    surf_entities = gmsh.model.getEntities(dim=2)

    def surfaces_at_z(z_target, tol=1e-9):
        hits = []
        for (dim, tag) in surf_entities:
            xb = gmsh.model.getBoundingBox(dim, tag)
            if abs(xb[2] - z_target) < tol and abs(xb[5] - z_target) < tol:
                hits.append(tag)
        return hits

    top_surfs = surfaces_at_z(Lz)
    bot_surfs = surfaces_at_z(0.0)
    if len(top_surfs) == 0 or len(bot_surfs) == 0:
        raise RuntimeError("Failed to identify top/bottom boundary surfaces")

    # Physical IDs
    vol_bg_id = tag_offset
    vol_well_id = tag_offset + 1
    vol_inner_ids = [tag_offset + 2 + i for i in range(len(inner_tags))]
    surf_top_id = tag_offset + 200
    surf_bot_id = tag_offset + 201

    gmsh.model.addPhysicalGroup(3, [bg_tag], vol_bg_id)
    gmsh.model.setPhysicalName(3, vol_bg_id, "VOL_BG")

    gmsh.model.addPhysicalGroup(3, [well_tag], vol_well_id)
    gmsh.model.setPhysicalName(3, vol_well_id, "VOL_WELL")

    for pid, vtag in zip(vol_inner_ids, inner_tags):
        gmsh.model.addPhysicalGroup(3, [vtag], pid)
        gmsh.model.setPhysicalName(3, pid, f"VOL_INNER_{pid}")

    gmsh.model.addPhysicalGroup(2, top_surfs, surf_top_id)
    gmsh.model.setPhysicalName(2, surf_top_id, "SURF_TOP")

    gmsh.model.addPhysicalGroup(2, bot_surfs, surf_bot_id)
    gmsh.model.setPhysicalName(2, surf_bot_id, "SURF_BOTTOM")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_h)
    gmsh.model.mesh.generate(3)

    tags = PhysTags(
        vol_bg=vol_bg_id,
        vol_well=vol_well_id,
        vol_inners=vol_inner_ids,
        surf_top=surf_top_id,
        surf_bottom=surf_bot_id,
    )

    name_to_tag = {"VOL_BG": vol_bg_id, "VOL_WELL": vol_well_id, "SURF_TOP": surf_top_id, "SURF_BOTTOM": surf_bot_id}
    for pid in vol_inner_ids:
        name_to_tag[f"VOL_INNER_{pid}"] = pid

    return tags, name_to_tag


def solve_poisson(
    msh: dmesh.Mesh,
    cell_tags: dmesh.MeshTags,
    facet_tags: dmesh.MeshTags,
    phys: PhysTags,
    *,
    p: int,
    eps_bg: float,
    eps_well: float,
    eps_inner: float,
    Vtop: float,
    Vbottom: float,
    out_prefix: str,
):
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)

    V = fem.functionspace(msh, ("CG", p))
    Q = fem.functionspace(msh, ("DG", 0))

    eps = fem.Function(Q)
    eps.x.array[:] = eps_bg

    if cell_tags is not None:
        tag_to_eps: Dict[int, float] = {phys.vol_bg: eps_bg, phys.vol_well: eps_well}
        for pid in phys.vol_inners:
            tag_to_eps[pid] = eps_inner
        for tag_id, val in tag_to_eps.items():
            cells = cell_tags.find(tag_id)
            if len(cells) > 0:
                eps.x.array[cells] = val

    bcs = []
    u_top = fem.Function(V); u_top.x.array[:] = Vtop
    u_bot = fem.Function(V); u_bot.x.array[:] = Vbottom

    def add_bc(tag_id: int, val_fun: fem.Function, name: str):
        facets = facet_tags.find(tag_id)
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        ndofs_global = msh.comm.allreduce(len(dofs), op=MPI.SUM)
        if msh.comm.rank == 0:
            print(f"[BC] {name}: tag_id={tag_id} dofs_global={ndofs_global}")
        if ndofs_global == 0:
            raise RuntimeError(f"BC '{name}' got 0 dofs globally (missing facet tag?)")
        bcs.append(fem.dirichletbc(val_fun, dofs))

    add_bc(phys.surf_top, u_top, "TOP")
    add_bc(phys.surf_bottom, u_bot, "BOTTOM")

    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx

    problem = fem.petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10, "ksp_atol": 1e-14},
    )
    uh = problem.solve()
    uh.name = "phi"

    local_min = np.min(uh.x.array) if uh.x.array.size else np.inf
    local_max = np.max(uh.x.array) if uh.x.array.size else -np.inf
    gmin = msh.comm.allreduce(local_min, op=MPI.MIN)
    gmax = msh.comm.allreduce(local_max, op=MPI.MAX)
    if msh.comm.rank == 0:
        print(f"[SOL] phi min={gmin:.6e}, max={gmax:.6e}")

    with io.XDMFFile(msh.comm, f"{out_prefix}_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        if cell_tags is not None:
            xdmf.write_meshtags(cell_tags, msh.geometry)
        if facet_tags is not None:
            xdmf.write_meshtags(facet_tags, msh.geometry)

    with io.XDMFFile(msh.comm, f"{out_prefix}_phi.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(uh)

    return uh


def parse_inner_circles(s: str) -> List[Tuple[float, float, float]]:
    s = s.strip()
    if not s:
        return []
    items = s.split(";")
    out = []
    for it in items:
        it = it.strip().strip("()")
        parts = [p.strip() for p in it.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad inner circle entry: {it}")
        out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["square", "circle", "multi_circle"], default="square")
    ap.add_argument("--Lx", type=float, default=300e-9)
    ap.add_argument("--Ly", type=float, default=300e-9)
    ap.add_argument("--Lz", type=float, default=200e-9)
    ap.add_argument("--z0", type=float, default=40e-9)
    ap.add_argument("--z1", type=float, default=48e-9)
    ap.add_argument("--square_w", type=float, default=120e-9)
    ap.add_argument("--circle_r", type=float, default=60e-9)
    ap.add_argument("--inner_circles", type=str, default="")
    ap.add_argument("--clearance", type=float, default=2e-9, help="boolean clearance (avoid tangency)")
    ap.add_argument("--h", type=float, default=10e-9)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--eps_bg", type=float, default=11.7)
    ap.add_argument("--eps_well", type=float, default=11.7)
    ap.add_argument("--eps_inner", type=float, default=11.7)
    ap.add_argument("--Vtop", type=float, default=0.0)
    ap.add_argument("--Vbottom", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="out_extruded_wells")
    args = ap.parse_args()

    inner = parse_inner_circles(args.inner_circles)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if COMM.rank == 0 else 0)

    try:
        phys, name_to_tag = build_geometry(
            Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
            z0=args.z0, z1=args.z1,
            shape=args.shape,
            square_w=args.square_w,
            circle_r=args.circle_r,
            inner_circles=inner,
            clearance=args.clearance,
            mesh_h=args.h,
            tag_offset=1,
            occ_disable_check=True,
        )
        if COMM.rank == 0:
            print("[TAGS] Physical tag map:")
            for k, v in name_to_tag.items():
                print(f"  {k:>12s}: {v}")

        msh, ct, ft, _ = _model_to_mesh_robust(gmsh.model, COMM, rank=0)
    finally:
        gmsh.finalize()

    solve_poisson(
        msh, ct, ft, phys,
        p=args.p,
        eps_bg=args.eps_bg,
        eps_well=args.eps_well,
        eps_inner=args.eps_inner,
        Vtop=args.Vtop,
        Vbottom=args.Vbottom,
        out_prefix=args.out,
    )


if __name__ == "__main__":
    main()
