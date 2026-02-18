#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

from dolfinx.fem.petsc import LinearProblem as LinearProblem

COMM = MPI.COMM_WORLD
RANK = COMM.rank


@dataclass(frozen=True)
class Well2D:
    name: str
    xmin: float
    ymin: float
    dx: float
    dy: float


@dataclass(frozen=True)
class Phys:
    VOL: int = 1
    TOP: int = 10
    BOTTOM: int = 11
    SIDES: int = 12
    WELL0: int = 1000  # well i surfaces start here


def build_mesh_with_extruded_wells(
    Lx: float,
    Ly: float,
    z_top: float,
    Lz: float,
    wells: List[Well2D],
    well_depth: float,
    h: float,
    model_name: str = "extruded_wells",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags, Dict[str, int], Phys]:
    """
    Geometry:
      - Device volume: box [xmin,xmax]x[ymin,ymax]x[z_top-Lz, z_top]
      - Wells: rectangular prisms extruded downward from top plane by 'well_depth'
        We tag their *inner surfaces* (2D facets) so we can apply Dirichlet on one well.
    """
    phys = Phys()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    well_to_tag: Dict[str, int] = {}

    if RANK == 0:
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        xmin, xmax = -0.5 * Lx, 0.5 * Lx
        ymin, ymax = -0.5 * Ly, 0.5 * Ly
        zmin, zmax = z_top - Lz, z_top

        dev = occ.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin)
        occ.synchronize()

        # Create well solids (as separate volumes) by extruding top rectangles downward
        well_vols: List[Tuple[int, int]] = []
        well_side_surfs: Dict[str, List[int]] = {w.name: [] for w in wells}

        for i, w in enumerate(wells):
            r = occ.addRectangle(w.xmin, w.ymin, z_top, w.dx, w.dy)
            # Extrude down: returns list of new entities; includes lateral surfaces and a volume
            out = occ.extrude([(2, r)], 0.0, 0.0, -abs(well_depth))
            occ.synchronize()

            # Collect volume and lateral surfaces from extrusion output
            vols_i = [ent for ent in out if ent[0] == 3]
            if not vols_i:
                raise RuntimeError(f"Well '{w.name}' extrusion produced no volume.")
            well_vols.append(vols_i[0])

            # Lateral surfaces are dim=2 and not the top cap.
            surfs_i = [ent for ent in out if ent[0] == 2]
            # Filter out the top surface at z=z_top (cap) and bottom cap at z=z_top-well_depth
            # We only want the *walls* (side surfaces) as our "well boundary" for Dirichlet.
            walls = []
            for (dim, sid) in surfs_i:
                cx, cy, cz = occ.getCenterOfMass(dim, sid)
                if abs(cz - z_top) < 1e-6:
                    continue
                if abs(cz - (z_top - abs(well_depth))) < 1e-6:
                    continue
                walls.append(sid)
            well_side_surfs[w.name] = walls

            well_to_tag[w.name] = phys.WELL0 + i

        # Cut wells out of the device to make actual cavities
        if well_vols:
            # Cut device by well volumes
            cut = occ.cut([(3, dev)], well_vols, removeObject=True, removeTool=True)
            occ.synchronize()

        # Tag all remaining volumes as phys.VOL
        vols = gmsh.model.getEntities(dim=3)
        gmsh.model.addPhysicalGroup(3, [t for (_, t) in vols], phys.VOL)
        gmsh.model.setPhysicalName(3, phys.VOL, "device")

        # Tag outer boundary surfaces: top, bottom, sides
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

        # Tag well wall surfaces (these survive the cut as cavity walls if geometry matches)
        # We re-find them by nearest center-of-mass match is hard; instead we tag by bounding boxes.
        # So we do a robust approach: select surfaces whose CoM lies within each well footprint
        # and whose z is between (z_top-well_depth, z_top).
        tol = 1e-9
        for w in wells:
            tag = well_to_tag[w.name]
            ids = []
            for (dim, sid) in gmsh.model.getEntities(dim=2):
                cx, cy, cz = occ.getCenterOfMass(dim, sid)
                if (cx >= w.xmin - tol and cx <= (w.xmin + w.dx) + tol and
                    cy >= w.ymin - tol and cy <= (w.ymin + w.dy) + tol and
                    cz <= z_top - tol and cz >= (z_top - abs(well_depth)) + tol):
                    # This should hit the cavity wall surfaces
                    ids.append(sid)
            if ids:
                gmsh.model.addPhysicalGroup(2, ids, tag)
                gmsh.model.setPhysicalName(2, tag, f"well_{w.name}")
            else:
                print(f"[WARN] No cavity-wall facets tagged for well '{w.name}' (tag={tag}).")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)

    msh, ct, ft = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    gmsh.finalize()
    return msh, ct, ft, well_to_tag, phys


def solve_laplace(
    msh: dmesh.Mesh,
    ft: dmesh.MeshTags,
    phys: Phys,
    well_to_tag: Dict[str, int],
    well_perturb: str,
    Vwell: float,
    deg: int,
    eps_r: float,
    bc_bottom: str,
    bc_sides: str,
    bc_top: str,
) -> fem.Function:
    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps0 = 8.8541878128e-12
    eps = fem.Constant(msh, PETSc.ScalarType(eps0 * eps_r))

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx

    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    def add_tag_bc(tag: int, val: float):
        facets = ft.find(tag)
        if facets.size == 0:
            return
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(val), dofs, V))

    # Outer BCs
    if bc_top == "dirichlet0":
        add_tag_bc(phys.TOP, 0.0)
    if bc_bottom == "dirichlet0":
        add_tag_bc(phys.BOTTOM, 0.0)
    if bc_sides == "dirichlet0":
        add_tag_bc(phys.SIDES, 0.0)

    # Well perturbation BC
    if well_perturb not in well_to_tag:
        raise ValueError(f"--well_perturb '{well_perturb}' not in wells {list(well_to_tag.keys())}")
    add_tag_bc(well_to_tag[well_perturb], Vwell)

    problem = LinearProblem(
        a, L, bcs=bcs, u=phi,
        petsc_options_prefix="laplace_",
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
    ap = argparse.ArgumentParser(description="3D extruded cavities (wells) + apply 1V on one well boundary.")
    ap.add_argument("--outdir", default="Results/extruded_wells", type=str)
    ap.add_argument("--basename", default="extruded_wells", type=str)

    ap.add_argument("--Lx", type=float, default=650e-9)
    ap.add_argument("--Ly", type=float, default=510e-9)
    ap.add_argument("--z_top", type=float, default=0.0)
    ap.add_argument("--Lz", type=float, default=120e-9)

    ap.add_argument("--h", type=float, default=15e-9)
    ap.add_argument("--deg", type=int, default=1)
    ap.add_argument("--eps_r", type=float, default=11.7)

    ap.add_argument("--well_depth", type=float, default=60e-9, help="How deep wells are extruded into device.")
    ap.add_argument("--well", action="append", default=[],
                    help="Define well as name,xmin,ymin,dx,dy in meters. Repeatable.")
    ap.add_argument("--well_perturb", type=str, default="W0", help="Which well gets Dirichlet Vwell.")
    ap.add_argument("--Vwell", type=float, default=1.0)

    ap.add_argument("--bc_top", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")

    args = ap.parse_args()

    # Default: make 2 wells if none provided
    wells: List[Well2D] = []
    if args.well:
        for s in args.well:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) != 5:
                raise ValueError("--well must be name,xmin,ymin,dx,dy")
            name = parts[0]
            xmin, ymin, dx, dy = map(float, parts[1:])
            wells.append(Well2D(name=name, xmin=xmin, ymin=ymin, dx=dx, dy=dy))
    else:
        # Two example wells near center
        wells = [
            Well2D("W0", xmin=-50e-9, ymin=0e-9, dx=40e-9, dy=40e-9),
            Well2D("W1", xmin=+10e-9, ymin=0e-9, dx=40e-9, dy=40e-9),
        ]

    if RANK == 0:
        print("\n=== Extruded wells run ===")
        print(f"Domain: Lx={args.Lx:.3e} Ly={args.Ly:.3e} z in [{args.z_top-args.Lz:.3e}, {args.z_top:.3e}]")
        print(f"h={args.h:.3e} deg={args.deg} eps_r={args.eps_r}")
        print(f"well_depth={args.well_depth:.3e} Vwell={args.Vwell} on well '{args.well_perturb}'")
        print("Wells:")
        for w in wells:
            print(f"  {w.name}: xmin={w.xmin:.3e} ymin={w.ymin:.3e} dx={w.dx:.3e} dy={w.dy:.3e}")

    msh, ct, ft, well_to_tag, phys = build_mesh_with_extruded_wells(
        Lx=args.Lx, Ly=args.Ly, z_top=args.z_top, Lz=args.Lz,
        wells=wells, well_depth=args.well_depth, h=args.h
    )

    phi = solve_laplace(
        msh=msh, ft=ft, phys=phys, well_to_tag=well_to_tag,
        well_perturb=args.well_perturb, Vwell=args.Vwell,
        deg=args.deg, eps_r=args.eps_r,
        bc_bottom=args.bc_bottom, bc_sides=args.bc_sides, bc_top=args.bc_top
    )

    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)

    if RANK == 0:
        print(f"\nphi min/max = [{gmin:.6e}, {gmax:.6e}]")
        print("Well facet tags:")
        for k, v in well_to_tag.items():
            print(f"  {k}: {v}")

    os.makedirs(args.outdir, exist_ok=True)
    x_phi = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    x_ft = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")

    with io.XDMFFile(COMM, x_phi, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)

    with io.XDMFFile(COMM, x_ft, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ft, msh)

    if RANK == 0:
        print("\nParaView:")
        print(f"  open {x_phi} (phi)")
        print(f"  open {x_ft} and color by gmsh:physical (facet tags)")
        print("  confirm well_* surfaces exist and that only one has phi=1V")

def write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)

if __name__ == "__main__":
    main()
