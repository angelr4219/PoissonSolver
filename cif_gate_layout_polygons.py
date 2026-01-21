#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
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

from dolfinx.fem.petsc import LinearProblem as _LinearProblem

COMM = MPI.COMM_WORLD
RANK = COMM.rank


DEFAULT_CIF_TEXT = r"""
Layer base;
Box 650 510 0 35;

Layer A1;
Box 100 75 -270 77.5;

Layer B0;
Box 40 75 -180 77.5;

Layer P1;
Box 40 75 -120 77.5;

Layer B1;
Box 40 75 -60 77.5;

Layer P2;
Box 40 75 0 77.5;

Layer B2;
Box 40 75 60 77.5;

Layer P3;
Box 40 75 120 77.5;

Layer B3;
Box 40 75 180 77.5;

Layer A2;
Box 100 75 270 77.5;

Layer CG0A;
Box 640 40 0 155;

Layer SG;
Box 640 40 0 0;

Layer A1_2;
Box 100 75 -150 -77.5;

Layer B1_2;
Box 40 75 -60 -77.5;

Layer P2_2;
Box 40 75 0 -77.5;

Layer B2_2;
Box 40 75 60 -77.5;

Layer A2_2;
Box 100 75 150 -77.5;

END
""".strip()


@dataclass(frozen=True)
class Box2D:
    layer: str
    W: float
    H: float
    xc: float
    yc: float

    @property
    def xmin(self) -> float:
        return self.xc - 0.5 * self.W

    @property
    def xmax(self) -> float:
        return self.xc + 0.5 * self.W

    @property
    def ymin(self) -> float:
        return self.yc - 0.5 * self.H

    @property
    def ymax(self) -> float:
        return self.yc + 0.5 * self.H


def parse_cif_boxes(text: str) -> List[Box2D]:
    text_wo_paren = re.sub(r"\(.*?\)", "", text, flags=re.DOTALL)
    layer_re = re.compile(r"^\s*Layer\s+([A-Za-z0-9_]+)\s*;\s*$")
    box_re = re.compile(
        r"^\s*Box\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*;\s*$"
    )
    end_re = re.compile(r"^\s*END\b", re.IGNORECASE)

    cur_layer = None
    boxes: List[Box2D] = []
    for raw in text_wo_paren.splitlines():
        line = raw.strip()
        if not line:
            continue
        if end_re.match(line):
            break

        mL = layer_re.match(line)
        if mL:
            cur_layer = mL.group(1)
            continue

        mB = box_re.match(line)
        if mB:
            if cur_layer is None:
                raise ValueError(f"Found Box before any Layer: '{line}'")
            W = float(mB.group(1))
            H = float(mB.group(2))
            xc = float(mB.group(3))
            yc = float(mB.group(4))
            boxes.append(Box2D(cur_layer, W, H, xc, yc))
            continue

    return boxes


def write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


def _unique_in_order(names: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def parse_gate_voltage_pairs(pairs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"--gateV entries must be like Layer=V, got '{s}'")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Bad --gateV key in '{s}'")
        out[k] = float(v)
    return out


def parse_gate_ngon_pairs(pairs: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"--gateN entries must be like Layer=N, got '{s}'")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Bad --gateN key in '{s}'")
        n = int(v)
        if n < 3:
            raise ValueError(f"Need N>=3 for polygons, got {n} for layer {k}")
        out[k] = n
    return out


def _degree_compat(deg_or_callable) -> int:
    try:
        return int(deg_or_callable())
    except TypeError:
        return int(deg_or_callable)


def normalize_outdir(outdir: str) -> str:
    if outdir == "results":
        return "Results"
    if outdir.startswith("results/"):
        return "Results/" + outdir[len("results/") :]
    return outdir


@dataclass(frozen=True)
class PhysTags:
    # volumes
    SIGE: int = 1
    CAP: int = 2
    # outer boundary facets
    TOP_GROUND: int = 10
    BOTTOM: int = 11
    SIDES: int = 12
    # gate facets start
    GATE0: int = 100


# ----------------------------
# Polygon helpers
# ----------------------------
def regular_ngon_vertices(xc: float, yc: float, n: int, R: float, rot_deg: float = 0.0) -> np.ndarray:
    th0 = np.deg2rad(rot_deg)
    k = np.arange(n, dtype=float)
    th = th0 + 2.0 * np.pi * k / n
    x = xc + R * np.cos(th)
    y = yc + R * np.sin(th)
    return np.stack([x, y], axis=1)


def point_in_polygon_2d(x: float, y: float, poly_xy: np.ndarray) -> bool:
    n = poly_xy.shape[0]
    inside = False
    x0, y0 = poly_xy[-1, 0], poly_xy[-1, 1]
    for i in range(n):
        x1, y1 = poly_xy[i, 0], poly_xy[i, 1]
        cond = ((y1 > y) != (y0 > y)) and (x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-300) + x1)
        if cond:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = (wx * vx + wy * vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t * vx, ay + t * vy
    return float(np.hypot(px - cx, py - cy))


def point_in_or_on_polygon_2d(x: float, y: float, poly_xy: np.ndarray, tol: float) -> bool:
    if point_in_polygon_2d(x, y, poly_xy):
        return True
    # treat boundary as inside (important for vertical walls whose COM sits on an edge)
    n = poly_xy.shape[0]
    for i in range(n):
        ax, ay = float(poly_xy[i, 0]), float(poly_xy[i, 1])
        bx, by = float(poly_xy[(i + 1) % n, 0]), float(poly_xy[(i + 1) % n, 1])
        if _dist_point_to_segment(x, y, ax, ay, bx, by) <= tol:
            return True
    return False


def polygon_footprint_for_box(b: Box2D, n: int, shrink: float, rot_deg: float) -> np.ndarray:
    R = 0.5 * shrink * min(b.W, b.H)
    return regular_ngon_vertices(b.xc, b.yc, n=n, R=R, rot_deg=rot_deg)


def occ_add_regular_ngon_surface_from_box(occ, b: Box2D, z: float, n: int, h: float, shrink: float, rot_deg: float) -> int:
    poly = polygon_footprint_for_box(b, n=n, shrink=shrink, rot_deg=rot_deg)
    pts = [occ.addPoint(float(x), float(y), float(z), float(h)) for (x, y) in poly]
    lines = [occ.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
    loop = occ.addCurveLoop(lines)
    surf = occ.addPlaneSurface([loop])
    return surf


# ----------------------------
# Gmsh build + tags
# ----------------------------
def build_gmsh_3d_device(
    base: Box2D,
    gate_boxes: List[Box2D],
    z_top: float,
    Lz: float,
    gate_mode: str,
    Lcap: float,
    gate_h: float,
    h: float,
    include_layers_regex: str,
    layer_to_ngon: Dict[str, int],
    default_N: int,
    poly_shrink: float,
    poly_rot_deg: float,
    model_name: str = "sige_polygons",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags, Dict[str, int], PhysTags, List[str]]:
    phys = PhysTags()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    kept_layer_names: List[str] = []

    if RANK == 0:
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        xmin, xmax = base.xmin, base.xmax
        ymin, ymax = base.ymin, base.ymax

        zmin_sige, zmax_sige = z_top - Lz, z_top
        sige = occ.addBox(xmin, ymin, zmin_sige, xmax - xmin, ymax - ymin, zmax_sige - zmin_sige)

        cap = None
        if gate_mode == "extruded":
            if Lcap <= 0:
                raise ValueError("For gate_mode=extruded, need Lcap>0")
            if gate_h <= 0 or gate_h > Lcap:
                raise ValueError("For gate_mode=extruded, need 0<gate_h<=Lcap")
            cap = occ.addBox(xmin, ymin, z_top, xmax - xmin, ymax - ymin, Lcap)

        occ.synchronize()

        keep_re = re.compile(include_layers_regex) if include_layers_regex else None
        kept_gate_boxes = [b for b in gate_boxes if (keep_re is None or keep_re.search(b.layer))]
        kept_layer_names = _unique_in_order([b.layer for b in kept_gate_boxes])

        layer_to_facet_tag: Dict[str, int] = {name: phys.GATE0 + i for i, name in enumerate(kept_layer_names)}

        # Precompute polygon footprints (CIF units) for tagging
        polys_by_layer: Dict[str, List[np.ndarray]] = {name: [] for name in kept_layer_names}
        for b in kept_gate_boxes:
            N = int(layer_to_ngon.get(b.layer, default_N))
            if N < 3:
                N = 3
            polys_by_layer[b.layer].append(polygon_footprint_for_box(b, n=N, shrink=poly_shrink, rot_deg=poly_rot_deg))

        # Create gates
        if gate_mode == "patch":
            gate_surfs: List[Tuple[int, int]] = []
            for b in kept_gate_boxes:
                N = int(layer_to_ngon.get(b.layer, default_N))
                if N < 3:
                    N = 3
                s = occ_add_regular_ngon_surface_from_box(
                    occ=occ, b=b, z=z_top, n=N, h=h, shrink=poly_shrink, rot_deg=poly_rot_deg
                )
                gate_surfs.append((2, s))
            occ.synchronize()

            if gate_surfs:
                occ.fragment([(3, sige)], gate_surfs)
                occ.synchronize()

        elif gate_mode == "extruded":
            assert cap is not None

            # Build extruded prisms in the cap volume and CUT them out, so gate surfaces become boundary facets.
            gate_prisms: List[Tuple[int, int]] = []
            for b in kept_gate_boxes:
                N = int(layer_to_ngon.get(b.layer, default_N))
                if N < 3:
                    N = 3
                s = occ_add_regular_ngon_surface_from_box(
                    occ=occ, b=b, z=z_top, n=N, h=h, shrink=poly_shrink, rot_deg=poly_rot_deg
                )
                # extrude surface in +z to make volume
                ext = occ.extrude([(2, s)], 0.0, 0.0, gate_h)
                # find the volume created by extrude
                for (dim, tag) in ext:
                    if dim == 3:
                        gate_prisms.append((3, tag))
            occ.synchronize()

            # Cut prisms out of the cap volume
            if gate_prisms:
                cut_out, _ = occ.cut([(3, cap)], gate_prisms, removeObject=True, removeTool=True)
                occ.synchronize()
                if len(cut_out) != 1 or cut_out[0][0] != 3:
                    raise RuntimeError(f"Unexpected cut result: {cut_out}")
                cap = cut_out[0][1]

            # Ensure conformity at the SiGe-cap interface
            occ.fragment([(3, sige), (3, cap)], [])
            occ.synchronize()

        else:
            raise ValueError(f"Unknown gate_mode={gate_mode}")

        # Tag volumes
        vols = gmsh.model.getEntities(dim=3)
        sige_vols: List[int] = []
        cap_vols: List[int] = []
        for (dim, vtag) in vols:
            cx, cy, cz = occ.getCenterOfMass(dim, vtag)
            if cz < z_top - 1e-9:
                sige_vols.append(vtag)
            else:
                cap_vols.append(vtag)

        if sige_vols:
            gmsh.model.addPhysicalGroup(3, sige_vols, phys.SIGE)
            gmsh.model.setPhysicalName(3, phys.SIGE, "SiGe")
        if cap_vols:
            gmsh.model.addPhysicalGroup(3, cap_vols, phys.CAP)
            gmsh.model.setPhysicalName(3, phys.CAP, "cap")

        # Tag facets
        surfs = gmsh.model.getEntities(dim=2)
        top_ground: List[int] = []
        bottom: List[int] = []
        sides: List[int] = []
        layer_surfs: Dict[str, List[int]] = {name: [] for name in kept_layer_names}

        z_bot = z_top - Lz
        z_top_outer = z_top if gate_mode == "patch" else (z_top + Lcap)

        # Choose a tolerance based on typical polygon size
        if kept_gate_boxes:
            Lref = max(max(b.W for b in kept_gate_boxes), max(b.H for b in kept_gate_boxes))
        else:
            Lref = 1.0
        tol_xy = 1e-6 * max(Lref, 1.0)
        tol_z = 1e-6 * max(Lref, 1.0)

        for (dim, stag) in surfs:
            cx, cy, cz = occ.getCenterOfMass(dim, stag)

            # Outer boundaries
            if abs(cz - z_bot) < tol_z:
                bottom.append(stag)
                continue
            if abs(cz - z_top_outer) < tol_z:
                # outer top only, not gates (gates are at z_top or inside the cap)
                top_ground.append(stag)
                continue

            # Gate surfaces:
            # patch mode: gates live on z=z_top
            # extruded mode: gates include surfaces with cz in [z_top, z_top+gate_h]
            if gate_mode == "patch":
                in_gate_z = abs(cz - z_top) < tol_z
            else:
                in_gate_z = (cz >= z_top - tol_z) and (cz <= (z_top + gate_h + tol_z))

            if in_gate_z:
                assigned = False
                for name in kept_layer_names:
                    for poly in polys_by_layer[name]:
                        if point_in_or_on_polygon_2d(cx, cy, poly, tol=tol_xy):
                            layer_surfs[name].append(stag)
                            assigned = True
                            break
                    if assigned:
                        break
                if assigned:
                    continue

            # Everything else is a side boundary (including cap sides)
            sides.append(stag)

        if top_ground:
            gmsh.model.addPhysicalGroup(2, top_ground, phys.TOP_GROUND)
            gmsh.model.setPhysicalName(2, phys.TOP_GROUND, "top_ground")
        if bottom:
            gmsh.model.addPhysicalGroup(2, bottom, phys.BOTTOM)
            gmsh.model.setPhysicalName(2, phys.BOTTOM, "bottom")
        if sides:
            gmsh.model.addPhysicalGroup(2, sides, phys.SIDES)
            gmsh.model.setPhysicalName(2, phys.SIDES, "sides")

        for name in kept_layer_names:
            ss = layer_surfs[name]
            if not ss:
                print(f"[WARN] Layer '{name}' got 0 gate facets.")
                continue
            tag = layer_to_facet_tag[name]
            gmsh.model.addPhysicalGroup(2, ss, tag)
            gmsh.model.setPhysicalName(2, tag, f"gate_{name}")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]

    if RANK != 0:
        kept_layer_names = None
    kept_layer_names = COMM.bcast(kept_layer_names, root=0)

    layer_to_facet_tag = {name: phys.GATE0 + i for i, name in enumerate(kept_layer_names)}

    gmsh.finalize()
    return msh, ct, ft, layer_to_facet_tag, phys, kept_layer_names


def solve_poisson_with_gate_dirichlet(
    msh: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
    layer_to_facet_tag: Dict[str, int],
    gate_voltages: Dict[str, float],
    default_gateV: float,
    deg: int,
    eps_r: float,
    rhs_rho: float,
    phys: PhysTags,
    bc_sides: str,
    bc_bottom: str,
) -> fem.Function:
    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps0 = 8.8541878128e-12
    eps = fem.Constant(msh, PETSc.ScalarType(eps0 * eps_r))
    rho = fem.Constant(msh, PETSc.ScalarType(rhs_rho))

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    for layer, tag in layer_to_facet_tag.items():
        Vg = float(gate_voltages.get(layer, default_gateV))
        facets = facet_tags.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    facets = facet_tags.find(phys.TOP_GROUND)
    if facets.size > 0:
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if bc_bottom == "dirichlet0":
        facets = facet_tags.find(phys.BOTTOM)
        if facets.size > 0:
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if bc_sides == "dirichlet0":
        facets = facet_tags.find(phys.SIDES)
        if facets.size > 0:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="CIF gate layout as Dirichlet polygon gates, with optional extrusion.")
    ap.add_argument("--cif", type=str, default="", help="Path to CIF-like file. If empty, uses built-in example.")
    ap.add_argument("--outdir", type=str, default="Results/cif_demo", help="Output directory.")
    ap.add_argument("--basename", type=str, default="sige_gate_layout", help="Basename for outputs.")

    ap.add_argument("--Lz", type=float, default=120.0, help="SiGe thickness (CIF units).")
    ap.add_argument("--z_top", type=float, default=0.0, help="Top plane z coordinate (CIF units).")
    ap.add_argument("--h", type=float, default=10.0, help="Mesh size in gmsh units (CIF units).")
    ap.add_argument("--deg", type=int, default=1, help="CG degree for phi solve (can be >1).")

    ap.add_argument("--scale", type=float, default=1.0,
                    help="After meshing, multiply mesh coordinates by this factor (e.g. 1e-9 to convert nm->m).")

    ap.add_argument("--include_layers_regex", type=str, default=".*",
                    help="Regex for which CIF layers become gates (default: all non-base).")

    ap.add_argument("--gateV", type=str, nargs="*", default=[],
                    help="Gate voltages as Layer=V (repeatable). Example: --gateV A1=0.1 B0=-0.2")
    ap.add_argument("--default_gateV", type=float, default=0.0,
                    help="Default gate voltage if a layer is not listed in --gateV.")

    ap.add_argument("--gateN", type=str, nargs="*", default=[],
                    help="Polygon sides per layer as Layer=N (repeatable). Example: --gateN A1=5 B0=3")
    ap.add_argument("--default_N", type=int, default=4, help="Default polygon sides for layers not in --gateN.")
    ap.add_argument("--poly_shrink", type=float, default=0.95,
                    help="Shrink factor <1 to keep polygon inside the CIF rectangle footprint.")
    ap.add_argument("--poly_rot_deg", type=float, default=0.0,
                    help="Rotate each regular polygon by this angle (degrees).")

    ap.add_argument("--gate_mode", choices=["patch", "extruded"], default="patch",
                    help="patch: polygon patches on z_top. extruded: cut extruded polygon prisms out of a cap volume.")
    ap.add_argument("--Lcap", type=float, default=120.0,
                    help="Cap thickness above z_top (CIF units). Used only for gate_mode=extruded.")
    ap.add_argument("--gate_h", type=float, default=60.0,
                    help="Extruded gate height (CIF units, <= Lcap). Used only for gate_mode=extruded.")

    ap.add_argument("--eps_r", type=float, default=11.7, help="Uniform relative permittivity (placeholder).")
    ap.add_argument("--rho", type=float, default=0.0, help="Uniform charge density (placeholder).")

    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")

    args = ap.parse_args()
    args.outdir = normalize_outdir(args.outdir)

    if args.cif:
        if RANK == 0:
            print(f"[INFO] Reading CIF spec from: {args.cif}")
        with open(args.cif, "r", encoding="utf-8") as f:
            cif_text = f.read()
    else:
        if RANK == 0:
            print("[INFO] Using built-in CIF example.")
        cif_text = DEFAULT_CIF_TEXT

    boxes = parse_cif_boxes(cif_text)
    if not boxes:
        raise RuntimeError("No boxes parsed from CIF text.")

    base_boxes = [b for b in boxes if b.layer.lower() == "base"]
    if len(base_boxes) != 1:
        raise RuntimeError(f"Expected exactly 1 base box, found {len(base_boxes)}.")
    base = base_boxes[0]

    gate_boxes = [b for b in boxes if b.layer.lower() != "base"]

    gate_voltages = parse_gate_voltage_pairs(args.gateV)
    layer_to_ngon = parse_gate_ngon_pairs(args.gateN)

    msh, ct, ft, layer_to_facet_tag, phys, kept_layer_names = build_gmsh_3d_device(
        base=base,
        gate_boxes=gate_boxes,
        z_top=args.z_top,
        Lz=args.Lz,
        gate_mode=args.gate_mode,
        Lcap=args.Lcap,
        gate_h=args.gate_h,
        h=args.h,
        include_layers_regex=args.include_layers_regex,
        layer_to_ngon=layer_to_ngon,
        default_N=args.default_N,
        poly_shrink=args.poly_shrink,
        poly_rot_deg=args.poly_rot_deg,
    )

    if args.scale != 1.0:
        msh.geometry.x[:] *= args.scale

    phi = solve_poisson_with_gate_dirichlet(
        msh=msh,
        facet_tags=ft,
        layer_to_facet_tag=layer_to_facet_tag,
        gate_voltages=gate_voltages,
        default_gateV=args.default_gateV,
        deg=args.deg,
        eps_r=args.eps_r,
        rhs_rho=args.rho,
        phys=phys,
        bc_sides=args.bc_sides,
        bc_bottom=args.bc_bottom,
    )

    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)

    if RANK == 0:
        print("\n=== Solve summary ===")
        print(f"phi min/max = [{gmin:.6e}, {gmax:.6e}]")

    os.makedirs(args.outdir, exist_ok=True)

    xdmf_solution = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    xdmf_facets = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")
    xdmf_cells = os.path.join(args.outdir, f"{args.basename}_cell_tags.xdmf")

    phi_write = function_for_xdmf(phi, msh)

    with io.XDMFFile(COMM, xdmf_solution, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_write)

    with io.XDMFFile(COMM, xdmf_facets, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ft, msh)

    with io.XDMFFile(COMM, xdmf_cells, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ct, msh)

    if RANK == 0:
        print("\nParaView:")
        print("  - Open *_phi.xdmf and view phi")
        print("  - Open *_facet_tags.xdmf and color by 'gmsh:physical' (gates should be vertical walls in extruded mode)")
        print("  - Open *_cell_tags.xdmf and color by 'gmsh:physical' (SiGe vs cap)")

if __name__ == "__main__":
    main()
