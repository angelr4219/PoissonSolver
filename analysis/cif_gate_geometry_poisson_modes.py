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
    # Cell tags
    SIGE: int = 1
    AIR: int = 2

    # Outer boundary facet tags
    TOP: int = 10
    BOTTOM: int = 11
    SIDES: int = 12

    # Gate boundary facet tags start here
    GATE0: int = 100


def build_gmsh_model(
    base: Box2D,
    gate_boxes: List[Box2D],
    include_layers_regex: str,
    geom_mode: str,
    z_top: float,
    Lz: float,
    h: float,
    Lair: float,
    gate_height: float,
    cavity_depth: float,
    model_name: str = "cif_device",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags, Dict[str, int], PhysTags, List[str]]:
    """
    geom_mode:
      patch    : top-surface 2D Dirichlet patches (no 3D features)
      extruded : 3D holes in an AIR slab above z_top; Dirichlet on hole walls
      cavity   : 3D holes cut into SiGe from z_top downward; Dirichlet on hole walls

    Important: DOLFINx DirichletBC is straightforward on boundary facets only.
    So for 3D gates, we model metal as a removed region (a void) and apply Dirichlet on its boundary.
    """

    phys = PhysTags()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    kept_layer_names: List[str] = []

    if RANK == 0:
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        keep_re = re.compile(include_layers_regex) if include_layers_regex else None
        kept_gate_boxes = [b for b in gate_boxes if (keep_re is None or keep_re.search(b.layer))]
        kept_layer_names = _unique_in_order([b.layer for b in kept_gate_boxes])

        layer_to_facet_tag: Dict[str, int] = {name: phys.GATE0 + i for i, name in enumerate(kept_layer_names)}

        xmin, xmax = base.xmin, base.xmax
        ymin, ymax = base.ymin, base.ymax
        zmin_sige, zmax_sige = z_top - Lz, z_top

        # Always create SiGe volume
        sige = occ.addBox(xmin, ymin, zmin_sige, xmax - xmin, ymax - ymin, zmax_sige - zmin_sige)

        air = None
        if geom_mode == "extruded":
            if Lair <= 0:
                raise ValueError("geom_mode=extruded requires --Lair > 0")
            if gate_height <= 0:
                raise ValueError("geom_mode=extruded requires --gate_height > 0")
            if gate_height > Lair + 1e-12:
                raise ValueError("Require gate_height <= Lair so gates fit inside AIR slab.")
            air = occ.addBox(xmin, ymin, z_top, xmax - xmin, ymax - ymin, Lair)

        occ.synchronize()

        tol = 1e-9

        def in_rect(b: Box2D, x: float, y: float) -> bool:
            return (x >= b.xmin - tol) and (x <= b.xmax + tol) and (y >= b.ymin - tol) and (y <= b.ymax + tol)

        # Build gate geometry
        if geom_mode == "patch":
            gate_surfs: List[Tuple[int, int]] = []
            for b in kept_gate_boxes:
                dx = b.xmax - b.xmin
                dy = b.ymax - b.ymin
                if dx <= 0 or dy <= 0:
                    continue
                s = occ.addRectangle(b.xmin, b.ymin, z_top, dx, dy)
                gate_surfs.append((2, s))
            occ.synchronize()

            if gate_surfs:
                occ.fragment([(3, sige)], gate_surfs)
                occ.synchronize()

        elif geom_mode == "cavity":
            if cavity_depth <= 0:
                raise ValueError("geom_mode=cavity requires --cavity_depth > 0")

            tools = []
            for b in kept_gate_boxes:
                dx = b.xmax - b.xmin
                dy = b.ymax - b.ymin
                if dx <= 0 or dy <= 0:
                    continue
                # cut downward into SiGe
                t = occ.addBox(b.xmin, b.ymin, z_top - cavity_depth, dx, dy, cavity_depth)
                tools.append((3, t))
            occ.synchronize()

            if tools:
                out = occ.cut([(3, sige)], tools, removeObject=True, removeTool=True)
                occ.synchronize()
                # After cut, SiGe may be split, we will tag by z COM later

        elif geom_mode == "extruded":
            # Cut holes out of AIR slab where gates live
            tools = []
            for b in kept_gate_boxes:
                dx = b.xmax - b.xmin
                dy = b.ymax - b.ymin
                if dx <= 0 or dy <= 0:
                    continue
                t = occ.addBox(b.xmin, b.ymin, z_top, dx, dy, gate_height)
                tools.append((3, t))
            occ.synchronize()

            if tools:
                out = occ.cut([(3, air)], tools, removeObject=True, removeTool=True)
                occ.synchronize()

            # Fragment air and sige so the interface is conformal
            occ.fragment([(3, sige), (3, air)], [])
            occ.synchronize()

        else:
            raise ValueError(f"Unknown geom_mode: {geom_mode}")

        # Tag volumes: any volume with COM z < z_top is SiGe, else AIR (if present)
        vols = gmsh.model.getEntities(dim=3)
        sige_vols: List[int] = []
        air_vols: List[int] = []
        for (d, t) in vols:
            cx, cy, cz = occ.getCenterOfMass(d, t)
            if cz < z_top - 1e-6:
                sige_vols.append(t)
            else:
                air_vols.append(t)

        if sige_vols:
            gmsh.model.addPhysicalGroup(3, sige_vols, phys.SIGE)
            gmsh.model.setPhysicalName(3, phys.SIGE, "SiGe")

        if geom_mode == "extruded" and air_vols:
            gmsh.model.addPhysicalGroup(3, air_vols, phys.AIR)
            gmsh.model.setPhysicalName(3, phys.AIR, "Air")

        # Tag boundary facets: TOP, BOTTOM, SIDES, and gate_* facets.
        surfs = gmsh.model.getEntities(dim=2)

        top_surfs: List[int] = []
        bot_surfs: List[int] = []
        side_surfs: List[int] = []

        # Gate facets per layer
        layer_surfs: Dict[str, List[int]] = {name: [] for name in kept_layer_names}

        z_bottom = z_top - Lz
        z_top_outer = z_top + (Lair if geom_mode == "extruded" else 0.0)

        for (d, s) in surfs:
            cx, cy, cz = occ.getCenterOfMass(d, s)

            # Outer box boundary checks
            if abs(cz - z_top_outer) < 1e-6:
                top_surfs.append(s)
                continue
            if abs(cz - z_bottom) < 1e-6:
                bot_surfs.append(s)
                continue

            # Side boundary: near x min/max or y min/max
            if (abs(cx - xmin) < 1e-6) or (abs(cx - xmax) < 1e-6) or (abs(cy - ymin) < 1e-6) or (abs(cy - ymax) < 1e-6):
                side_surfs.append(s)
                continue

            # Gate boundary facets:
            # For patch mode: surfaces on z=z_top within rectangles
            # For cavity mode: surfaces inside rectangles with z between [z_top-cavity_depth, z_top]
            # For extruded mode: surfaces inside rectangles with z between [z_top, z_top+gate_height]
            for name in kept_layer_names:
                # A layer can have multiple boxes in CIF, so we check all boxes with that layer
                hit = False
                for b in kept_gate_boxes:
                    if b.layer != name:
                        continue
                    if not in_rect(b, cx, cy):
                        continue
                    if geom_mode == "patch":
                        if abs(cz - z_top) < 1e-6:
                            layer_surfs[name].append(s)
                            hit = True
                            break
                    elif geom_mode == "cavity":
                        if (cz >= (z_top - cavity_depth - 1e-6)) and (cz <= (z_top + 1e-6)):
                            layer_surfs[name].append(s)
                            hit = True
                            break
                    elif geom_mode == "extruded":
                        if (cz >= (z_top - 1e-6)) and (cz <= (z_top + gate_height + 1e-6)):
                            layer_surfs[name].append(s)
                            hit = True
                            break
                if hit:
                    break

        # Clean duplicates
        top_surfs = sorted(set(top_surfs))
        bot_surfs = sorted(set(bot_surfs))
        side_surfs = sorted(set(side_surfs))
        for name in kept_layer_names:
            layer_surfs[name] = sorted(set(layer_surfs[name]))

        if top_surfs:
            gmsh.model.addPhysicalGroup(2, top_surfs, phys.TOP)
            gmsh.model.setPhysicalName(2, phys.TOP, "top_outer")
        if bot_surfs:
            gmsh.model.addPhysicalGroup(2, bot_surfs, phys.BOTTOM)
            gmsh.model.setPhysicalName(2, phys.BOTTOM, "bottom")
        if side_surfs:
            gmsh.model.addPhysicalGroup(2, side_surfs, phys.SIDES)
            gmsh.model.setPhysicalName(2, phys.SIDES, "sides")

        for name in kept_layer_names:
            ss = layer_surfs[name]
            if not ss:
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


def make_eps_piecewise(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    phys: PhysTags,
    eps_r_sige: float,
    eps_r_air: float,
) -> fem.Function:
    eps0 = 8.8541878128e-12
    Q = fem.functionspace(msh, ("DG", 0))
    eps = fem.Function(Q, name="eps")

    # Default SiGe everywhere, then override AIR-tagged cells if present
    eps.x.array[:] = eps0 * eps_r_sige

    # MeshTags stores local indices for tagged entities
    if ct is not None and ct.indices is not None and ct.values is not None:
        # Only override where tag == AIR
        air_mask = (ct.values == phys.AIR)
        air_cells = ct.indices[air_mask]
        if air_cells.size > 0:
            eps.x.array[air_cells] = eps0 * eps_r_air

    return eps


def solve_poisson(
    msh: dmesh.Mesh,
    ft: dmesh.MeshTags,
    layer_to_facet_tag: Dict[str, int],
    gate_voltages: Dict[str, float],
    default_gateV: float,
    deg: int,
    eps: fem.Function,
    rho_val: float,
    phys: PhysTags,
    bc_sides: str,
    bc_bottom: str,
    bc_top: str,
) -> fem.Function:
    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    rho = fem.Constant(msh, PETSc.ScalarType(rho_val))

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    # Gate Dirichlet on boundary facets tagged per layer
    for layer, tag in layer_to_facet_tag.items():
        Vg = float(gate_voltages.get(layer, default_gateV))
        facets = ft.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    # Outer boundaries
    if bc_top == "dirichlet0":
        facets = ft.find(phys.TOP)
        if facets.size > 0:
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if bc_bottom == "dirichlet0":
        facets = ft.find(phys.BOTTOM)
        if facets.size > 0:
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if bc_sides == "dirichlet0":
        facets = ft.find(phys.SIDES)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="CIF gate layout with patch, extruded, or cavity 3D gate BC surfaces.")
    ap.add_argument("--cif", type=str, default="", help="Path to CIF-like file. If empty, uses built-in example.")
    ap.add_argument("--outdir", type=str, default="Results/cif_modes", help="Output directory.")
    ap.add_argument("--basename", type=str, default="cif_device", help="Basename for outputs.")

    ap.add_argument("--geom_mode", choices=["patch", "extruded", "cavity"], default="patch",
                    help="patch: 2D top patches. extruded: holes in air slab. cavity: holes in SiGe.")

    ap.add_argument("--Lz", type=float, default=120.0, help="SiGe thickness (CIF units).")
    ap.add_argument("--z_top", type=float, default=0.0, help="Top plane z coordinate (CIF units).")
    ap.add_argument("--h", type=float, default=10.0, help="Mesh size (CIF units).")
    ap.add_argument("--deg", type=int, default=1, help="CG degree for phi solve.")

    ap.add_argument("--Lair", type=float, default=80.0, help="Air thickness above z_top (only used for extruded).")
    ap.add_argument("--gate_height", type=float, default=40.0, help="Extruded gate height in air (extruded mode).")
    ap.add_argument("--cavity_depth", type=float, default=40.0, help="Cavity depth into SiGe (cavity mode).")

    ap.add_argument("--include_layers_regex", type=str, default=".*", help="Regex selecting which layers become gates.")
    ap.add_argument("--gateV", type=str, nargs="*", default=[], help="Gate voltages Layer=V (repeatable).")
    ap.add_argument("--default_gateV", type=float, default=0.0, help="Default gate voltage.")

    ap.add_argument("--eps_r_sige", type=float, default=11.7, help="Relative permittivity in SiGe.")
    ap.add_argument("--eps_r_air", type=float, default=1.0, help="Relative permittivity in air (extruded mode).")
    ap.add_argument("--rho", type=float, default=0.0, help="Uniform charge density (all regions).")

    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_top", choices=["dirichlet0", "none"], default="dirichlet0")

    args = ap.parse_args()
    args.outdir = normalize_outdir(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

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

    if RANK == 0:
        print("\n=== Build params ===")
        print(f"geom_mode={args.geom_mode}")
        print(f"SiGe z in [{args.z_top-args.Lz}, {args.z_top}] (CIF units)")
        if args.geom_mode == "extruded":
            print(f"Air  z in [{args.z_top}, {args.z_top+args.Lair}] (CIF units)")
            print(f"Gate height in air: {args.gate_height}")
        if args.geom_mode == "cavity":
            print(f"Cavity depth into SiGe: {args.cavity_depth}")
        print(f"Base footprint: x in [{base.xmin}, {base.xmax}], y in [{base.ymin}, {base.ymax}]")
        print(f"h={args.h}, deg={args.deg}")
        print(f"eps_r_sige={args.eps_r_sige}, eps_r_air={args.eps_r_air}, rho={args.rho}")
        print(f"bc top={args.bc_top}, sides={args.bc_sides}, bottom={args.bc_bottom}")
        print(f"OUTDIR={args.outdir}")

    msh, ct, ft, layer_to_facet_tag, phys, kept_layer_names = build_gmsh_model(
        base=base,
        gate_boxes=gate_boxes,
        include_layers_regex=args.include_layers_regex,
        geom_mode=args.geom_mode,
        z_top=args.z_top,
        Lz=args.Lz,
        h=args.h,
        Lair=args.Lair,
        gate_height=args.gate_height,
        cavity_depth=args.cavity_depth,
        model_name="cif_device_modes",
    )

    eps = make_eps_piecewise(
        msh=msh,
        ct=ct,
        phys=phys,
        eps_r_sige=args.eps_r_sige,
        eps_r_air=args.eps_r_air,
    )

    phi = solve_poisson(
        msh=msh,
        ft=ft,
        layer_to_facet_tag=layer_to_facet_tag,
        gate_voltages=gate_voltages,
        default_gateV=args.default_gateV,
        deg=args.deg,
        eps=eps,
        rho_val=args.rho,
        phys=phys,
        bc_sides=args.bc_sides,
        bc_bottom=args.bc_bottom,
        bc_top=args.bc_top,
    )

    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)

    if RANK == 0:
        print("\n=== Solve summary ===")
        print(f"phi min/max = [{gmin:.6e}, {gmax:.6e}]")
        print("\nDevice box (from base CIF):")
        print(f"x in [{base.xmin}, {base.xmax}], y in [{base.ymin}, {base.ymax}], z in [{args.z_top-args.Lz}, {args.z_top}]")

    mesh_deg = _degree_compat(msh.geometry.cmap.degree)
    sol_deg = _degree_compat(phi.function_space.ufl_element().degree)
    if sol_deg != mesh_deg:
        # write interpolated phi for XDMF compatibility
        Vout = fem.functionspace(msh, ("CG", mesh_deg))
        phi_out = fem.Function(Vout, name="phi")
        try:
            phi_out.interpolate(phi)
        except Exception:
            expr = fem.Expression(phi, Vout.element.interpolation_points())
            phi_out.interpolate(expr)
        phi_write = phi_out
    else:
        phi_write = phi

    xdmf_phi = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    xdmf_ct = os.path.join(args.outdir, f"{args.basename}_cell_tags.xdmf")
    xdmf_ft = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")

    with io.XDMFFile(COMM, xdmf_phi, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_write)

    with io.XDMFFile(COMM, xdmf_ct, "w") as xdmf:
        xdmf.write_mesh(msh)
        try:
            xdmf.write_meshtags(ct)
        except TypeError:
            xdmf.write_meshtags(ct, msh.geometry)

    with io.XDMFFile(COMM, xdmf_ft, "w") as xdmf:
        xdmf.write_mesh(msh)
        try:
            xdmf.write_meshtags(ft)
        except TypeError:
            xdmf.write_meshtags(ft, msh.geometry)

    if RANK == 0:
        print("\n=== Wrote files ===")
        print(f"  phi:       {xdmf_phi}")
        print(f"  cell tags: {xdmf_ct}")
        print(f"  facet tags:{xdmf_ft}")
        print("\nParaView tips:")
        print("  - Open *_facet_tags.xdmf and color by 'gmsh:physical' to see gate boundary surfaces")
        print("  - In extruded mode, you will see air slab with gate-shaped voids (hole walls are Dirichlet)")
        print("  - In cavity mode, you will see holes cut into SiGe (hole walls are Dirichlet)")
        print("  - Open *_cell_tags.xdmf and color by 'gmsh:physical' to see SiGe vs Air")


if __name__ == "__main__":
    main()
