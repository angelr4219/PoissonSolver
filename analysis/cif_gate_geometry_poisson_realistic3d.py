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


def write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


@dataclass(frozen=True)
class PhysTags:
    OX: int = 11
    SIGE: int = 12

    TOP_OUTER: int = 21
    BOTTOM: int = 22
    SIDES: int = 23

    # IMPORTANT: do not apply BCs using this tag.
    # It only exists to guarantee dolfinx returns facet tags (ft) instead of None.
    ALL_BOUNDARY: int = 24

    GATE0: int = 100


def build_gmsh_3d_stack_with_gate_cavities_bbox_tagged(
    base: Box2D,
    gate_boxes: List[Box2D],
    include_layers_regex: str,
    z_top: float,
    Lz: float,
    tox: float,
    h: float,
    model_name: str = "sige_ox_gate_cavities_bbox",
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

        zmin_dev, zmax_dev = z_top - Lz, z_top
        zmin_ox, zmax_ox = z_top, z_top + tox

        sige = occ.addBox(xmin, ymin, zmin_dev, xmax - xmin, ymax - ymin, zmax_dev - zmin_dev)
        ox = occ.addBox(xmin, ymin, zmin_ox, xmax - xmin, ymax - ymin, zmax_ox - zmin_ox)
        occ.synchronize()

        keep_re = re.compile(include_layers_regex) if include_layers_regex else None
        kept_gate_boxes = [b for b in gate_boxes if (keep_re is None or keep_re.search(b.layer))]
        kept_layer_names = _unique_in_order([b.layer for b in kept_gate_boxes])
        layer_to_facet_tag: Dict[str, int] = {name: phys.GATE0 + i for i, name in enumerate(kept_layer_names)}

        # Gate volumes for cutting oxide
        gate_vols: List[Tuple[int, int]] = []
        for b in kept_gate_boxes:
            dx = b.xmax - b.xmin
            dy = b.ymax - b.ymin
            if dx <= 0 or dy <= 0:
                print(f"[WARN] Skipping degenerate box layer={b.layer} dx={dx} dy={dy}")
                continue
            gv = occ.addBox(b.xmin, b.ymin, zmin_ox, dx, dy, zmax_ox - zmin_ox)
            gate_vols.append((3, gv))
        occ.synchronize()

        if gate_vols:
            occ.cut([(3, ox)], gate_vols, removeObject=True, removeTool=True)
            occ.synchronize()

        # Fragment so SiGe and Ox share interface mesh
        occ.fragment([(3, sige)], gmsh.model.getEntities(3))
        occ.synchronize()

        # Tag volumes by COM z
        vols = gmsh.model.getEntities(dim=3)
        sige_vols: List[int] = []
        ox_vols: List[int] = []
        for (_, t) in vols:
            cx, cy, cz = occ.getCenterOfMass(3, t)
            if cz < z_top - 0.25 * Lz:
                sige_vols.append(t)
            else:
                ox_vols.append(t)

        if not sige_vols:
            raise RuntimeError("No SiGe volumes found.")
        if not ox_vols:
            raise RuntimeError("No oxide volumes found.")

        gmsh.model.addPhysicalGroup(3, sige_vols, phys.SIGE)
        gmsh.model.setPhysicalName(3, phys.SIGE, "SiGe")
        gmsh.model.addPhysicalGroup(3, ox_vols, phys.OX)
        gmsh.model.setPhysicalName(3, phys.OX, "Ox")

        # Collect boundary surfaces (surfaces with exactly 1 adjacent volume)
        all_surfs = [t for (_, t) in gmsh.model.getEntities(dim=2)]
        boundary_surfs: List[int] = []
        for s in all_surfs:
            up_dims, up_tags = gmsh.model.getAdjacencies(2, s)
            if len(up_tags) == 1:
                boundary_surfs.append(s)

        if not boundary_surfs:
            # fallback: still create something
            boundary_surfs = all_surfs[:]

        boundary_surfs = sorted(set(boundary_surfs))

        # GUARANTEE: at least one 2D physical group exists (so ft is not None in dolfinx)
        gmsh.model.addPhysicalGroup(2, boundary_surfs, phys.ALL_BOUNDARY)
        gmsh.model.setPhysicalName(2, phys.ALL_BOUNDARY, "all_boundary")

        # Gate cavity surfaces by bbox
        delta = 1e-4  # padding in CIF units
        layer_surfs: Dict[str, List[int]] = {name: [] for name in kept_layer_names}

        bset = set(boundary_surfs)
        for name in kept_layer_names:
            boxes = [b for b in kept_gate_boxes if b.layer == name]
            picks: List[int] = []
            for b in boxes:
                cand = gmsh.model.getEntitiesInBoundingBox(
                    b.xmin - delta, b.ymin - delta, zmin_ox - delta,
                    b.xmax + delta, b.ymax + delta, zmax_ox + delta,
                    dim=2
                )
                cand_tags = [t for (_, t) in cand]
                for s in cand_tags:
                    if s in bset:
                        picks.append(s)
            picks = sorted(set(picks))
            layer_surfs[name] = picks

        # Outer boundaries
        top_outer: List[int] = []
        bottom: List[int] = []
        sides: List[int] = []

        def in_any_gate_xy(x: float, y: float) -> bool:
            for b in kept_gate_boxes:
                if (x >= b.xmin - 1e-9) and (x <= b.xmax + 1e-9) and (y >= b.ymin - 1e-9) and (y <= b.ymax + 1e-9):
                    return True
            return False

        zmin = z_top - Lz
        zmax = z_top + tox

        gate_union = set()
        for name in kept_layer_names:
            gate_union |= set(layer_surfs[name])

        for s in boundary_surfs:
            cx, cy, cz = occ.getCenterOfMass(2, s)
            if abs(cz - zmin) < 1e-6:
                bottom.append(s)
            elif abs(cz - zmax) < 1e-6:
                if not in_any_gate_xy(cx, cy):
                    top_outer.append(s)
            else:
                if s not in gate_union:
                    sides.append(s)

        if top_outer:
            gmsh.model.addPhysicalGroup(2, top_outer, phys.TOP_OUTER)
            gmsh.model.setPhysicalName(2, phys.TOP_OUTER, "top_outer")
        if bottom:
            gmsh.model.addPhysicalGroup(2, bottom, phys.BOTTOM)
            gmsh.model.setPhysicalName(2, phys.BOTTOM, "bottom")
        if sides:
            gmsh.model.addPhysicalGroup(2, sides, phys.SIDES)
            gmsh.model.setPhysicalName(2, phys.SIDES, "sides")

        for name in kept_layer_names:
            ss = layer_surfs[name]
            if not ss:
                print(f"[WARN] Layer '{name}' got 0 cavity boundary surfaces (bbox).")
                continue
            tag = layer_to_facet_tag[name]
            gmsh.model.addPhysicalGroup(2, ss, tag)
            gmsh.model.setPhysicalName(2, tag, f"gate_{name}")
            print(f"[INFO] gate_{name}: tagged {len(ss)} surfaces")

        # Debug: show physical groups counts
        print("[INFO] Physical groups dim=2:", gmsh.model.getPhysicalGroups(dim=2))
        print("[INFO] Physical groups dim=3:", gmsh.model.getPhysicalGroups(dim=3))

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


def build_eps_rho_DG0(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    phys: PhysTags,
    eps_r_sige: float,
    eps_r_ox: float,
    rho_sige: float,
) -> Tuple[fem.Function, fem.Function]:
    V0 = fem.functionspace(msh, ("DG", 0))
    eps0 = 8.8541878128e-12

    eps = fem.Function(V0, name="eps")
    rho = fem.Function(V0, name="rho")

    eps_vals = np.full(eps.x.array.shape, eps0 * eps_r_ox, dtype=eps.x.array.dtype)
    rho_vals = np.zeros_like(rho.x.array, dtype=rho.x.array.dtype)

    cell_tags = ct.values
    sige_mask = (cell_tags == phys.SIGE)
    eps_vals[sige_mask] = eps0 * eps_r_sige
    rho_vals[sige_mask] = rho_sige

    eps.x.array[:] = eps_vals
    rho.x.array[:] = rho_vals
    return eps, rho


def solve_poisson(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    ft: dmesh.MeshTags,
    layer_to_facet_tag: Dict[str, int],
    gate_voltages: Dict[str, float],
    default_gateV: float,
    deg: int,
    phys: PhysTags,
    eps_r_sige: float,
    eps_r_ox: float,
    rho_sige: float,
    bc_sides: str,
    bc_bottom: str,
    bc_top_outer: str,
) -> fem.Function:
    if ft is None:
        raise RuntimeError("Facet tags ft is None even after forcing ALL_BOUNDARY physical group.")

    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps, rho = build_eps_rho_DG0(msh, ct, phys, eps_r_sige, eps_r_ox, rho_sige)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * dx
    L = rho * v * dx

    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    # Gates first
    for layer, tag in layer_to_facet_tag.items():
        Vg = float(gate_voltages.get(layer, default_gateV))
        facets = ft.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    if bc_top_outer == "dirichlet0":
        facets = ft.find(phys.TOP_OUTER)
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
    ap = argparse.ArgumentParser(description="Realistic 3D: SiGe + oxide, gate cavities tagged, with guaranteed facet tags.")
    ap.add_argument("--cif", type=str, default="")
    ap.add_argument("--outdir", type=str, default="Results/cif_realistic3d")
    ap.add_argument("--basename", type=str, default="sige_gate_layout")

    ap.add_argument("--Lz", type=float, default=120.0)
    ap.add_argument("--tox", type=float, default=20.0)
    ap.add_argument("--z_top", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=10.0)
    ap.add_argument("--deg", type=int, default=1)

    ap.add_argument("--scale", type=float, default=1.0)

    ap.add_argument("--include_layers_regex", type=str, default=".*")
    ap.add_argument("--gateV", type=str, nargs="*", default=[])
    ap.add_argument("--default_gateV", type=float, default=0.0)

    ap.add_argument("--eps_r_sige", type=float, default=11.7)
    ap.add_argument("--eps_r_ox", type=float, default=3.9)
    ap.add_argument("--rho_sige", type=float, default=0.0)

    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_top_outer", choices=["dirichlet0", "none"], default="dirichlet0")

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
    base_boxes = [b for b in boxes if b.layer.lower() == "base"]
    if len(base_boxes) != 1:
        raise RuntimeError(f"Expected exactly 1 base box, found {len(base_boxes)}")
    base = base_boxes[0]
    gate_boxes = [b for b in boxes if b.layer.lower() != "base"]

    gate_voltages = parse_gate_voltage_pairs(args.gateV)

    msh, ct, ft, layer_to_facet_tag, phys, kept_layer_names = build_gmsh_3d_stack_with_gate_cavities_bbox_tagged(
        base=base,
        gate_boxes=gate_boxes,
        include_layers_regex=args.include_layers_regex,
        z_top=args.z_top,
        Lz=args.Lz,
        tox=args.tox,
        h=args.h,
    )

    if args.scale != 1.0:
        msh.geometry.x[:] *= args.scale

    phi = solve_poisson(
        msh=msh,
        ct=ct,
        ft=ft,
        layer_to_facet_tag=layer_to_facet_tag,
        gate_voltages=gate_voltages,
        default_gateV=args.default_gateV,
        deg=args.deg,
        phys=phys,
        eps_r_sige=args.eps_r_sige,
        eps_r_ox=args.eps_r_ox,
        rho_sige=args.rho_sige,
        bc_sides=args.bc_sides,
        bc_bottom=args.bc_bottom,
        bc_top_outer=args.bc_top_outer,
    )

    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)

    if RANK == 0:
        print(f"\nphi min/max = [{gmin:.6e}, {gmax:.6e}]")

    os.makedirs(args.outdir, exist_ok=True)
    xdmf_solution = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    xdmf_cells = os.path.join(args.outdir, f"{args.basename}_cell_tags.xdmf")
    xdmf_facets = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")

    phi_write = function_for_xdmf(phi, msh)

    with io.XDMFFile(COMM, xdmf_solution, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_write)

    with io.XDMFFile(COMM, xdmf_cells, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ct, msh)

    with io.XDMFFile(COMM, xdmf_facets, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ft, msh)

    if RANK == 0:
        print("Wrote:")
        print("  ", xdmf_solution)
        print("  ", xdmf_cells)
        print("  ", xdmf_facets)
        print("ParaView: color facet_tags by 'gmsh:physical'. Gate tags start at", phys.GATE0)
        print("Note: 'all_boundary' tag is", phys.ALL_BOUNDARY, "and is not used for BCs.")

if __name__ == "__main__":
    main()
