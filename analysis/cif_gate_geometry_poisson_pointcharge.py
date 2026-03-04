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
        out[k.strip()] = float(v.strip())
    return out


def parse_qpoints(entries: List[str]) -> List[Tuple[float, float, float, float]]:
    """
    Each entry: "x,y,z,q" in CIF units for x,y,z and q in chosen units (e or C).
    """
    out = []
    for s in entries:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"--qpoint must be x,y,z,q got '{s}'")
        x, y, z, q = map(float, parts)
        out.append((x, y, z, q))
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
    Sige: int = 1
    TOP_GROUND: int = 10
    BOTTOM: int = 11
    SIDES: int = 12
    GATE0: int = 100


def build_gmsh_3d_device_with_top_patch_gates(
    base: Box2D,
    gate_boxes: List[Box2D],
    z_top: float,
    Lz: float,
    h: float,
    include_layers_regex: str,
    model_name: str = "sige_top_patch_gates",
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
        dev = occ.addBox(xmin, ymin, zmin_dev, xmax - xmin, ymax - ymin, zmax_dev - zmin_dev)
        occ.synchronize()

        keep_re = re.compile(include_layers_regex) if include_layers_regex else None
        kept_gate_boxes = [b for b in gate_boxes if (keep_re is None or keep_re.search(b.layer))]
        kept_layer_names = _unique_in_order([b.layer for b in kept_gate_boxes])

        gate_surfs: List[Tuple[int, int]] = []
        for b in kept_gate_boxes:
            dx = b.xmax - b.xmin
            dy = b.ymax - b.ymin
            if dx <= 0 or dy <= 0:
                print(f"[WARN] Skipping degenerate box layer={b.layer} dx={dx} dy={dy}")
                continue
            s = occ.addRectangle(b.xmin, b.ymin, z_top, dx, dy)
            gate_surfs.append((2, s))
        occ.synchronize()

        if gate_surfs:
            occ.fragment([(3, dev)], gate_surfs)
            occ.synchronize()

        vols = gmsh.model.getEntities(dim=3)
        vol_tags = [t for (d, t) in vols]
        gmsh.model.addPhysicalGroup(3, vol_tags, phys.Sige)
        gmsh.model.setPhysicalName(3, phys.Sige, "SiGe")

        layer_to_facet_tag: Dict[str, int] = {name: phys.GATE0 + i for i, name in enumerate(kept_layer_names)}

        surfs = gmsh.model.getEntities(dim=2)
        tol = 1e-9

        def rect_contains(b: Box2D, x: float, y: float) -> bool:
            return (x >= b.xmin - tol) and (x <= b.xmax + tol) and (y >= b.ymin - tol) and (y <= b.ymax + tol)

        layer_surfs: Dict[str, List[int]] = {name: [] for name in kept_layer_names}
        top_ground: List[int] = []
        bottom: List[int] = []
        sides: List[int] = []

        zmin = z_top - Lz
        boxes_by_layer: Dict[str, List[Box2D]] = {name: [] for name in kept_layer_names}
        for b in kept_gate_boxes:
            boxes_by_layer[b.layer].append(b)

        for (dim, s) in surfs:
            cx, cy, cz = occ.getCenterOfMass(dim, s)
            if abs(cz - z_top) < 1e-6:
                assigned = False
                for name in kept_layer_names:
                    for b in boxes_by_layer[name]:
                        if rect_contains(b, cx, cy):
                            layer_surfs[name].append(s)
                            assigned = True
                            break
                    if assigned:
                        break
                if not assigned:
                    top_ground.append(s)
            elif abs(cz - zmin) < 1e-6:
                bottom.append(s)
            else:
                sides.append(s)

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
                print(f"[WARN] Layer '{name}' got 0 top facets.")
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
    phys = PhysTags()
    layer_to_facet_tag = {name: phys.GATE0 + i for i, name in enumerate(kept_layer_names)}

    gmsh.finalize()
    return msh, ct, ft, layer_to_facet_tag, phys, kept_layer_names


def gaussian_point_charge_rho(
    msh: dmesh.Mesh,
    qpoints_scaled: List[Tuple[float, float, float, float]],
    sigma_scaled: float,
    q_units: str,
) -> ufl.core.expr.Expr:
    """
    Return UFL expression rho(x) [C/m^3] as sum of 3D Gaussians with total charge q.
    sigma_scaled is in mesh coordinate units (after scaling).
    """
    x = ufl.SpatialCoordinate(msh)
    sig = fem.Constant(msh, PETSc.ScalarType(float(sigma_scaled)))
    sig2 = sig * sig

    # Convert q to Coulombs if needed
    echarge = 1.602176634e-19
    qscale = echarge if q_units == "e" else 1.0

    # Normalization so ∫ rho dV = q
    norm = 1.0 / ((2.0 * np.pi) ** 1.5)

    rho_expr = 0
    for (x0, y0, z0, q) in qpoints_scaled:
        qC = float(q) * qscale
        dx = x[0] - x0
        dy = x[1] - y0
        dz = x[2] - z0
        r2 = dx * dx + dy * dy + dz * dz
        rho_expr += PETSc.ScalarType(qC) * PETSc.ScalarType(norm) * ufl.exp(-0.5 * r2 / sig2) / (sig**3)

    return rho_expr


def solve_poisson_with_gate_dirichlet_and_rho(
    msh: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
    layer_to_facet_tag: Dict[str, int],
    gate_voltages: Dict[str, float],
    default_gateV: float,
    deg: int,
    eps_r: float,
    rho_expr: ufl.core.expr.Expr,
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

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho_expr * v * ufl.dx

    fdim = msh.topology.dim - 1
    bcs: List[fem.DirichletBC] = []

    # Gate patches
    for layer, tag in layer_to_facet_tag.items():
        Vg = float(gate_voltages.get(layer, default_gateV))
        facets = facet_tags.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    # Ungated top surface grounded
    facets = facet_tags.find(phys.TOP_GROUND)
    if facets.size > 0:
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    # Bottom / sides
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
    ap = argparse.ArgumentParser(description="CIF top-surface gates + optional mollified point charges.")
    ap.add_argument("--cif", type=str, default="", help="Path to CIF-like file. If empty, uses built-in example.")
    ap.add_argument("--outdir", type=str, default="Results/cif_pointcharge", help="Output directory.")
    ap.add_argument("--basename", type=str, default="sige_gate_layout", help="Basename for outputs.")

    ap.add_argument("--Lz", type=float, default=120.0, help="SiGe thickness (CIF units).")
    ap.add_argument("--z_top", type=float, default=0.0, help="Top plane z coordinate (CIF units).")
    ap.add_argument("--h", type=float, default=15.0, help="Mesh size in gmsh units (CIF units).")
    ap.add_argument("--deg", type=int, default=1, help="CG degree for phi solve (can be >1).")

    ap.add_argument("--scale", type=float, default=1e-9, help="Multiply mesh coords by this (nm->m use 1e-9).")

    ap.add_argument("--include_layers_regex", type=str, default=".*",
                    help="Regex for which CIF layers become gates.")
    ap.add_argument("--gateV", type=str, nargs="*", default=[],
                    help="Gate voltages as Layer=V (repeatable).")
    ap.add_argument("--default_gateV", type=float, default=0.0)

    ap.add_argument("--eps_r", type=float, default=11.7, help="Uniform relative permittivity.")
    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")

    # Point charge inputs
    ap.add_argument("--qpoint", type=str, action="append", default=[],
                    help="Point charge as x,y,z,q (CIF units). Repeatable.")
    ap.add_argument("--q_units", choices=["e", "C"], default="e",
                    help="Charge units for q in --qpoint. 'e' means multiples of electron charge.")
    ap.add_argument("--sigma", type=float, default=5.0,
                    help="Gaussian width sigma (CIF units). Should be resolved by mesh (sigma >= ~2h).")

    args = ap.parse_args()
    args.outdir = normalize_outdir(args.outdir)

    if args.cif:
        with open(args.cif, "r", encoding="utf-8") as f:
            cif_text = f.read()
    else:
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
        print("\n=== CIF + gates + point charges ===")
        print(f"base: 650 x 510 (CIF units), center=(0,35)")
        print(f"z in [{args.z_top-args.Lz}, {args.z_top}] (CIF units), scale={args.scale}")
        print(f"h={args.h}, deg={args.deg}, eps_r={args.eps_r}")
        print(f"q_units={args.q_units}, sigma={args.sigma} (CIF units)")
        if args.qpoint:
            print("qpoints (CIF units):")
            for s in args.qpoint:
                print(f"  {s}")

    msh, ct, ft, layer_to_facet_tag, phys, kept_layer_names = build_gmsh_3d_device_with_top_patch_gates(
        base=base,
        gate_boxes=gate_boxes,
        z_top=args.z_top,
        Lz=args.Lz,
        h=args.h,
        include_layers_regex=args.include_layers_regex,
        model_name="sige_top_patch_gates",
    )

    # Scale coordinates to meters (or whatever you choose)
    if args.scale != 1.0:
        msh.geometry.x[:] *= args.scale

    # Build rho: either zero, or Gaussian point charges
    if args.qpoint:
        qpoints = parse_qpoints(args.qpoint)
        qpoints_scaled = [(x*args.scale, y*args.scale, z*args.scale, q) for (x, y, z, q) in qpoints]
        sigma_scaled = args.sigma * args.scale
        rho_expr = gaussian_point_charge_rho(msh, qpoints_scaled, sigma_scaled, args.q_units)
    else:
        rho_expr = fem.Constant(msh, PETSc.ScalarType(0.0))

    phi = solve_poisson_with_gate_dirichlet_and_rho(
        msh=msh,
        facet_tags=ft,
        layer_to_facet_tag=layer_to_facet_tag,
        gate_voltages=gate_voltages,
        default_gateV=args.default_gateV,
        deg=args.deg,
        eps_r=args.eps_r,
        rho_expr=rho_expr,
        phys=phys,
        bc_sides=args.bc_sides,
        bc_bottom=args.bc_bottom,
    )

    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)
    if RANK == 0:
        print(f"\n=== Solve summary ===\nphi min/max = [{gmin:.6e}, {gmax:.6e}]")

    os.makedirs(args.outdir, exist_ok=True)

    xdmf_solution = os.path.join(args.outdir, f"{args.basename}_phi.xdmf")
    xdmf_facets = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")

    phi_write = function_for_xdmf(phi, msh)

    with io.XDMFFile(COMM, xdmf_solution, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_write)

    with io.XDMFFile(COMM, xdmf_facets, "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, ft, msh)

    if RANK == 0:
        print("\nOutputs:")
        print(f"  {xdmf_solution}")
        print(f"  {xdmf_facets}")
        print("ParaView: open *_phi.xdmf and color by phi; open *_facet_tags.xdmf and color by gmsh:physical")

if __name__ == "__main__":
    main()
