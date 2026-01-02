#!/usr/bin/env python3
"""
cif_gate_geometry_laplace.py

One-file, end-to-end demo:
1) Parse a CIF-like "Layer ... Box W H Xc Yc;" spec (like you pasted).
2) Build a 3D SiGe device volume (extruded base footprint) using gmsh.
3) Import mesh into DOLFINx.
4) Create facet tags for each gate rectangle on the TOP surface (z = z_top)
   using geometric predicates from the CIF boxes.
5) Solve Laplace/Poisson with rho=0 and ALL boundary Dirichlet = 0 V
   (solution should be identically ~0).
6) Write XDMF outputs so you can verify:
   - the 3D device volume
   - the labeled gate patches on the top surface

Notes:
- This is geometry + tagging validation. The mesh is a simple box volume.
- Gate shapes are encoded as tagged boundary facets on the top surface.
- If you want perfectly sharp rectangular patch boundaries, make h small enough
  so facets resolve the rectangles.

Run in dolfinx container (example commands at bottom of this message).
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from mpi4py import MPI

import gmsh
import ufl
from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
from petsc4py import PETSc


COMM = MPI.COMM_WORLD
RANK = COMM.rank


DEFAULT_CIF_TEXT = r"""
(                                           )
( Geometric Specification                   )
( Wedge dot bar dot Wedge                   )

Layer base;                                  (Required Layer that specifies the device size.          )
Box 650 510 0 35;                             (Box 600 by 300 centered at [0,0] specifies device size  )
                                             (The Layer name "base" is used to specify Shottky value. )


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


( Screening gates at top of device )
Layer CG0A;
Box 640 40 0 155;

Layer SG;
Box 640 40 0 0;


( Gates for sensor dots )
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



END                                          (END statement indicates end of geometry specification. )
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
    """
    Parse:
      Layer NAME;
      Box W H Xc Yc;

    Ignores comments and blank lines. Stops at END if present.
    """
    # strip inline "( ... )" comments (non-greedy)
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

        # ignore anything else silently (robust against extra CIF tokens)
    return boxes


def build_gmsh_box_volume(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    h: float,
    vol_phys_id: int = 1,
    model_name: str = "sige_device",
) -> Tuple[mesh.Mesh, mesh.MeshTags, mesh.MeshTags]:
    """
    Create a simple 3D box mesh in gmsh (OCC), tag the volume, import into dolfinx.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)
    gmsh.model.add(model_name)

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("Invalid extents for box volume")

    # OCC box
    box_tag = gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz)
    gmsh.model.occ.synchronize()

    # Mesh sizing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    # Physical group for the volume
    gmsh.model.addPhysicalGroup(3, [box_tag], vol_phys_id)
    gmsh.model.setPhysicalName(3, vol_phys_id, "SiGe_volume")

    # Generate tetra mesh
    gmsh.model.mesh.generate(3)

    # Convert to dolfinx
    msh, ct, ft = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)

    gmsh.finalize()
    return msh, ct, ft


def make_gate_facet_tags_from_boxes(
    msh: mesh.Mesh,
    boxes: List[Box2D],
    z_top: float,
    tol_z: float,
    tol_xy: float,
    include_outer_top_background: bool = True,
) -> Tuple[mesh.MeshTags, Dict[str, int], Dict[str, int]]:
    """
    Create a facet MeshTags where each gate layer gets a unique integer id.
    Tags are assigned on facets on the top boundary (|z - z_top| <= tol_z)
    AND inside each rectangle (xmin-tol <= x <= xmax+tol etc).

    Returns:
      mt_facets, layer_to_tag, layer_to_nfacets
    """
    fdim = msh.topology.dim - 1

    # Precompute all top facets (z ~= z_top)
    def is_top(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[2], z_top, atol=tol_z)

    top_facets = mesh.locate_entities_boundary(msh, fdim, is_top)

    # Will build an array of tags for a subset of facets
    # We'll assign each gate by selecting from top_facets using a tighter predicate.
    layer_names = []
    for b in boxes:
        if b.layer.lower() == "base":
            continue
        if b.layer not in layer_names:
            layer_names.append(b.layer)

    layer_to_tag: Dict[str, int] = {}
    # Reserve:
    # 1 = outer_top_background (optional)
    # 2.. = gates
    next_tag = 2
    if include_outer_top_background:
        layer_to_tag["TOP_BACKGROUND"] = 1

    for name in layer_names:
        layer_to_tag[name] = next_tag
        next_tag += 1

    # Start with background tagging for all top facets
    facet_values = np.zeros_like(top_facets, dtype=np.int32)
    if include_outer_top_background:
        facet_values[:] = layer_to_tag["TOP_BACKGROUND"]

    layer_to_nfacets: Dict[str, int] = {k: 0 for k in layer_to_tag.keys()}

    # Assign gates (overwrites background where matched)
    for b in boxes:
        if b.layer.lower() == "base":
            continue
        tag = layer_to_tag[b.layer]

        xmin, xmax = b.xmin - tol_xy, b.xmax + tol_xy
        ymin, ymax = b.ymin - tol_xy, b.ymax + tol_xy

        def in_gate(x: np.ndarray) -> np.ndarray:
            # x shape (3, N)
            on_top = np.isclose(x[2], z_top, atol=tol_z)
            inside = (x[0] >= xmin) & (x[0] <= xmax) & (x[1] >= ymin) & (x[1] <= ymax)
            return on_top & inside

        gate_facets = mesh.locate_entities_boundary(msh, fdim, in_gate)

        # Intersect with top_facets for indexing safety
        gate_set = set(int(i) for i in gate_facets.tolist())
        mask = np.array([int(f) in gate_set for f in top_facets], dtype=bool)

        facet_values[mask] = tag
        layer_to_nfacets[b.layer] = layer_to_nfacets.get(b.layer, 0) + int(mask.sum())

    # Build MeshTags for all *top* facets only (compact)
    # Note: Paraview is fine with a partial facet tag file.
    sort_idx = np.argsort(top_facets)
    mt = mesh.meshtags(msh, fdim, top_facets[sort_idx], facet_values[sort_idx])

    # Count totals per tag (nice printout)
    for name, t in layer_to_tag.items():
        layer_to_nfacets[name] = int(np.sum(facet_values == t))

    return mt, layer_to_tag, layer_to_nfacets


def solve_laplace_all_dirichlet_zero(msh: mesh.Mesh, deg: int = 1) -> fem.Function:
    """
    Solve ∇·(∇phi) = 0 with phi=0 on the entire boundary.
    """
    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx

    # Dirichlet on all exterior facets
    fdim = msh.topology.dim - 1

    def on_boundary(x: np.ndarray) -> np.ndarray:
        # locate_entities_boundary already restricts to boundary entities,
        # predicate can be True everywhere
        return np.full(x.shape[1], True, dtype=bool)

    bfacets = mesh.locate_entities_boundary(msh, fdim, on_boundary)
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bdofs, V)

    # Solve
    problem = fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        u=phi,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="CIF-like gate geometry -> gmsh mesh -> dolfinx facet tags -> Laplace")
    ap.add_argument("--cif", type=str, default="", help="Path to CIF-like text file. If empty, uses built-in example.")
    ap.add_argument("--outdir", type=str, default="results/cif_geometry_demo", help="Output directory for XDMF files.")
    ap.add_argument("--basename", type=str, default="cif_geom", help="Base name for output files (no extension).")

    ap.add_argument("--Lz", type=float, default=120.0, help="SiGe thickness (same units as CIF unless --scale used).")
    ap.add_argument("--z_top", type=float, default=0.0, help="Top surface z coordinate.")
    ap.add_argument("--extrude_down", type=int, default=1, help="1: volume goes from z_top down to z_top-Lz. 0: up.")
    ap.add_argument("--h", type=float, default=15.0, help="Target gmsh element size (same units as geometry).")
    ap.add_argument("--deg", type=int, default=1, help="CG polynomial degree for Laplace solve.")

    ap.add_argument("--scale", type=float, default=1.0, help="Multiply all CIF coordinates by this scale (e.g. 1e-9).")

    ap.add_argument("--tol_z", type=float, default=-1.0, help="Z tolerance for selecting top facets. If <0, auto.")
    ap.add_argument("--tol_xy", type=float, default=-1.0, help="XY tolerance for selecting facets inside boxes. If <0, auto.")

    args = ap.parse_args()

    # Load CIF text
    if args.cif:
        if RANK == 0:
            print(f"[INFO] Reading CIF-like spec from: {args.cif}")
        with open(args.cif, "r", encoding="utf-8") as f:
            cif_text = f.read()
    else:
        if RANK == 0:
            print("[INFO] Using built-in CIF-like example text.")
        cif_text = DEFAULT_CIF_TEXT

    boxes = parse_cif_boxes(cif_text)
    if not boxes:
        raise RuntimeError("No boxes parsed from CIF text.")

    # Find base box
    base_boxes = [b for b in boxes if b.layer.lower() == "base"]
    if len(base_boxes) != 1:
        raise RuntimeError(f"Expected exactly 1 base box, found {len(base_boxes)}.")
    base = base_boxes[0]

    # Apply scale
    def scl(x: float) -> float:
        return args.scale * x

    scaled_boxes = [
        Box2D(b.layer, scl(b.W), scl(b.H), scl(b.xc), scl(b.yc))
        for b in boxes
    ]
    base = [b for b in scaled_boxes if b.layer.lower() == "base"][0]

    # Extents from base
    xmin, xmax = base.xmin, base.xmax
    ymin, ymax = base.ymin, base.ymax

    z_top = scl(args.z_top) if args.scale != 1.0 else args.z_top
    Lz = scl(args.Lz) if args.scale != 1.0 else args.Lz

    if args.extrude_down:
        zmin, zmax = z_top - Lz, z_top
    else:
        zmin, zmax = z_top, z_top + Lz

    # Auto tolerances
    Lref = max(abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin))
    tol_z = args.tol_z if args.tol_z > 0 else 1.0e-8 * max(1.0, Lref)
    tol_xy = args.tol_xy if args.tol_xy > 0 else 1.0e-10 * max(1.0, Lref)

    if RANK == 0:
        print("\n=== Parsed geometry (scaled) ===")
        print(f"scale={args.scale:g}")
        print(f"BASE: W={base.W:.6g}, H={base.H:.6g}, center=({base.xc:.6g},{base.yc:.6g})")
        print(f"XY extents: x=[{xmin:.6g},{xmax:.6g}], y=[{ymin:.6g},{ymax:.6g}]")
        print(f"Z extents:  z=[{zmin:.6g},{zmax:.6g}] (z_top={z_top:.6g}, Lz={Lz:.6g})")
        print(f"Mesh size h={args.h*g if False else args.h:.6g} (passed directly to gmsh)")
        print(f"Auto tolerances: tol_z={tol_z:.3e}, tol_xy={tol_xy:.3e}")
        print(f"Total boxes parsed: {len(scaled_boxes)} (including base)")

    # Build gmsh mesh for the SiGe volume
    msh, ct, ft = build_gmsh_box_volume(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        h=scl(args.h) if args.scale != 1.0 else args.h,
        vol_phys_id=1,
        model_name="sige_box",
    )

    # Build gate facet tags from CIF rectangles (on top surface)
    mt_gates, layer_to_tag, layer_to_nfacets = make_gate_facet_tags_from_boxes(
        msh=msh,
        boxes=scaled_boxes,
        z_top=z_top if not args.extrude_down else zmax,  # if extruded down, top is zmax
        tol_z=tol_z,
        tol_xy=tol_xy,
        include_outer_top_background=True,
    )

    if RANK == 0:
        print("\n=== Gate tag map (facet tags on top surface) ===")
        for name, tag in layer_to_tag.items():
            print(f"  tag {tag:3d} : {name:12s}   (top facets matched: {layer_to_nfacets.get(name,0)})")

    # Solve Laplace with all boundary Dirichlet = 0
    phi = solve_laplace_all_dirichlet_zero(msh, deg=args.deg)

    # Report min/max
    local_min = float(np.min(phi.x.array))
    local_max = float(np.max(phi.x.array))
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)
    if RANK == 0:
        print("\n=== Solve summary ===")
        print(f"phi min/max = [{gmin:.3e}, {gmax:.3e}] (should be ~0, up to solver tol)")

    # Write outputs
    os.makedirs(args.outdir, exist_ok=True)
    xdmf_path = os.path.join(args.outdir, f"{args.basename}.xdmf")
    tags_path = os.path.join(args.outdir, f"{args.basename}_facet_tags.xdmf")

    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)

    with io.XDMFFile(COMM, tags_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(mt_gates)

    if RANK == 0:
        print("\n=== Wrote files ===")
        print(f"  Mesh + phi:        {xdmf_path}")
        print(f"  Top facet tags:    {tags_path}")
        print("\nParaView tip:")
        print("  - Open BOTH XDMF files.")
        print("  - For the facet tag file, color by the 'values' array to see gate patches.")
        print("  - If rectangles look blocky, reduce --h (finer mesh).")


if __name__ == "__main__":
    main()
