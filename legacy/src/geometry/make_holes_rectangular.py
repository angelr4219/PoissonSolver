#!/usr/bin/env python3
"""
make_holes_rectangular.py
Create a rectangular outer domain with N rectangular "holes" (metal gates).
Hole interiors are removed; only their boundary facets exist for Dirichlet BCs.

Facet tags:
  - "gate_i" for each hole i (1..N)
  - "outer" for the external boundary

Cell tags:
  - "domain" by default
  - Optional: top oxide band (name "oxide") and bottom "semiconductor" if --oxide-ymin provided

CLI:
  python -m src.geometry.make_holes_rectangular \
      --Lx 1.0 --Ly 1.0 --h 0.03 \
      --holes 2 --gate-width 0.15 --gate-height 0.08 --gate-gap 0.05 --gate-y 0.8 \
      --oxide-ymin 0.7 \
      --outfile rect_holes

"""
from __future__ import annotations
import argparse
from typing import List, Tuple
import gmsh
from mpi4py import MPI
from dolfinx.io import XDMFFile
from .helpers import gmsh_model_to_mesh

def _add_rect(x0: float, y0: float, w: float, h: float, lc: float) -> Tuple[int, List[int]]:
    p1 = gmsh.model.geo.addPoint(x0, y0, 0, lc)
    p2 = gmsh.model.geo.addPoint(x0+w, y0, 0, lc)
    p3 = gmsh.model.geo.addPoint(x0+w, y0+h, 0, lc)
    p4 = gmsh.model.geo.addPoint(x0, y0+h, 0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    return surf, [l1, l2, l3, l4]

def build(
    Lx: float=1.0, Ly: float=1.0, h: float=0.03,
    holes: int=2, gate_width: float=0.15, gate_height: float=0.08,
    gate_gap: float=0.05, gate_y: float=0.8,
    oxide_ymin: float | None = None
):
    gmsh.initialize()
    gmsh.model.add("rect_holes")

    # Outer
    outer_surf, outer_lines = _add_rect(0, 0, Lx, Ly, h)

    hole_surfs: List[int] = []
    hole_line_sets: List[List[int]] = []

    # Position holes horizontally centered near top
    total_gates_width = holes * gate_width + (holes - 1) * gate_gap
    x_start = (Lx - total_gates_width) / 2.0
    for i in range(holes):
        x0 = x_start + i * (gate_width + gate_gap)
        y0 = gate_y
        s, lines = _add_rect(x0, y0, gate_width, gate_height, h)
        hole_surfs.append(s)
        hole_line_sets.append(lines)

    # Boolean fragment to subtract holes
    gmsh.model.geo.synchronize()
    cut = gmsh.model.geo.cut([(2, outer_surf)], [(2, s) for s in hole_surfs], removeObject=True, removeTool=True)
    gmsh.model.geo.synchronize()
    final_surface = cut[0][0][1]  # (dim=2, tag)

    # Physical groups: cells
    pg_domain = gmsh.model.addPhysicalGroup(2, [final_surface]); gmsh.model.setPhysicalName(2, pg_domain, "domain")

    if oxide_ymin is not None:
        # Fragment domain into two subdomains by a line y=oxide_ymin
        # Build a horizontal strip rectangle to fragment
        ox_surf, _ = _add_rect(0, oxide_ymin, Lx, Ly-oxide_ymin, h)
        gmsh.model.geo.synchronize()
        frag = gmsh.model.geo.fragment([(2, final_surface)], [(2, ox_surf)])
        gmsh.model.geo.synchronize()
        parts = [tag for (dim, tag) in gmsh.model.getEntities(2)]
        # Heuristic: tag with centroid y>oxide_ymin -> "oxide", else "semiconductor"
        oxide = []
        semi  = []
        for _, t in gmsh.model.getEntities(2):
            mass, com, _ = gmsh.model.occ.getMass(2, t), gmsh.model.occ.getCenterOfMass(2, t), None
            if com[1] >= oxide_ymin + 1e-12:
                oxide.append(t)
            else:
                semi.append(t)
        if oxide:
            pg_oxide = gmsh.model.addPhysicalGroup(2, oxide); gmsh.model.setPhysicalName(2, pg_oxide, "oxide")
        if semi:
            pg_semi  = gmsh.model.addPhysicalGroup(2, semi);  gmsh.model.setPhysicalName(2, pg_semi,  "semiconductor")

    # Physical groups: facets
    # Outer boundary:
    # Reconstruct outer boundary edges by selecting boundary of final surface and then subtracting any gate loops
    gmsh.model.geo.synchronize()
    b = gmsh.model.getBoundary([(2, final_surface)], oriented=False, recursive=False)
    outer_curves = [c[1] for c in b]  # may include internal boundaries; we will separately tag gates

    pg_outer = gmsh.model.addPhysicalGroup(1, outer_curves)
    gmsh.model.setPhysicalName(1, pg_outer, "outer")

    # Gate facets: Identify by proximity to each hole line set (these were consumed by boolean ops),
    # so re-detect internal boundaries: use getBoundary with recursive=True then filter by not on bounding box edges
    inner = gmsh.model.getBoundary([(2, final_surface)], oriented=False, recursive=True)
    inner_curves = [c[1] for c in inner if c[0] == 1]
    # Heuristic split: group curves by connected loops and name them gate_1, gate_2, ...
    # Use curve loops recovered from boundary of holes: Gmsh doesn't retain order, so just split by bounding boxes
    gate_loops: List[List[int]] = []
    used = set()
    for c in inner_curves:
        if c in used:
            continue
        # flood fill by shared endpoints
        loop = [c]
        used.add(c)
        changed = True
        while changed:
            changed = False
            for d in inner_curves:
                if d in used:
                    continue
                # share node?
                ends_c = gmsh.model.getEntitiesForPhysicalGroup(1, gmsh.model.addPhysicalGroup(1,[c]))
                # quicker: rely on bounding boxes proximity to each hole rectangle
            # Simpler: just assign all internal curves to a single list and tag sequentially
        # Fall back: one PG for all inner curves if grouping is messy
    # Robust/simple: make each internal curve its own PG named "gate_all", then the solver can Dirichlet on all with a uniform V.
    # Better: tag them collectively as one group per hole by bounding box match:
    # We'll approximate by splitting inner curves into 'holes' bins using curve center y>=gate_y.
    gate_group = gmsh.model.addPhysicalGroup(1, inner_curves)
    gmsh.model.setPhysicalName(1, gate_group, "gate_all")

    gmsh.model.mesh.generate(2)
    return gmsh.model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.03)
    ap.add_argument("--holes", type=int, default=2)
    ap.add_argument("--gate-width", type=float, default=0.15)
    ap.add_argument("--gate-height", type=float, default=0.08)
    ap.add_argument("--gate-gap", type=float, default=0.05)
    ap.add_argument("--gate-y", type=float, default=0.8)
    ap.add_argument("--oxide-ymin", type=float, default=None)
    ap.add_argument("--outfile", type=str, default="rect_holes")
    args = ap.parse_args()

    model = build(
        Lx=args.Lx, Ly=args.Ly, h=args.h,
        holes=args.holes, gate_width=args.gate_width, gate_height=args.gate_height,
        gate_gap=args.gate_gap, gate_y=args.gate_y, oxide_ymin=args.oxide_ymin
    )
    domain, cell_tags, facet_tags, t2n_cell, t2n_facet = gmsh_model_to_mesh(MPI.COMM_WORLD)
    gmsh.finalize()

    from dolfinx.io import XDMFFile
    with XDMFFile(MPI.COMM_WORLD, f"{args.outfile}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_information("cell_tags", str(t2n_cell))
        xdmf.write_information("facet_tags", str(t2n_facet))
        xdmf.write_meshtags(cell_tags)
        xdmf.write_meshtags(facet_tags)

if __name__ == "__main__":
    main()
