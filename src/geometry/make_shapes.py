#!/usr/bin/env python3
"""
make_shapes.py
Generate canonical shapes with tagging for 2D (disk/rod sections) and 3D (sphere/cylinder/+1 extra: ellipse).

2D modes:
  --mode disk2d, rod2d, ellipse2d

3D modes:
  --mode sphere3d, cylinder3d, wedge3d

For each: tags cells by "domain" (and "inclusion"), facets by "outer" and "inclusion_boundary".

CLI examples:
  python -m src.geometry.make_shapes --mode disk2d --R 0.2 --center 0.5 0.5 --Lx 1 --Ly 1 --h 0.03 --outfile disk2d
  python -m src.geometry.make_shapes --mode sphere3d --R 0.25 --L 1 --h 0.08 --outfile sphere3d

"""
from __future__ import annotations
import argparse, sys
from typing import Tuple
import gmsh
from mpi4py import MPI
from ._utils import gmsh_model_to_mesh

def _square(Lx, Ly, lc):
    p = []
    p.append(gmsh.model.geo.addPoint(0, 0, 0, lc))
    p.append(gmsh.model.geo.addPoint(Lx, 0, 0, lc))
    p.append(gmsh.model.geo.addPoint(Lx, Ly, 0, lc))
    p.append(gmsh.model.geo.addPoint(0, Ly, 0, lc))
    l1 = gmsh.model.geo.addLine(p[0], p[1])
    l2 = gmsh.model.geo.addLine(p[1], p[2])
    l3 = gmsh.model.geo.addLine(p[2], p[3])
    l4 = gmsh.model.geo.addLine(p[3], p[0])
    loop = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4])
    s = gmsh.model.geo.addPlaneSurface([loop])
    return s, [l1,l2,l3,l4]

def build_2d(mode: str, Lx: float, Ly: float, lc: float, **kw):
    gmsh.initialize()
    gmsh.model.add(mode)
    s_out, outer_edges = _square(Lx, Ly, lc)
    gmsh.model.occ.synchronize()

    inc = None
    if mode == "disk2d":
        cx, cy = kw.get("center", (Lx/2, Ly/2))
        R = kw.get("R", min(Lx,Ly)/4)
        inc = gmsh.model.occ.addDisk(cx, cy, 0, R, R)
    elif mode == "rod2d":
        # vertical rod (rectangle)
        x0 = kw.get("x0", Lx*0.45)
        w  = kw.get("w", 0.10)
        inc, _ = _square(w, Ly*0.8, lc)  # create then translate
        gmsh.model.geo.translate([(2,inc)], x0, Ly*0.1, 0)
    elif mode == "ellipse2d":
        cx, cy = kw.get("center", (Lx/2, Ly*0.7))
        a, b   = kw.get("a", 0.25), kw.get("b", 0.12)
        inc = gmsh.model.geo.addEllipse(cx, cy, 0, a, b)
    else:
        raise ValueError("Unsupported 2D mode")

    gmsh.model.occ.synchronize()
    if inc is not None:
        cut = gmsh.model.occ.fragment([(2, s_out)], [(2, inc)])
        gmsh.model.occ.synchronize()

    # Tag cells
    ent2d = [e for e in gmsh.model.getEntities(2)]
    outer = []
    inclusion = []
    for dim, tag in ent2d:
        mass, c = gmsh.model.occ.getMass(dim, tag), gmsh.model.occ.getCenterOfMass(dim, tag)
        # simple centroid test: if |c - center| small and area smaller than outer, label as inclusion
        # Instead, tag all surfaces except one with "inclusion" by area
    areas = [(tag, gmsh.model.occ.getMass(2, tag)) for _,tag in ent2d]
    largest = max(areas, key=lambda t: t[1])[0]
    for _,tag in ent2d:
        if tag == largest:
            outer.append(tag)
        else:
            inclusion.append(tag)

    pg_outer  = gmsh.model.addPhysicalGroup(2, outer);     gmsh.model.setPhysicalName(2, pg_outer,  "domain")
    if inclusion:
        pg_incl = gmsh.model.addPhysicalGroup(2, inclusion); gmsh.model.setPhysicalName(2, pg_incl, "inclusion")

    # Facets
    gmsh.model.occ.synchronize()
    b_all = gmsh.model.getBoundary([(2,t) for t in outer], oriented=False, recursive=True)
    outer_edges = list({c[1] for c in b_all})
    pg_outer_f = gmsh.model.addPhysicalGroup(1, outer_edges); gmsh.model.setPhysicalName(1, pg_outer_f, "outer")

    if inclusion:
        b_inc = gmsh.model.getBoundary([(2,t) for t in inclusion], oriented=False, recursive=True)
        inc_edges = list({c[1] for c in b_inc})
        pg_inc_f = gmsh.model.addPhysicalGroup(1, inc_edges); gmsh.model.setPhysicalName(1, pg_inc_f, "inclusion_boundary")

    gmsh.model.mesh.generate(2)
    return gmsh.model

def build_3d(mode: str, L: float, lc: float, **kw):
    gmsh.initialize()
    gmsh.model.add(mode)
    # Outer box
    box = gmsh.model.occ.addBox(0,0,0, L,L,L)
    gmsh.model.occ.synchronize()

    inclusion = None
    if mode == "sphere3d":
        R = kw.get("R", L/4)
        inclusion = gmsh.model.occ.addSphere(L/2, L/2, L/2, R)
    elif mode == "cylinder3d":
        R = kw.get("R", L/6)
        H = kw.get("H", L*0.8)
        inclusion = gmsh.model.occ.addCylinder(L/2, L/2, (L-H)/2, 0,0,H, R)
    elif mode == "wedge3d":
        # Simple wedge from a triangular prism
        p1=gmsh.model.occ.addPoint(0,0,0); p2=gmsh.model.occ.addPoint(L,0,0); p3=gmsh.model.occ.addPoint(0,L,0)
        t = gmsh.model.occ.addTriangle(p1,p2,p3)
        inclusion = gmsh.model.occ.extrude([(2,t)], 0,0,L/2)[1][1]
    else:
        raise ValueError("Unsupported 3D mode")

    gmsh.model.occ.synchronize()
    frag = gmsh.model.occ.fragment([(3, box)], [(3, inclusion)])
    gmsh.model.occ.synchronize()

    # Tag volumes
    vols = [v[1] for v in gmsh.model.getEntities(3)]
    volumes_by_vol = [(v, gmsh.model.occ.getMass(3, v)) for v in vols]
    largest = max(volumes_by_vol, key=lambda t: t[1])[0]
    outer = [largest]
    inclusion_vols = [v for v in vols if v != largest]

    pg_dom = gmsh.model.addPhysicalGroup(3, outer); gmsh.model.setPhysicalName(3, pg_dom, "domain")
    if inclusion_vols:
        pg_inc = gmsh.model.addPhysicalGroup(3, inclusion_vols); gmsh.model.setPhysicalName(3, pg_inc, "inclusion")

    # Facets (surfaces)
    gmsh.model.occ.synchronize()
    s_dom = gmsh.model.getBoundary([(3, v) for v in outer], oriented=False, recursive=True)
    s_dom_ids = list({s[1] for s in s_dom})
    pg_outer = gmsh.model.addPhysicalGroup(2, s_dom_ids); gmsh.model.setPhysicalName(2, pg_outer, "outer")

    if inclusion_vols:
        s_inc = gmsh.model.getBoundary([(3,v) for v in inclusion_vols], oriented=False, recursive=True)
        s_inc_ids = list({s[1] for s in s_inc})
        pg_inc_f = gmsh.model.addPhysicalGroup(2, s_inc_ids); gmsh.model.setPhysicalName(2, pg_inc_f, "inclusion_boundary")

    gmsh.model.mesh.generate(3)
    return gmsh.model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=[
        "disk2d","rod2d","ellipse2d","sphere3d","cylinder3d","wedge3d"
    ])
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--R", type=float, default=0.25)
    ap.add_argument("--outfile", type=str, default="shape")
    args = ap.parse_args()

    if args.mode.endswith("2d"):
        model = build_2d(args.mode, args.Lx, args.Ly, args.h, R=args.R)
        domain_dim = 2
    else:
        model = build_3d(args.mode, args.L, args.h, R=args.R)
        domain_dim = 3

    from dolfinx.io import XDMFFile
    from ._utils import gmsh_model_to_mesh
    domain, cell_tags, facet_tags, t2n_cell, t2n_facet = gmsh_model_to_mesh(MPI.COMM_WORLD, gdim=domain_dim)
    gmsh.finalize()

    with XDMFFile(MPI.COMM_WORLD, f"{args.outfile}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_information("cell_tags", str(t2n_cell))
        xdmf.write_information("facet_tags", str(t2n_facet))
        xdmf.write_meshtags(cell_tags)
        xdmf.write_meshtags(facet_tags)

if __name__ == "__main__":
    main()
