#!/usr/bin/env python3
"""
make_square_triangle.py
Parametric square/rectangle 2D meshes with triangle elements and boundary tags.

CLI:
  python -m src.geometry.make_square_triangle --Lx 1.0 --Ly 1.0 --h 0.05 --outfile square

Tags:
  Cells: "domain"
  Facets: "left","right","bottom","top"

"""
from __future__ import annotations
import argparse
import gmsh
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import cpp
from .helpers import gmsh_model_to_mesh

def build(Lx: float=1.0, Ly: float=1.0, h: float=0.05):
    gmsh.initialize()
    gmsh.model.add("square")
    p = []
    p.append(gmsh.model.geo.addPoint(0, 0, 0, h))
    p.append(gmsh.model.geo.addPoint(Lx, 0, 0, h))
    p.append(gmsh.model.geo.addPoint(Lx, Ly, 0, h))
    p.append(gmsh.model.geo.addPoint(0, Ly, 0, h))

    l1 = gmsh.model.geo.addLine(p[0], p[1])
    l2 = gmsh.model.geo.addLine(p[1], p[2])
    l3 = gmsh.model.geo.addLine(p[2], p[3])
    l4 = gmsh.model.geo.addLine(p[3], p[0])

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # Physical groups
    pg_dom = gmsh.model.addPhysicalGroup(2, [s])
    gmsh.model.setPhysicalName(2, pg_dom, "domain")

    pg_left = gmsh.model.addPhysicalGroup(1, [l4]); gmsh.model.setPhysicalName(1, pg_left, "left")
    pg_right= gmsh.model.addPhysicalGroup(1, [l2]); gmsh.model.setPhysicalName(1, pg_right,"right")
    pg_bottom=gmsh.model.addPhysicalGroup(1, [l1]); gmsh.model.setPhysicalName(1, pg_bottom,"bottom")
    pg_top   =gmsh.model.addPhysicalGroup(1, [l3]); gmsh.model.setPhysicalName(1, pg_top,   "top")

    gmsh.model.mesh.generate(2)
    return gmsh.model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--outfile", type=str, default="square")
    args = ap.parse_args()

    model = build(args.Lx, args.Ly, args.h)
    domain, cell_tags, facet_tags, t2n_cell, t2n_facet = gmsh_model_to_mesh(MPI.COMM_WORLD)
    gmsh.finalize()

    # Save mesh/tags to XDMF
    with XDMFFile(MPI.COMM_WORLD, f"{args.outfile}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_information("cell_tags", str(t2n_cell))
        xdmf.write_information("facet_tags", str(t2n_facet))
        xdmf.write_meshtags(cell_tags)
        xdmf.write_meshtags(facet_tags)

if __name__ == "__main__":
    main()
