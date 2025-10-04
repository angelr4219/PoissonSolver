#!/usr/bin/env python3
from __future__ import annotations
import gmsh
from dataclasses import dataclass
from typing import List, Tuple
from mpi4py import MPI
from dolfinx.io import gmshio

@dataclass
class Gate2D:
    cx: float; cy: float
    wx: float; wy: float
    V:  float   # potential on the hole boundary

@dataclass
class Gate3D:
    cx: float; cy: float; cz: float
    wx: float; wy: float; wz: float
    V:  float

def _init_gmsh(h: float):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

def build_rect_with_rod_holes_2d(comm: MPI.Comm, Lx: float, Ly: float, h: float,
                                 gates: List[Gate2D]):
    """
    2D domain [0,Lx]x[0,Ly] with rectangular holes.
    Facet tags:
      1 = outer boundary
      100+i = hole i boundary (Dirichlet gates)
    Cell tag:
      1 = bulk surface
    """
    _init_gmsh(h)
    gmsh.model.add("rect_holes_2d")
    occ = gmsh.model.occ
    outer = occ.addRectangle(0, 0, 0, Lx, Ly)

    holes = []
    for g in gates:
        xmin, ymin = g.cx - g.wx/2, g.cy - g.wy/2
        holes.append(occ.addRectangle(xmin, ymin, 0, g.wx, g.wy))

    if holes:
        occ.cut([(2, outer)], [(2, hid) for hid in holes], removeTool=False)
    occ.synchronize()

    # Physical groups
    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, tag=1)  # bulk

    # Outer boundary
    outer_edges = [e[1] for e in gmsh.model.getBoundary([(2, outer)], oriented=False) if e[0] == 1]
    gmsh.model.addPhysicalGroup(1, outer_edges, tag=1)

    # Each hole boundary
    for i, hid in enumerate(holes):
        hedges = [e[1] for e in gmsh.model.getBoundary([(2, hid)], oriented=False) if e[0] == 1]
        gmsh.model.addPhysicalGroup(1, hedges, tag=100 + i)

    gmsh.model.mesh.generate(2)
    out = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    domain, cell_tags, facet_tags = out[0], out[1], out[2]
    gmsh.finalize()
    return domain, cell_tags, facet_tags

def build_cube_with_rod_holes_3d(comm: MPI.Comm, Lx: float, Ly: float, Lz: float, h: float,
                                 gates: List[Gate3D]):
    """
    3D domain [0,Lx]x[0,Ly]x[0,Lz] with rectangular “rod” holes.
    Facet tags:
      1 = outer boundary surfaces
      100+i = hole i wall surfaces (Dirichlet gates)
    Cell tag:
      1 = bulk volume
    """
    _init_gmsh(h)
    gmsh.model.add("cube_rods_3d")
    occ = gmsh.model.occ
    box = occ.addBox(0, 0, 0, Lx, Ly, Lz)

    holes = []
    for g in gates:
        xmin, ymin, zmin = g.cx - g.wx/2, g.cy - g.wy/2, g.cz - g.wz/2
        holes.append(occ.addBox(xmin, ymin, zmin, g.wx, g.wy, g.wz))

    if holes:
        occ.cut([(3, box)], [(3, hid) for hid in holes], removeTool=False)
    occ.synchronize()

    # Physical groups
    vols = [v[1] for v in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, vols, tag=1)  # bulk

    # Outer boundary of the outer box
    outer_faces = [e[1] for e in gmsh.model.getBoundary([(3, box)], oriented=False) if e[0] == 2]
    gmsh.model.addPhysicalGroup(2, outer_faces, tag=1)

    # Gate walls
    for i, hid in enumerate(holes):
        walls = [e[1] for e in gmsh.model.getBoundary([(3, hid)], oriented=False) if e[0] == 2]
        gmsh.model.addPhysicalGroup(2, walls, tag=100 + i)

    gmsh.model.mesh.generate(3)
    out = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    domain, cell_tags, facet_tags = out[0], out[1], out[2]
    gmsh.finalize()
    return domain, cell_tags, facet_tags

