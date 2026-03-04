# src/geometry.py
from __future__ import annotations
import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import gmshio

def unit_square(comm=MPI.COMM_WORLD, nx=64, ny=64):
    return mesh.create_rectangle(comm, [np.array([0.,0.]), np.array([1.,1.])],
                                 n=(nx, ny), cell_type=mesh.CellType.triangle)

def unit_cube(comm=MPI.COMM_WORLD, n=(24,24,24)):
    return mesh.create_box(comm, [np.array([0,0,0]), np.array([1,1,1])],
                           n=n, cell_type=mesh.CellType.tetrahedron)

def disk_in_box(comm=MPI.COMM_WORLD, Lx=1.0, Ly=1.0, R=0.2):
    """Make a square with a centered circular hole (outer facets tag=1, hole=2)."""
    gmsh.initialize()
    gmsh.model.add("disk2d")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    circ = gmsh.model.occ.addDisk(Lx/2, Ly/2, 0, R, R)
    geom = gmsh.model.occ.cut([(2, rect)], [(2, circ)], removeObject=True, removeTool=False)
    gmsh.model.occ.synchronize()
    # tag boundaries
    b_all  = gmsh.model.getBoundary(geom[0], oriented=False, recursive=False)
    b_hole = gmsh.model.getBoundary([(2, circ)], oriented=False, recursive=False)
    outer  = [c for c in b_all if c not in b_hole]
    gmsh.model.addPhysicalGroup(1, [e[1] for e in outer], tag=1)
    gmsh.model.addPhysicalGroup(1, [e[1] for e in b_hole], tag=2)
    gmsh.model.addPhysicalGroup(2, [geom[0][0][1]], tag=1)
    gmsh.model.mesh.generate(2)
    # robust unpack (3+ returns)
    domain, cell_tags, facet_tags, *rest = gmshio.model_to_mesh(gmsh.model, comm, 0)
    gmsh.finalize()
    return domain, cell_tags, facet_tags
