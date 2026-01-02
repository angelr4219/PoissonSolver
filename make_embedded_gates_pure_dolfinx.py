#!/usr/bin/env python3
"""
Pure-Dolfinx 3D rectangular device with "virtual" gate regions tagged on top surface.
No gmsh or OCC dependencies.

- Domain: x,y in [-Lx/2, Lx/2], z in [0,H]
- Three gate rectangles near the top tagged 101, 102, 103.
- Top, bottom, and side surfaces tagged 11, 10, 12.
"""

from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io, cpp

def make_device(Lx, Ly, H, nx, ny, nz):
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([-Lx/2, -Ly/2, 0.0]), np.array([Lx/2, Ly/2, H])],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron
    )
    return domain

def tag_facets(domain, gates):
    tdim = domain.topology.dim
    fdim = tdim - 1
    coords = domain.geometry.x

    # Initialize mesh tags
    facet_indices = []
    facet_values = []

    def near(a,b,eps=1e-12): return abs(a-b) < eps

    # --- Top, bottom, sides
    facets_top = mesh.locate_entities_boundary(domain, fdim, lambda x: near(x[2], 0.0))
    facets_bot = mesh.locate_entities_boundary(domain, fdim, lambda x: near(x[2], H))
    facets_side = mesh.locate_entities_boundary(domain, fdim, lambda x: (near(x[0], -Lx/2) | near(x[0], Lx/2)
                                                                       | near(x[1], -Ly/2) | near(x[1], Ly/2)))

    facet_indices += facets_bot.tolist()
    facet_values += [10]*len(facets_bot)

    facet_indices += facets_top.tolist()
    facet_values += [11]*len(facets_top)

    facet_indices += facets_side.tolist()
    facet_values += [12]*len(facets_side)

    # --- Gate rectangles on top (z≈0)
    for g in gates:
        tag = int(g["tag"])
        x0, y0, dx, dy = g["x0"], g["y0"], g["dx"], g["dy"]
        facets_gate = mesh.locate_entities_boundary(
            domain, fdim,
            lambda x: (near(x[2], 0.0) &
                       (x0 <= x[0]) & (x[0] <= x0+dx) &
                       (y0 <= x[1]) & (x[1] <= y0+dy))
        )
        facet_indices += facets_gate.tolist()
        facet_values += [tag]*len(facets_gate)

    facet_tags = mesh.meshtags(domain, fdim, np.array(facet_indices, dtype=np.int32),
                               np.array(facet_values, dtype=np.int32))
    return facet_tags

if __name__ == "__main__":
    Lx, Ly, H = 2e-7, 2e-7, 1e-7
    nx, ny, nz = 40, 40, 20

    gates = [
        {"x0": -90e-9, "y0": -50e-9, "dx": 70e-9, "dy": 100e-9, "tag": 101},
        {"x0": -10e-9, "y0": -50e-9, "dx": 70e-9, "dy": 100e-9, "tag": 102},
        {"x0":  70e-9, "y0": -50e-9, "dx": 70e-9, "dy": 100e-9, "tag": 103},
    ]

    domain = make_device(Lx, Ly, H, nx, ny, nz)
    facet_tags = tag_facets(domain, gates)

    with io.XDMFFile(domain.comm, "device_with_gates.xdmf", "w") as xf:
        xf.write_mesh(domain)
    with io.XDMFFile(domain.comm, "device_with_gates_facets.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_meshtags(facet_tags, domain.geometry)

    if domain.comm.rank == 0:
        print("[OK] wrote device_with_gates.xdmf + facets (no gmsh).")

