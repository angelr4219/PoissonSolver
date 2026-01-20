#!/usr/bin/env python3
"""
Patch for DOLFINx create_submesh API differences (3 vs 4 return values).

This file is NOT your full solver. It contains ONLY the robust replacement
function `write_top_view_gate_id` that you should paste into your existing
cif_geometry_stack_poisson.py (or copy from here into that file).

How to use:
1) Open your existing cif_geometry_stack_poisson.py
2) Find the function write_top_view_gate_id(...)
3) Replace its entire body with the function below (or replace the whole function)
"""

from __future__ import annotations

from typing import List
import numpy as np
from mpi4py import MPI

from dolfinx import fem, io, mesh as dmesh

COMM = MPI.COMM_WORLD
RANK = COMM.rank


# Keep this class signature consistent with your file
class Box2D:
    def __init__(self, layer: str, W: float, H: float, xc: float, yc: float):
        self.layer = layer
        self.W = W
        self.H = H
        self.xc = xc
        self.yc = yc

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


def write_top_view_gate_id(
    msh: dmesh.Mesh,
    gate_plane: float,
    gates: List[Box2D],
    outpath: str,
    tol: float,
) -> None:
    """
    Build a 2D submesh of the plane z = gate_plane (boundary facets),
    then write a DG0 "gate_id" on that surface via centroid classification.

    Robust to DOLFINx versions where:
      create_submesh(...) returns either (submesh, entity_map, vertex_map)
      or (submesh, entity_map, vertex_map, geom_map)  (newer)
    """
    fdim = msh.topology.dim - 1

    def on_plane(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[2], gate_plane, atol=tol)

    plane_facets = dmesh.locate_entities_boundary(msh, fdim, on_plane)
    if plane_facets.size == 0:
        if RANK == 0:
            print("[WARN] No facets found on gate_plane for top view export.")
        return

    # ---- Version-safe create_submesh ----
    out = dmesh.create_submesh(msh, fdim, plane_facets)
    smesh = out[0]
    # entity_map = out[1]  # not needed
    # -----------------------------------

    tdim = smesh.topology.dim
    smesh.topology.create_connectivity(tdim, 0)
    c2v = smesh.topology.connectivity(tdim, 0)

    V0 = fem.functionspace(smesh, ("DG", 0))
    gate_id = fem.Function(V0, name="gate_id")
    gate_id.x.array[:] = 0

    X = smesh.geometry.x

    # Number of cells we can safely loop over (local)
    im = smesh.topology.index_map(tdim)
    ncells_local = im.size_local
    ncells_local = min(ncells_local, gate_id.x.array.size)

    for c in range(ncells_local):
        vs = c2v.links(c)
        xc = X[vs].mean(axis=0)
        x, y = float(xc[0]), float(xc[1])

        gid = 0
        # Assign gid based on first matching gate rectangle
        for i, g in enumerate(gates, start=1):
            if (
                (x >= g.xmin - tol)
                and (x <= g.xmax + tol)
                and (y >= g.ymin - tol)
                and (y <= g.ymax + tol)
            ):
                gid = i
                break

        gate_id.x.array[c] = gid

    with io.XDMFFile(COMM, outpath, "w") as xdmf:
        xdmf.write_mesh(smesh)
        xdmf.write_function(gate_id)

    if RANK == 0:
        print(f"[INFO] Wrote top-view layout (gate_id): {outpath}")
