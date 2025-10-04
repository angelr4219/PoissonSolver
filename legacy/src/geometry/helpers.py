#!/usr/bin/env python3
"""
helpers.py
Shared Gmsh → dolfinx helpers: model creation, mesh conversion, tag maps.

"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import mesh, cpp

def gmsh_model_to_mesh(
    comm: MPI.Comm,
    model: Optional[gmsh.model] = None,
    rank: int = 0,
    gdim: int = 2,
) -> Tuple[mesh.Mesh, cpp.mesh.MeshTags_int32, cpp.mesh.MeshTags_int32, Dict[int, str], Dict[int, str]]:
    """
    Convert current Gmsh model to dolfinx mesh + cell/facet tags + name maps.

    Returns
    -------
    (domain, cell_tags, facet_tags, cell_tag2name, facet_tag2name)
    """
    if model is None:
        model = gmsh.model
    # Collect physical groups → names
    phys = model.getPhysicalGroups()
    tag2name_cell: Dict[int, str] = {}
    tag2name_facet: Dict[int, str] = {}
    for dim, tag in phys:
        name = model.getPhysicalName(dim, tag) or f"pg_{dim}_{tag}"
        if dim == gdim:
            tag2name_cell[tag] = name
        elif dim == gdim - 1:
            tag2name_facet[tag] = name

    domain, cell_tags, facet_tags, *rest = gmshio.model_to_mesh(model, comm, rank, gdim=gdim)
    return domain, cell_tags, facet_tags, tag2name_cell, tag2name_facet
