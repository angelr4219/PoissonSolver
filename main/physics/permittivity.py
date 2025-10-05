# src/permittivity.py
from __future__ import annotations
import numpy as np
from dolfinx import fem, mesh

def dg0_from_indicator(domain, indicator, val_true: float, val_false: float):
    """
    Build DG0 epsilon from a boolean indicator function on cell centroids.
    indicator: function (x->bool) operating on coordinates array.
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)  # ensure cell->cell connectivity
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="epsilon")
    cells_true  = mesh.locate_entities(domain, tdim, indicator)
    dofs_true   = fem.locate_dofs_topological(W, tdim, cells_true)
    eps.x.array[:] = val_false
    eps.x.array[dofs_true] = val_true
    return eps
