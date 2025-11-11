from __future__ import annotations
import numpy as np
from dolfinx import fem, mesh as dmesh

def piecewise_eps_DG0_2regions(domain, split_predicate, eps1: float, eps2: float):
    """DG0 epsilon: region 1 if split_predicate(x) else region 2."""
    tdim = domain.topology.dim
    cells_reg1 = dmesh.locate_entities(domain, tdim, lambda x: split_predicate(x))
    cells_reg2 = dmesh.locate_entities(domain, tdim, lambda x: ~split_predicate(x))
    DG0 = fem.FunctionSpace(domain, ("DG", 0))
    eps = fem.Function(DG0, name="epsilon")
    eps.x.array[:] = eps2  # default
    # assign region 1
    if len(cells_reg1) > 0:
        dofs = fem.locate_dofs_topological(DG0, tdim, cells_reg1)
        eps.x.array[dofs] = eps1
    return eps, DG0
