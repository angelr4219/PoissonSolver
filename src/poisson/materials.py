from __future__ import annotations
import numpy as np
from dolfinx import fem, mesh

def piecewise_eps_DG0_2regions(domain, split_predicate, eps1: float, eps2: float):
    """
    Create DG0 ε field where cells satisfying split_predicate(x) are tagged as region 1 else region 2.

    split_predicate: callable R^{dim}->{bool array}, evaluated vectorized on coordinates.
                     Example: lambda x: x[1] > 1e-14 (y>0) or x[2] > 1e-14 (z>0)
    """
    tdim = domain.topology.dim
    cells_reg1 = mesh.locate_entities(domain, tdim, lambda x: split_predicate(x))
    cells_reg2 = mesh.locate_entities(domain, tdim, lambda x: ~split_predicate(x))

    DG0 = fem.FunctionSpace(domain, ("DG", 0))
    eps = fem.Function(DG0, name="epsilon")

    # Start with eps2 everywhere (safer if some cells not found due to tolerances)
    eps.x.array[:] = eps2

    # Assign reg1
    dofs = fem.locate_dofs_topological(DG0, tdim, cells_reg1)
    eps.x.array[dofs] = eps1
    return eps, DG0
