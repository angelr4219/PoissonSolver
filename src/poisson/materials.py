from __future__ import annotations
import numpy as np
<<<<<<< HEAD
from dolfinx import fem

def make_eps_r_constant(domain, epsr: float):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")
    eps.x.array[:] = epsr
    return eps

def make_eps_r_two_layer(domain, t_ox: float, epsr_top: float, epsr_bulk: float):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")
    z_coords = domain.geometry.x[:, 2]
    z_min = domain.comm.allreduce(float(z_coords.min()))
    z_split = z_min + t_ox
    def _eps_callable(X):
        z = X[2]
        return np.where(z <= z_split, float(epsr_top), float(epsr_bulk))
    eps.interpolate(_eps_callable)
    return eps
=======
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
>>>>>>> 7b176cafc436c4f4f7c51f1364ef42c02d769d99
