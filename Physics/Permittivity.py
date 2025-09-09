#!/usr/bin/env python3
"""
permittivity.py
Material registry and helpers for permittivity ε in SI units.

- Provides relative permittivities (room-temp, low-frequency reference)
  Si≈11.7, Ge≈16.0, SiO2≈3.9, air≈1.0. Overrideable via CLI or kwargs.
- Builds piecewise-constant ε (DG0) from cell-tags.
- Supports continuous ε(x) by projecting a Python callable into CG1 or DGk.

Units:
  ε = ε0 * εr, with ε0 = 8.8541878128e-12 F/m

"""
from __future__ import annotations
from typing import Callable, Dict, Mapping, Optional
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

EPS0 = 8.8541878128e-12  # F/m

DEFAULT_EPS_R: Dict[str, float] = {
    "air": 1.0,
    "vacuum": 1.0,
    "SiO2": 3.9,
    "Si": 11.7,
    "Ge": 16.0,
}

def eps_from_materials(
    domain: mesh.Mesh,
    cell_tags: Optional[fem.Function] = None,
    tag2name: Optional[Mapping[int, str]] = None,
    overrides: Optional[Mapping[str, float]] = None,
    space: str = "DG", degree: int = 0,
) -> fem.Function:
    """
    Build a finite-element function ε(x) = ε0 * εr(x).

    If cell_tags is provided (as an int-valued MeshFunction/Function over cells),
    use tag2name to map tag -> material name, then lookup εr in registry.
    Otherwise, default to air (1.0).

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    cell_tags : fem.Function[int] or None
    tag2name : dict[int,str]
    overrides : dict[str,float] : override default εr by name
    space : "DG" or "CG"
    degree : FE degree (DG0 recommended for piecewise constants)

    Returns
    -------
    eps_fun : fem.Function  (scalar field)
    """
    V = fem.functionspace(domain, (("Discontinuous Lagrange" if space=="DG" else "Lagrange"), degree))
    eps_fun = fem.Function(V)
    registry = dict(DEFAULT_EPS_R)
    if overrides:
        registry.update(overrides)

    # Fill values cellwise
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    values = np.empty(num_cells, dtype=np.float64)

    if cell_tags is None or tag2name is None:
        values.fill(EPS0 * registry["air"])
    else:
        # cell_tags is a Function on cells with dtype=int
        ct_array = cell_tags.x.array.astype(np.int64)
        for cid in range(num_cells):
            tag = int(ct_array[cid])
            name = tag2name.get(tag, "air")
            er = registry.get(name, 1.0)
            values[cid] = EPS0 * er

    with eps_fun.vector.localForm() as lf:
        lf.set(values)

    return eps_fun


def project_callable(
    domain: mesh.Mesh,
    func: Callable[[np.ndarray], np.ndarray],
    space: str = "CG",
    degree: int = 1,
) -> fem.Function:
    """
    Project a Python callable f(x) -> value into a FE space.

    The callable should accept (3, N) shaped coordinates or (gdim,) vector.
    """
    V = fem.functionspace(domain, (("Lagrange" if space=="CG" else "Discontinuous Lagrange"), degree))
    f_fun = fem.Function(V)

    # Interpolate via ufl.SpatialCoordinate
    x = ufl.SpatialCoordinate(domain)
    # Wrap: evaluate on numpy arrays via fem.Expression
    expr = fem.Expression(lambda X: func(X), V.element.interpolation_points())
    f_fun.interpolate(expr)
    return f_fun
