#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict
from dolfinx import fem
import ufl

def eps_constant_DG0(domain, eps_r: float):
    """DG0 (cellwise constant) relative permittivity for assembly."""
    Ve = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(Ve); eps.name = "eps"
    eps.x.array[:] = eps_r
    return eps

def eps_from_cell_tags(domain, cell_tags, mapping: Dict[int, float]):
    """
    Build DG0 eps from cell tag -> eps_r mapping.
    Any unlisted tag keeps the last mapping value.
    """
    Ve = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(Ve); eps.name = "eps"
    default = list(mapping.values())[-1] if mapping else 1.0
    eps.x.array[:] = default
    if cell_tags is not None:
        for tag, val in mapping.items():
            cells = cell_tags.find(tag)
            if cells.size:
                eps.x.array[cells] = val
    return eps

def eps_to_cg1_for_viz(domain, eps_dg):
    """Make a CG1 copy for ParaView visualization."""
    Vcg = fem.functionspace(domain, ("Lagrange", 1))
    eps_cg = fem.Function(Vcg); eps_cg.name = eps_dg.name
    from dolfinx.fem import create_expression
    eps_cg.interpolate(create_expression(eps_dg, Vcg.element.interpolation_points()))
    return eps_cg

