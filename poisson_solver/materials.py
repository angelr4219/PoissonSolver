from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh

from .common import Phys, make_function_space


def make_eps_cellwise_from_ct(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    phys: Phys,
    eps_r0: float,
    eps_r1: Optional[float],
) -> fem.Function:
    eps0 = 8.8541878128e-12
    V0 = make_function_space(msh, ("DG", 0))
    eps = fem.Function(V0, name="eps_abs")
    eps.x.array[:] = eps0 * float(eps_r0)

    if eps_r1 is not None:
        cells1 = ct.find(phys.VOL1)
        if cells1.size > 0:
            eps.x.array[cells1] = eps0 * float(eps_r1)

    return eps


def build_piecewise_z_epsilon(
    msh: dmesh.Mesh,
    tox: float,
    eps_ox: float,
    eps_si: float,
    name: str = "eps_abs",
) -> fem.Function:
    eps0 = 8.8541878128e-12
    Q = make_function_space(msh, ("DG", 0))
    eps = fem.Function(Q, name=name)

    def eps_interp(x):
        z = x[2]
        if tox > 0:
            return np.where(z < tox, eps0 * eps_ox, eps0 * eps_si)
        return (eps0 * eps_si) * np.ones_like(z)

    eps.interpolate(eps_interp)
    return eps


def build_disk_box_fields(
    msh: dmesh.Mesh,
    R: float,
    t: float,
    Q: float,
    z0: float,
    epsr_bulk: float,
):
    eps0 = 8.8541878128e-12
    eps_bulk = eps0 * epsr_bulk

    Q0 = make_function_space(msh, ("DG", 0))

    rho = fem.Function(Q0, name="rho")
    rho.x.array[:] = 0.0

    epsilon = fem.Function(Q0, name="epsilon")
    epsilon.x.array[:] = eps_bulk

    region_id = fem.Function(Q0, name="region_id")
    region_id.x.array[:] = 1.0

    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    r2 = x[:, 0] ** 2 + x[:, 1] ** 2
    inside = (r2 <= R * R) & (np.abs(x[:, 2] - z0) <= 0.5 * t)

    rho_disk = Q / (math.pi * R * R * t)
    rho.x.array[inside] = rho_disk

    return rho, epsilon, region_id, inside, rho_disk


def tag_rod_layers(
    domain: dmesh.Mesh,
    ct_raw: dmesh.MeshTags,
    Lz: float,
    t_si_bot: float,
    t_sige: float,
    t_si_top: float,
    air_tag: int = 101,
) -> dmesh.MeshTags:
    from dolfinx.cpp.mesh import entities_to_geometry
    from dolfinx.mesh import meshtags

    num_local_cells = domain.topology.index_map(domain.topology.dim).size_local
    cell_indices = np.arange(num_local_cells, dtype=np.int32)

    cell_tags = np.zeros(num_local_cells, dtype=np.int32)
    raw_vals = ct_raw.values
    cell_tags[raw_vals == air_tag] = air_tag

    x = domain.geometry.x
    cells_as_nodes = entities_to_geometry(domain, domain.topology.dim, cell_indices, False)
    cell_z = np.mean(x[cells_as_nodes, 2], axis=1)

    z1 = t_si_bot
    z2 = t_si_bot + t_sige

    solid_idx = np.where(cell_tags == 0)[0]
    cell_tags[solid_idx[(cell_z[solid_idx] >= 0.0) & (cell_z[solid_idx] < z1)]] = 201
    cell_tags[solid_idx[(cell_z[solid_idx] >= z1) & (cell_z[solid_idx] < z2)]] = 202
    cell_tags[solid_idx[(cell_z[solid_idx] >= z2) & (cell_z[solid_idx] <= Lz)]] = 203

    return meshtags(domain, domain.topology.dim, cell_indices, cell_tags)


def build_eps_from_cell_tags(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    epsr_by_tag: Dict[int, float],
    name: str = "eps_abs",
) -> fem.Function:
    eps0 = 8.8541878128e-12
    Q0 = make_function_space(msh, ("DG", 0))
    eps = fem.Function(Q0, name=name)
    eps.x.array[:] = 0.0

    local_tags = ct.values
    for tag, epsr in epsr_by_tag.items():
        eps.x.array[local_tags == tag] = eps0 * float(epsr)

    return eps


def build_region_id_from_cell_tags(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    name: str = "region_id",
) -> fem.Function:
    Q0 = make_function_space(msh, ("DG", 0))
    region_id = fem.Function(Q0, name=name)
    region_id.x.array[:] = ct.values.astype(np.float64)
    return region_id


def gaussian_charge_expr(
    msh: dmesh.Mesh,
    q_value: float,
    center: Tuple[float, float, float],
    sigma: float,
):
    q = fem.Constant(msh, PETSc.ScalarType(q_value))
    x = ufl.SpatialCoordinate(msh)
    dx = x - ufl.as_vector(center)
    r2 = ufl.inner(dx, dx)
    norm = q * (2.0 * np.pi * sigma**2) ** (-1.5)
    return norm * ufl.exp(-r2 / (2.0 * sigma**2))
