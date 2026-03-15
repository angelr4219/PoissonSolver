from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh

from .common import Phys, make_function_space


EPS0 = 8.8541878128e-12


def make_eps_cellwise_from_ct(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    phys: Phys,
    eps_r0: float,
    eps_r1: Optional[float],
) -> fem.Function:
    eps = fem.Function(make_function_space(msh, ("DG", 0)), name="eps_abs")
    eps.x.array[:] = EPS0 * float(eps_r0)

    if eps_r1 is not None:
        cells1 = ct.find(phys.VOL1)
        if cells1.size > 0:
            eps.x.array[cells1] = EPS0 * float(eps_r1)

    return eps


def build_piecewise_z_epsilon(
    msh: dmesh.Mesh,
    tox: float,
    eps_ox: float,
    eps_si: float,
    name: str = "eps_abs",
) -> fem.Function:
    Q = make_function_space(msh, ("DG", 0))
    eps = fem.Function(Q, name=name)

    def eps_interp(x):
        z = x[2]
        if tox > 0:
            return np.where(z < tox, EPS0 * eps_ox, EPS0 * eps_si)
        return (EPS0 * eps_si) * np.ones_like(z)

    eps.interpolate(eps_interp)
    return eps


def build_layered_z_epsilon(
    msh: dmesh.Mesh,
    layers: Sequence[Tuple[float, float, float]],
    default_eps_r: Optional[float] = None,
    name: str = "epsilon",
) -> fem.Function:
    """
    layers: sequence of (z_min, z_max, eps_r)
    """
    Q0 = make_function_space(msh, ("DG", 0))
    eps = fem.Function(Q0, name=name)

    if default_eps_r is None:
        eps.x.array[:] = 0.0
    else:
        eps.x.array[:] = EPS0 * float(default_eps_r)

    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    zc = x[:, 2]

    for z_min, z_max, eps_r in layers:
        mask = (zc >= float(z_min)) & (zc < float(z_max))
        eps.x.array[mask] = EPS0 * float(eps_r)

    return eps


def build_region_id_from_layers(
    msh: dmesh.Mesh,
    layers: Sequence[Tuple[float, float, float]],
    default_value: float = 0.0,
    name: str = "region_id",
) -> fem.Function:
    Q0 = make_function_space(msh, ("DG", 0))
    region_id = fem.Function(Q0, name=name)
    region_id.x.array[:] = float(default_value)

    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    zc = x[:, 2]

    for i, (z_min, z_max, _eps_r) in enumerate(layers, start=1):
        mask = (zc >= float(z_min)) & (zc < float(z_max))
        region_id.x.array[mask] = float(i)

    return region_id


def build_disk_box_fields(
    msh: dmesh.Mesh,
    R: float,
    t: float,
    Q: float,
    z0: float,
    epsr_bulk: float,
):
    """
    Backward-compatible one-disk builder.
    """
    return build_multi_disk_box_fields(
        msh=msh,
        disks=[(0.0, 0.0, z0, R, t, Q)],
        epsr_bulk=epsr_bulk,
        layers=None,
    )


def build_disk_box_fields_layered(
    msh: dmesh.Mesh,
    R: float,
    t: float,
    Q: float,
    z0: float,
    layers: Sequence[Tuple[float, float, float]],
    default_eps_r: Optional[float] = None,
):
    """
    Backward-compatible one-disk layered builder.
    """
    return build_multi_disk_box_fields(
        msh=msh,
        disks=[(0.0, 0.0, z0, R, t, Q)],
        epsr_bulk=default_eps_r,
        layers=layers,
    )


def build_multi_disk_box_fields(
    msh: dmesh.Mesh,
    disks: Sequence[Tuple[float, float, float, float, float, float]],
    epsr_bulk: Optional[float],
    layers: Optional[Sequence[Tuple[float, float, float]]] = None,
):
    """
    disks: sequence of (xc, yc, zc, R, t, Q_total)

    Returns
    -------
    rho, epsilon, region_id, inside_any, rho_total_assigned
    """
    Q0 = make_function_space(msh, ("DG", 0))

    rho = fem.Function(Q0, name="rho")
    rho.x.array[:] = 0.0

    if layers:
        epsilon = build_layered_z_epsilon(
            msh, layers=layers, default_eps_r=epsr_bulk, name="epsilon"
        )
        region_id = build_region_id_from_layers(
            msh, layers=layers, default_value=(1.0 if epsr_bulk is not None else 0.0)
        )
    else:
        epsilon = fem.Function(Q0, name="epsilon")
        epsilon.x.array[:] = EPS0 * float(epsr_bulk if epsr_bulk is not None else 0.0)

        region_id = fem.Function(Q0, name="region_id")
        region_id.x.array[:] = 1.0 if epsr_bulk is not None else 0.0

    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    inside_any = np.zeros(x.shape[0], dtype=bool)
    rho_total_assigned = 0.0

    for (xc, yc, zc, R, t, Q_total) in disks:
        r2 = (x[:, 0] - xc) ** 2 + (x[:, 1] - yc) ** 2
        inside = (r2 <= R * R) & (np.abs(x[:, 2] - zc) <= 0.5 * t)

        rho_disk = Q_total / (math.pi * R * R * t)
        rho.x.array[inside] += rho_disk
        inside_any |= inside
        rho_total_assigned += rho_disk

    return rho, epsilon, region_id, inside_any, rho_total_assigned


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
    Q0 = make_function_space(msh, ("DG", 0))
    eps = fem.Function(Q0, name=name)
    eps.x.array[:] = 0.0

    local_tags = ct.values
    for tag, epsr in epsr_by_tag.items():
        eps.x.array[local_tags == tag] = EPS0 * float(epsr)

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
