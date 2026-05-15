from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh

from .common import COMM, RANK, make_function_space

EPS0 = 8.8541878128e-12


# ---------------------------------------------------------------------------
# Square gate stack: piecewise z-dependent permittivity
# ---------------------------------------------------------------------------

def build_piecewise_z_epsilon(domain, tox: float, eps_ox: float, eps_si: float) -> fem.Function:
    """
    Build a DG0 epsilon field for a Si/SiO2 box.

    The box is assumed to run from z=0 (top surface, gate side) to z=H (bottom).
    Oxide occupies z in [0, tox]; Si occupies z in (tox, H].
    If tox <= 0 the entire domain is silicon.
    """
    Q0 = make_function_space(domain, ("DG", 0))
    eps = fem.Function(Q0, name="epsilon")
    x_centers = Q0.tabulate_dof_coordinates().reshape((-1, 3))

    eps.x.array[:] = EPS0 * float(eps_si)
    if tox > 0.0:
        oxide_mask = x_centers[:, 2] <= float(tox)
        eps.x.array[oxide_mask] = EPS0 * float(eps_ox)

    return eps


# ---------------------------------------------------------------------------
# Top-disk case: cell-tag based permittivity
# ---------------------------------------------------------------------------

def make_eps_cellwise_from_ct(
    msh: dmesh.Mesh,
    ct: dmesh.MeshTags,
    phys,
    eps_r0: float,
    eps_r1: Optional[float],
) -> fem.Function:
    """
    Build DG0 epsilon from cell tags for the topdisk3d case.

    VOL0 cells get eps_r0, VOL1 cells (if eps_r1 is not None) get eps_r1.
    """
    Q0 = make_function_space(msh, ("DG", 0))
    eps = fem.Function(Q0, name="eps_abs")
    eps.x.array[:] = EPS0 * float(eps_r0)
    if eps_r1 is not None:
        cells1 = ct.find(phys.VOL1)
        if cells1.size > 0:
            eps.x.array[cells1] = EPS0 * float(eps_r1)
    return eps


# ---------------------------------------------------------------------------
# Rod-bore case: tag-dict based permittivity and region fields
# ---------------------------------------------------------------------------

def build_eps_from_cell_tags(
    domain: dmesh.Mesh,
    ct: dmesh.MeshTags,
    epsr_by_tag: dict,
    name: str = "eps_abs",
) -> fem.Function:
    """
    Build DG0 epsilon from a mapping of cell-tag → relative permittivity.

    Cells not in any tag entry keep the default value (vacuum, EPS0).
    """
    Q0 = make_function_space(domain, ("DG", 0))
    eps = fem.Function(Q0, name=name)
    eps.x.array[:] = EPS0  # vacuum fallback
    for tag, epsr in epsr_by_tag.items():
        cells = ct.find(int(tag))
        if cells.size > 0:
            eps.x.array[cells] = EPS0 * float(epsr)
    return eps


def build_region_id_from_cell_tags(
    domain: dmesh.Mesh,
    ct: dmesh.MeshTags,
    name: str = "region_id",
) -> fem.Function:
    """Build a DG0 region_id field directly from cell tag values."""
    Q0 = make_function_space(domain, ("DG", 0))
    region_id = fem.Function(Q0, name=name)
    region_id.x.array[:] = 0.0
    region_id.x.array[ct.indices] = ct.values.astype(float)
    return region_id


def tag_rod_layers(
    domain: dmesh.Mesh,
    ct_raw: dmesh.MeshTags,
    Lz: float,
    t_si_bot: float,
    t_sige: float,
    t_si_top: float,
    air_tag: int = 101,
) -> dmesh.MeshTags:
    """
    Re-tag all cells with layer assignments:
      air_tag : air bore (already in ct_raw)
      201     : Si bottom slab
      202     : SiGe middle slab
      203     : Si top slab

    Uses DG0 centroids to determine z-position of each cell.
    """
    from dolfinx.mesh import meshtags

    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local

    Q0 = make_function_space(domain, ("DG", 0))
    cell_z = Q0.tabulate_dof_coordinates().reshape((-1, 3))[:, 2]

    all_cells = np.arange(num_cells, dtype=np.int32)
    new_tags = np.zeros(num_cells, dtype=np.int32)

    # Preserve air tag from gmsh
    air_cells = ct_raw.find(air_tag)
    new_tags[air_cells] = air_tag

    z1 = float(t_si_bot)
    z2 = float(t_si_bot) + float(t_sige)

    solid = new_tags == 0
    new_tags[solid & (cell_z < z1)] = 201
    new_tags[solid & (cell_z >= z1) & (cell_z < z2)] = 202
    new_tags[solid & (cell_z >= z2)] = 203

    return meshtags(domain, tdim, all_cells, new_tags)


def gaussian_charge_expr(domain, q_value: float, center: tuple, sigma: float):
    """
    Return a UFL expression for a 3D Gaussian charge distribution:

        rho(x) = q * (2 pi sigma^2)^{-3/2} * exp(-|x - center|^2 / (2 sigma^2))

    Suitable for use as rhs_expr in solve_scalar_problem.
    """
    x = ufl.SpatialCoordinate(domain)
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r2 = (x[0] - cx)**2 + (x[1] - cy)**2 + (x[2] - cz)**2
    norm = float(q_value) * (2.0 * np.pi * float(sigma)**2) ** (-1.5)
    q_const = fem.Constant(domain, PETSc.ScalarType(norm))
    return q_const * ufl.exp(-r2 / (2.0 * float(sigma)**2))


# ---------------------------------------------------------------------------
# Charged disk box: multi-disk rho + layered or uniform epsilon
# ---------------------------------------------------------------------------

def build_multi_disk_box_fields(
    msh: dmesh.Mesh,
    disks: List[Tuple[float, float, float, float, float, float]],
    epsr_bulk: float,
    layers: Optional[List[Tuple[float, float, float]]] = None,
) -> Tuple[fem.Function, fem.Function, fem.Function, np.ndarray, float]:
    """
    Build DG0 rho, epsilon, region_id fields for a box containing one or more
    charged disk volumes.

    Parameters
    ----------
    disks : list of (xc, yc, zc, R, t, Q)
    epsr_bulk : fallback/uniform relative permittivity
    layers : optional list of (zmin, zmax, epsr) for layered dielectric

    Returns
    -------
    rho, epsilon, region_id, inside_mask, rho_total_assigned
    """
    Q0 = make_function_space(msh, ("DG", 0))
    x_centers = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    n_local = x_centers.shape[0]

    # --- epsilon ---
    epsilon = fem.Function(Q0, name="epsilon")
    region_id = fem.Function(Q0, name="region_id")

    if layers:
        epsilon.x.array[:] = EPS0 * float(epsr_bulk)
        region_id.x.array[:] = 0.0
        for i, (zmin, zmax, epsr) in enumerate(layers, start=1):
            mask = (x_centers[:, 2] >= float(zmin)) & (x_centers[:, 2] < float(zmax))
            epsilon.x.array[mask] = EPS0 * float(epsr)
            region_id.x.array[mask] = float(i)
    else:
        epsilon.x.array[:] = EPS0 * float(epsr_bulk)
        region_id.x.array[:] = 1.0

    # --- rho ---
    rho = fem.Function(Q0, name="rho")
    rho.x.array[:] = 0.0

    inside = np.zeros(n_local, dtype=bool)

    for xc, yc, zc, R, t, Q in disks:
        r2_xy = (x_centers[:, 0] - xc)**2 + (x_centers[:, 1] - yc)**2
        disk_mask = (r2_xy <= R * R) & (np.abs(x_centers[:, 2] - zc) <= 0.5 * t)

        n_marked_local = int(np.count_nonzero(disk_mask))
        n_marked_global = COMM.allreduce(n_marked_local, op=MPI.SUM)

        if n_marked_global == 0:
            if RANK == 0:
                print(
                    f"[WARN] Disk at ({xc:.3e},{yc:.3e},{zc:.3e}) R={R:.3e} t={t:.3e}: "
                    f"no cells marked — increase n_diam or check geometry."
                )
            continue

        # Discrete volume via DG0 integration
        marker = fem.Function(Q0)
        marker.x.array[:] = 0.0
        marker.x.array[disk_mask] = 1.0
        vol_local = float(fem.assemble_scalar(fem.form(marker * ufl.dx)))
        vol_global = COMM.allreduce(vol_local, op=MPI.SUM)

        if vol_global <= 0.0:
            continue

        rho.x.array[disk_mask] = float(Q) / vol_global
        inside |= disk_mask

    # Total assigned charge
    rho_total_local = float(fem.assemble_scalar(fem.form(rho * ufl.dx)))
    rho_total_assigned = COMM.allreduce(rho_total_local, op=MPI.SUM)

    return rho, epsilon, region_id, inside, rho_total_assigned
