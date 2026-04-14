from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh

from .common import COMM, RANK, make_function_space

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19


@dataclass(frozen=True)
class BoxSpec:
    Lx: float
    Ly: float
    Lz: float
    h: float


@dataclass(frozen=True)
class SphereSpec:
    xc: float
    yc: float
    zc: float
    R: float


@dataclass(frozen=True)
class DiskSpec:
    xc: float
    yc: float
    zc: float
    R: float
    t: float


@dataclass(frozen=True)
class CubeSpec:
    xc: float
    yc: float
    zc: float
    a: float


def build_centered_box_mesh(comm: MPI.Comm, box: BoxSpec):
    nx = max(2, int(math.ceil(box.Lx / box.h)))
    ny = max(2, int(math.ceil(box.Ly / box.h)))
    nz = max(2, int(math.ceil(box.Lz / box.h)))

    x0 = np.array([-0.5 * box.Lx, -0.5 * box.Ly, -0.5 * box.Lz], dtype=np.float64)
    x1 = np.array([+0.5 * box.Lx, +0.5 * box.Ly, +0.5 * box.Lz], dtype=np.float64)

    msh = dmesh.create_box(
        comm,
        [x0, x1],
        [nx, ny, nz],
        cell_type=dmesh.CellType.tetrahedron,
    )

    if RANK == 0:
        print(f"[mesh] nx,ny,nz = {nx}, {ny}, {nz}")

    return msh, nx, ny, nz


def _dg0_cell_centers(msh):
    Q0 = make_function_space(msh, ("DG", 0))
    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    return Q0, x


def _global_scalar(val_local: float, comm: MPI.Comm) -> float:
    return comm.allreduce(float(val_local), op=MPI.SUM)


def _mask_sphere(x: np.ndarray, spec: SphereSpec) -> np.ndarray:
    r2 = (
        (x[:, 0] - spec.xc) ** 2
        + (x[:, 1] - spec.yc) ** 2
        + (x[:, 2] - spec.zc) ** 2
    )
    return r2 <= spec.R * spec.R


def _mask_disk(x: np.ndarray, spec: DiskSpec) -> np.ndarray:
    r2 = (x[:, 0] - spec.xc) ** 2 + (x[:, 1] - spec.yc) ** 2
    return (r2 <= spec.R * spec.R) & (np.abs(x[:, 2] - spec.zc) <= 0.5 * spec.t)


def _mask_cube(x: np.ndarray, spec: CubeSpec) -> np.ndarray:
    h = 0.5 * spec.a
    return (
        (np.abs(x[:, 0] - spec.xc) <= h)
        & (np.abs(x[:, 1] - spec.yc) <= h)
        & (np.abs(x[:, 2] - spec.zc) <= h)
    )


def _assemble_global_scalar(expr, msh) -> float:
    val_local = fem.assemble_scalar(fem.form(expr))
    return msh.comm.allreduce(val_local, op=MPI.SUM)


def _build_uniform_epsilon(msh, eps_r: float, name: str = "epsilon") -> fem.Function:
    Q0 = make_function_space(msh, ("DG", 0))
    eps = fem.Function(Q0, name=name)
    eps.x.array[:] = EPS0 * float(eps_r)
    return eps


def _build_region_id(msh, inside_mask: np.ndarray, name: str = "region_id") -> fem.Function:
    Q0 = make_function_space(msh, ("DG", 0))
    region_id = fem.Function(Q0, name=name)
    region_id.x.array[:] = 1.0
    region_id.x.array[inside_mask] = 2.0
    return region_id


def _build_rho_from_mask_exact_total_charge(
    msh,
    inside_mask: np.ndarray,
    Q_total: float,
    name: str = "rho",
) -> Tuple[fem.Function, float]:
    """
    Build DG0 rho with total integrated charge exactly Q_total
    on the discrete marked region.
    """
    Q0 = make_function_space(msh, ("DG", 0))

    marker = fem.Function(Q0, name="shape_marker")
    marker.x.array[:] = 0.0
    marker.x.array[inside_mask] = 1.0

    vol_marked = _assemble_global_scalar(marker * ufl.dx, msh)
    if vol_marked <= 0.0:
        raise RuntimeError("Marked shape volume is zero on the mesh. Refine h or enlarge shape.")

    rho_val = float(Q_total) / vol_marked

    rho = fem.Function(Q0, name=name)
    rho.x.array[:] = 0.0
    rho.x.array[inside_mask] = rho_val

    q_assigned = _assemble_global_scalar(rho * ufl.dx, msh)
    return rho, q_assigned


def build_sphere_problem(
    msh,
    spec: SphereSpec,
    eps_r: float,
    Q_total: float,
):
    Q0, x = _dg0_cell_centers(msh)
    inside = _mask_sphere(x, spec)
    rho, q_assigned = _build_rho_from_mask_exact_total_charge(msh, inside, Q_total, name="rho")
    epsilon = _build_uniform_epsilon(msh, eps_r, name="epsilon")
    region_id = _build_region_id(msh, inside, name="region_id")
    return rho, epsilon, region_id, inside, q_assigned


def build_disk_problem(
    msh,
    spec: DiskSpec,
    eps_r: float,
    Q_total: float,
):
    Q0, x = _dg0_cell_centers(msh)
    inside = _mask_disk(x, spec)
    rho, q_assigned = _build_rho_from_mask_exact_total_charge(msh, inside, Q_total, name="rho")
    epsilon = _build_uniform_epsilon(msh, eps_r, name="epsilon")
    region_id = _build_region_id(msh, inside, name="region_id")
    return rho, epsilon, region_id, inside, q_assigned


def build_cube_problem(
    msh,
    spec: CubeSpec,
    eps_r: float,
    Q_total: float,
):
    Q0, x = _dg0_cell_centers(msh)
    inside = _mask_cube(x, spec)
    rho, q_assigned = _build_rho_from_mask_exact_total_charge(msh, inside, Q_total, name="rho")
    epsilon = _build_uniform_epsilon(msh, eps_r, name="epsilon")
    region_id = _build_region_id(msh, inside, name="region_id")
    return rho, epsilon, region_id, inside, q_assigned
