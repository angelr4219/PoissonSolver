from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem

COMM = MPI.COMM_WORLD
RANK = COMM.rank


@dataclass(frozen=True)
class Phys:
    # 3D volume tags
    VOL0: int = 1
    VOL1: int = 2

    # 3D boundary/facet tags
    TOP: int = 10
    BOTTOM: int = 11
    SIDES: int = 12
    TOPDISK: int = 20

    # 2D placeholder tags
    DISK_REGION: int = 1
    DISK_BND: int = 10


def make_function_space(msh, spec):
    try:
        return fem.functionspace(msh, spec)
    except AttributeError:
        return fem.FunctionSpace(msh, spec)


def degree_compat(deg_or_callable) -> int:
    try:
        return int(deg_or_callable())
    except TypeError:
        return int(deg_or_callable)


def global_minmax(phi: fem.Function) -> Tuple[float, float]:
    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)
    return gmin, gmax
