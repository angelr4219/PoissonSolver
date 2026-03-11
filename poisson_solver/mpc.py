from __future__ import annotations

import numpy as np
from mpi4py import MPI

from .common import RANK


def _global_bounding_box(msh):
    x = msh.geometry.x
    gdim = msh.geometry.dim

    if x.shape[0] > 0:
        local_min = np.min(x[:, :gdim], axis=0)
        local_max = np.max(x[:, :gdim], axis=0)
    else:
        local_min = np.full((gdim,), np.inf, dtype=np.float64)
        local_max = np.full((gdim,), -np.inf, dtype=np.float64)

    comm = msh.comm
    global_min = np.array(
        [comm.allreduce(float(v), op=MPI.MIN) for v in local_min], dtype=np.float64
    )
    global_max = np.array(
        [comm.allreduce(float(v), op=MPI.MAX) for v in local_max], dtype=np.float64
    )
    return global_min, global_max


def create_periodic_mpc(V, direction: str, bcs=None, scale=1.0, tol=None):
    """
    Create a one-direction periodic MPC on a scalar CG space.

    Supported directions:
      - "x"
      - "y"

    This deliberately does not try to do simultaneous x+y periodicity yet.
    Corner handling deserves its own verified step.
    """
    try:
        import dolfinx_mpc
    except ImportError as e:
        raise RuntimeError(
            "Periodic boundary mode requested, but dolfinx_mpc is not installed "
            "in the container/environment."
        ) from e

    if direction not in ("x", "y"):
        raise ValueError(f"Unsupported periodic direction '{direction}'. Use 'x' or 'y'.")

    axis = 0 if direction == "x" else 1
    gmin, gmax = _global_bounding_box(V.mesh)

    if axis >= len(gmin):
        raise RuntimeError(
            f"Cannot apply periodicity in direction '{direction}' for a mesh with "
            f"geometric dimension {len(gmin)}."
        )

    slave_value = float(gmax[axis])
    period = float(gmax[axis] - gmin[axis])

    if not np.isfinite(period) or period <= 0.0:
        raise RuntimeError(
            f"Invalid bounding box for periodic direction '{direction}': "
            f"min={gmin[axis]}, max={gmax[axis]}"
        )

    if tol is None:
        tol = 1.0e-12 * max(1.0, abs(slave_value), abs(period))

    if RANK == 0:
        print(
            f"[periodic] direction={direction}  "
            f"slave={slave_value:.6e}  period={period:.6e}  tol={tol:.3e}"
        )

    def indicator(x):
        return np.isclose(x[axis], slave_value, atol=tol)

    def relation(x):
        out = np.copy(x)
        out[axis] = x[axis] - period
        return out

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(
        V,
        indicator,
        relation,
        [] if bcs is None else bcs,
        scale=scale,
        tol=tol,
    )
    mpc.finalize()
    return mpc
