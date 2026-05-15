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
    Create periodic MPC constraints on a scalar CG space.

    Supported directions: "x", "y", "xy"
    """
    try:
        import dolfinx_mpc
    except ImportError as e:
        raise RuntimeError(
            "Periodic boundary mode requested, but dolfinx_mpc is not installed "
            "in the container/environment."
        ) from e

    if direction not in ("x", "y", "xy"):
        raise ValueError(
            f"Unsupported periodic direction '{direction}'. Use 'x', 'y', or 'xy'."
        )

    gmin, gmax = _global_bounding_box(V.mesh)
    gdim = len(gmin)

    if gdim < 2 and direction in ("y", "xy"):
        raise RuntimeError(f"Mesh geometric dimension {gdim} is incompatible with y-periodicity.")

    Lx = float(gmax[0] - gmin[0])
    Ly = float(gmax[1] - gmin[1]) if gdim > 1 else 0.0

    if direction in ("x", "xy") and (not np.isfinite(Lx) or Lx <= 0.0):
        raise RuntimeError(f"Invalid x-period from bounding box: min={gmin[0]}, max={gmax[0]}")
    if direction in ("y", "xy") and (not np.isfinite(Ly) or Ly <= 0.0):
        raise RuntimeError(f"Invalid y-period from bounding box: min={gmin[1]}, max={gmax[1]}")

    x_slave = float(gmax[0])
    y_slave = float(gmax[1]) if gdim > 1 else 0.0

    if tol is None:
        scale_ref = max(1.0, abs(x_slave), abs(y_slave), abs(Lx), abs(Ly))
        tol = 1.0e-12 * scale_ref

    if RANK == 0:
        print(
            f"[periodic] direction={direction}  "
            f"x_slave={x_slave:.6e}  Lx={Lx:.6e}  "
            f"y_slave={y_slave:.6e}  Ly={Ly:.6e}  tol={tol:.3e}"
        )

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    bcs_list = [] if bcs is None else bcs

    def relation_x(x):
        out = np.copy(x)
        out[0] = x[0] - Lx
        return out

    def relation_y(x):
        out = np.copy(x)
        out[1] = x[1] - Ly
        return out

    def relation_xy(x):
        out = np.copy(x)
        out[0] = x[0] - Lx
        out[1] = x[1] - Ly
        return out

    if direction == "x":
        def indicator_x(x):
            return np.isclose(x[0], x_slave, atol=tol)

        mpc.create_periodic_constraint_geometrical(
            V, indicator_x, relation_x, bcs_list, scale=scale, tol=tol
        )

    elif direction == "y":
        def indicator_y(x):
            return np.isclose(x[1], y_slave, atol=tol)

        mpc.create_periodic_constraint_geometrical(
            V, indicator_y, relation_y, bcs_list, scale=scale, tol=tol
        )

    else:
        # "xy": handle corners separately to avoid slave-of-slave conflicts
        def indicator_x_only(x):
            return np.logical_and(
                np.isclose(x[0], x_slave, atol=tol),
                np.logical_not(np.isclose(x[1], y_slave, atol=tol)),
            )

        def indicator_y_only(x):
            return np.logical_and(
                np.isclose(x[1], y_slave, atol=tol),
                np.logical_not(np.isclose(x[0], x_slave, atol=tol)),
            )

        def indicator_xy_edge(x):
            return np.logical_and(
                np.isclose(x[0], x_slave, atol=tol),
                np.isclose(x[1], y_slave, atol=tol),
            )

        mpc.create_periodic_constraint_geometrical(
            V, indicator_x_only, relation_x, bcs_list, scale=scale, tol=tol
        )
        mpc.create_periodic_constraint_geometrical(
            V, indicator_y_only, relation_y, bcs_list, scale=scale, tol=tol
        )
        mpc.create_periodic_constraint_geometrical(
            V, indicator_xy_edge, relation_xy, bcs_list, scale=scale, tol=tol
        )

    mpc.finalize()
    return mpc
