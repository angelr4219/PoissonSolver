from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import fem


def g_uvz(u, v, z):
    denom = z * np.sqrt(u * u + v * v + z * z)
    return (1.0 / (2.0 * np.pi)) * np.arctan2(u * v, denom)


def phi_square_gates_row(x, y, z, a, xs, Vs):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    out = np.zeros_like(x, dtype=float)
    for xi, Vi in zip(xs, Vs):
        out -= Vi * (
            g_uvz(a - xi + x, a + y, z)
            + g_uvz(a - xi + x, a - y, z)
            + g_uvz(a + xi - x, a + y, z)
            + g_uvz(a + xi - x, a - y, z)
        )
    return out


def eval_function_on_points(u: fem.Function, points: np.ndarray, comm: MPI.Comm):
    """
    Evaluate a scalar Function at physical points.

    Parameters
    ----------
    u : fem.Function
        Scalar FEM function.
    points : ndarray, shape (N, 3)
        Physical coordinates.
    comm : MPI.Comm
        Communicator.

    Returns
    -------
    On rank 0: ndarray shape (N,)
    On other ranks: None
    """
    from dolfinx import geometry

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    msh = u.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, points)
    cells = geometry.compute_colliding_cells(msh, cand, points)

    vals_local = np.full(points.shape[0], np.nan, dtype=np.float64)

    for i in range(points.shape[0]):
        cell_candidates = cells.links(i)
        if len(cell_candidates) == 0:
            continue

        cell = int(cell_candidates[0])

        # This DOLFINx build expects coordinates with length 3.
        p = np.asarray(points[i], dtype=np.float64)  # shape (3,)

        value = u.eval(p, np.array([cell], dtype=np.int32))

        # Handle scalar return robustly across builds
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.size > 0:
            vals_local[i] = value[0]

    gathered = comm.gather(vals_local, root=0)

    if comm.rank != 0:
        return None

    gathered = np.stack(gathered, axis=0)
    out = np.full(points.shape[0], np.nan, dtype=np.float64)

    for j in range(points.shape[0]):
        col = gathered[:, j]
        finite = col[np.isfinite(col)]
        if finite.size > 0:
            out[j] = finite[0]

    return out
