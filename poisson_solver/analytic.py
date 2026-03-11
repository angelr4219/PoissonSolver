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
    from dolfinx import geometry

    msh = u.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, points)
    cells = geometry.compute_colliding_cells(msh, cand, points)

    vals_local = np.full((points.shape[0],), np.nan, dtype=np.float64)
    for i in range(points.shape[0]):
        cell_candidates = cells.links(i)
        if len(cell_candidates) > 0:
            cell = cell_candidates[0]
            v = np.zeros((1,), dtype=np.float64)
            u.eval(v, points[i : i + 1], np.array([cell], dtype=np.int32))
            vals_local[i] = v[0]

    all_vals = comm.gather(vals_local, root=0)
    if comm.rank != 0:
        return None

    all_vals = np.stack(all_vals, axis=0)
    out = np.full((points.shape[0],), np.nan, dtype=np.float64)
    for j in range(points.shape[0]):
        col = all_vals[:, j]
        finite = col[np.isfinite(col)]
        out[j] = finite[0] if finite.size > 0 else np.nan
    return out
