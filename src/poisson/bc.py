from __future__ import annotations
import numpy as np
from petsc4py import PETSc
from dolfinx import mesh as dmesh, fem

def gate_top_and_bottom_bcs(V, a, xs, Vs, rect_tol=1e-8, ztol=1e-9):
    domain = V.mesh
    tdim = domain.topology.dim; fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0)

    Z = domain.geometry.x[:, 2]
    z_min = domain.comm.allreduce(float(Z.min()))
    z_max = domain.comm.allreduce(float(Z.max()))

    top_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_min, atol=ztol))
    dofs_top = np.unique(fem.locate_dofs_topological(V, fdim, top_facets))
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    Xtop = X[dofs_top]

    used = np.zeros(dofs_top.shape[0], dtype=bool)
    bcs = []
    for xi, Vi in zip(xs, Vs):
        in_rect = (
            (Xtop[:, 0] >= (xi - a) - rect_tol) & (Xtop[:, 0] <= (xi + a) + rect_tol) &
            (Xtop[:, 1] >= -a - rect_tol)      & (Xtop[:, 1] <=  a + rect_tol)
        )
        idx = np.where(in_rect)[0]
        if idx.size:
            bcs.append(fem.dirichletbc(PETSc.ScalarType(Vi), dofs_top[idx], V))
            used[idx] = True

    idx0 = np.where(~used)[0]
    if idx0.size:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs_top[idx0], V))

    bot_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_max, atol=ztol))
    dofs_bot = np.unique(fem.locate_dofs_topological(V, fdim, bot_facets))
    if dofs_bot.size:
        bcs.append(fem.dirichletbc(PETSC_ZERO, dofs_bot, V))
    return bcs

# PETSc zero scalar helper
PETSC_ZERO = PETSc.ScalarType(0.0)

# ---- MPC periodic hook (stub; fill later) ----
def build_bcs_and_mpc(V, mesh, bc_cfg: dict):
    """
    bc_cfg.type in {"dirichlet", "periodic-x"}
    Currently returns (bcs, mpc=None) for Dirichlet.
    """
    if bc_cfg.get("type", "dirichlet") == "dirichlet":
        args = bc_cfg["dirichlet_args"]  # expect {a, xs, Vs}
        bcs = gate_top_and_bottom_bcs(V, **args)
        return bcs, None
    elif bc_cfg["type"] == "periodic-x":
        raise NotImplementedError("MPC periodic-x not wired yet.")
    else:
        raise ValueError(f"Unknown BC type: {bc_cfg['type']}")
