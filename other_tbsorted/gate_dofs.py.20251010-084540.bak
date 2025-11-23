from petsc4py import PETSc
from dolfinx import mesh, fem
import numpy as np

def gate_dofs(V, a, xs, Vs, ztol=1e-12, rect_tol=1e-12):
    """
    Apply Dirichlet BCs on the top surface (z=0):
      - Vi on square gates centered at (xi, 0) of size 2a x 2a
      - 0 V on the rest of the top
    Uses facet -> dof mapping (robust across dolfinx versions).

    Args:
        V: function space
        a: half-size of gate (meters)
        xs, Vs: arrays of gate x-centers and voltages
        ztol: tolerance to detect z=0 facets
        rect_tol: tolerance for gate rectangle membership
    """
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 1) collect all boundary facets with z ~ 0 (the top)
    top_facets = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: np.isclose(x[2], 0.0, atol=ztol)
    )

    # 2) map those facets to DOFs
    dofs_top = fem.locate_dofs_topological(V, fdim, top_facets)

    # 3) coordinates of those DOFs
    X_all = V.tabulate_dof_coordinates().reshape((-1, 3))
    X_top = X_all[dofs_top]

    # 4) split DOFs: gate squares vs remainder
    used = np.zeros(len(dofs_top), dtype=bool)
    bcs = []

    for xi, Vi in zip(xs, Vs):
        in_rect = (
            (X_top[:, 0] >= (xi - a) - rect_tol) & (X_top[:, 0] <= (xi + a) + rect_tol) &
            (X_top[:, 1] >= -a - rect_tol)      & (X_top[:, 1] <=  a + rect_tol)
        )
        ii = np.where(in_rect)[0]
        if ii.size:
            dofs_i = dofs_top[ii]
            bcs.append(fem.dirichletbc(PETSc.ScalarType(Vi), dofs_i, V))
            used[ii] = True

    # remainder of the top is ground (0 V)
    ii0 = np.where(~used)[0]
    if ii0.size:
        dofs0 = dofs_top[ii0]
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs0, V))

    # 5) sanity check (this should be large — hundreds/thousands)
    try:
        ncon = sum(bc.dof_indices().size for bc in bcs)
    except AttributeError:
        ncon = sum(len(bc.dof_indices()) for bc in bcs)
    if ncon == 0 and domain.comm.rank == 0:
        raise RuntimeError("No top-surface DOFs were constrained; check ztol/rect_tol.")

    return bcs

