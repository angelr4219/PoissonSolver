# --- error_fields_to_xdmf.py (inline snippet) -------------------------------
import os
import numpy as np
import ufl
from dolfinx import fem, io

def _project_to(V, expr_ufl):
    """L2-project UFL expression onto FE space V."""
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    a = ufl.inner(w, v) * ufl.dx
    L = ufl.inner(expr_ufl, v) * ufl.dx
    prob = fem.petsc.LinearProblem(fem.form(a), fem.form(L),
                                   u=u,
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    return prob.solve()

def _interpolate_cellwise_max(V_dg0, V_src, arr_src):
    """
    Compute cell-wise (DG0) max of |arr_src| over the dofs that belong to each cell.
    Works well on Lagrange spaces; it’s a fast diagnostic for hotspots.
    """
    mesh = V_src.mesh
    # map dofs -> cells
    cmap = V_src.dofmap.list.cell_dofs
    max_per_cell = np.zeros(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=float)
    for c in range(max_per_cell.size):
        dofs = cmap(c)
        if len(dofs) > 0:
            max_per_cell[c] = np.max(np.abs(arr_src[dofs]))
    f = fem.Function(V_dg0)
    f.x.array[:] = max_per_cell
    return f

def write_error_fields_xdmf(mesh, V, uh, phi_exact_ufl, outpath, tau=1e-8):
    """
    Writes: abs_error (CGp), rel_error (CGp), cell_max_abs_error (DG0), phi_exact (CGp)
    into one XDMF for ParaView. 'tau' is the relative floor (Volts) for rel_error.
    """
    comm = mesh.comm
    rank = comm.rank

    # 1) Put analytic phi on the same space as uh (apples-to-apples)
    u_ex = _project_to(V, phi_exact_ufl)

    # 2) Nodal arrays
    uh_arr   = uh.x.array
    uex_arr  = u_ex.x.array
    abs_err  = np.abs(uh_arr - uex_arr)

    # Relative nodal error with floor
    denom = np.maximum(np.abs(uex_arr), tau)
    rel_err = abs_err / denom

    # 3) Create FE functions for abs/rel errors
    abs_err_f = fem.Function(V, name="abs_error")
    rel_err_f = fem.Function(V, name="rel_error")
    abs_err_f.x.array[:] = abs_err
    rel_err_f.x.array[:] = rel_err

    # 4) Cell-wise (DG0) max of |error| for crisp hotspot viewing
    V0 = fem.FunctionSpace(mesh, ("DG", 0))
    cell_max_abs = _interpolate_cellwise_max(V0, V, abs_err_f.x.array)
    cell_max_abs.name = "cell_max_abs_error"

    # 5) Write everything
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with io.XDMFFile(comm, outpath, "w") as xdmf:
        xdmf.write_mesh(mesh)
        # optional: write the analytic field too (nice for side-by-side coloring)
        u_ex.name = "phi_exact"
        xdmf.write_function(u_ex)
        xdmf.write_function(uh)          # your numerical solution, if not yet written
        xdmf.write_function(abs_err_f)
        xdmf.write_function(rel_err_f)
        xdmf.write_function(cell_max_abs)

    # 6) Print quick stats
    if rank == 0:
        k_abs = int(np.argmax(abs_err)) if abs_err.size else -1
        k_rel = int(np.argmax(rel_err)) if rel_err.size else -1
        print(f"[ERROR-FIELDS] wrote {outpath}")
        print(f"[ERROR-FIELDS] max|e| = {abs_err.max():.3e} at dof {k_abs}")
        print(f"[ERROR-FIELDS] max rel = {rel_err.max():.3e} (tau={tau:.3e}) at dof {k_rel}")
# ---------------------------------------------------------------------------

