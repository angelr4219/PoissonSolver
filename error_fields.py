import os, numpy as np, ufl
from dolfinx import fem, io

def _project_to(V, expr_ufl):
    u = fem.Function(V); v = ufl.TestFunction(V); w = ufl.TrialFunction(V)
    a = ufl.inner(w,v)*ufl.dx; L = ufl.inner(expr_ufl,v)*ufl.dx
    prob = fem.petsc.LinearProblem(fem.form(a), fem.form(L), u=u,
                                   petsc_options={"ksp_type":"preonly","pc_type":"lu"})
    return prob.solve()

def _cellwise_max_abs(V_dg0, V_src, nodal_vals):
    mesh = V_src.mesh
    cmap = V_src.dofmap.list.cell_dofs
    ncell = mesh.topology.index_map(mesh.topology.dim).size_local
    out = np.zeros(ncell)
    for c in range(ncell):
        dofs = cmap(c)
        if len(dofs): out[c] = np.max(np.abs(nodal_vals[dofs]))
    f = fem.Function(V_dg0); f.x.array[:] = out; return f

def write_error_fields_xdmf(mesh, V, uh, phi_exact_ufl, outpath, tau=1e-9):
    u_ex = _project_to(V, phi_exact_ufl)
    abs_err = np.abs(uh.x.array - u_ex.x.array)
    rel_err = abs_err / np.maximum(np.abs(u_ex.x.array), tau)
    abs_err_f = fem.Function(V, name="abs_error"); abs_err_f.x.array[:] = abs_err
    rel_err_f = fem.Function(V, name="rel_error"); rel_err_f.x.array[:] = rel_err
    V0 = fem.FunctionSpace(mesh, ("DG", 0))
    cell_max = _cellwise_max_abs(V0, V, abs_err_f.x.array); cell_max.name = "cell_max_abs_error"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with io.XDMFFile(mesh.comm, outpath, "w") as xdmf:
        xdmf.write_mesh(mesh)
        u_ex.name = "phi_exact"; xdmf.write_function(u_ex)
        xdmf.write_function(uh)
        xdmf.write_function(abs_err_f)
        xdmf.write_function(rel_err_f)
        xdmf.write_function(cell_max)
    if mesh.comm.rank == 0:
        print(f"[ERROR-FIELDS] wrote {outpath}")
        print(f"[ERROR-FIELDS] max|e|={abs_err.max():.3e}, max rel={rel_err.max():.3e} (tau={tau:.1e})")
