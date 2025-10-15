# err_fields.py  —  write mesh-wise error fields to XDMF
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_vector
from ufl import dx

# --- Assumptions: you already have mesh, V, uh, and either u_ref (Function) or a callable phi_exact(x) ---
# If you’re running this standalone, import your mesh/solution here.
# Example:
# from mysolve import mesh, V, uh, phi_exact  # phi_exact: callable R^3 -> R

comm = MPI.COMM_WORLD

# V is the CG1 space used for uh
V = uh.function_space

# 1) Build/reference u_ref on V (either copy from an existing Function or interpolate from a callable)
if isinstance('u_ref', fem.Function):  # placeholder safeguard if you already have u_ref
    uref = u_ref
else:
    uref = fem.Function(V, name="u_ref")
    # Interpolate your analytic/reference solution:
    # uref.interpolate(lambda x: phi_exact(x))   # <-- uncomment and provide phi_exact

# 2) Nodal error (CG1)
e = fem.Function(V, name="err_node")
with e.vector.localForm() as el, uh.vector.localForm() as uhl, ur.vector.localForm() as url:
    # If you named the reference 'uref' above, keep consistent
    pass
# Simpler: set by array math (works for P1 nodal DOFs)
e.x.array[:] = uh.x.array - uref.x.array

# 3) Robust relative error (CG1) with floor τ
# τ = 1e-6 * (range of |u_ref|); tweak if you prefer
abs_ref = np.abs(uref.x.array)
tau = 1e-6 * (abs_ref.max() - abs_ref.min() + 1e-30)
rel = fem.Function(V, name="err_rel_node")
rel.x.array[:] = np.abs(e.x.array) / np.maximum(abs_ref, tau)

# 4) Cellwise DG0 spaces/fields
W = fem.FunctionSpace(V.mesh, ("DG", 0))
v = ufl.TestFunction(W)

# Helper: assemble cellwise integrals into the DG0 coefficient vector
def cellwise_integral(expr):
    """Return numpy array of size n_cells with ∫_K expr dx for each cell K."""
    form = fem.form(expr * v * dx)
    vec = assemble_vector(form)
    vec.ghostUpdate(addv=fem.petsc.InsertMode.ADD_VALUES,
                    mode=fem.petsc.ScatterMode.REVERSE)
    # Pull local (owned) portion
    return vec.array

# (a) ∫_K e^2 dx  and per-cell “L2”
e2_cell_int = cellwise_integral(ufl.inner(e, e))
err_cell_int = fem.Function(W, name="err_cell_e2_int")   # integral of e^2 on each cell
err_cell_int.x.array[:] = e2_cell_int

err_cell_L2 = fem.Function(W, name="err_cell_L2")        # sqrt(∫_K e^2 dx)
err_cell_L2.x.array[:] = np.sqrt(np.maximum(e2_cell_int, 0.0))

# (b) Cell-average |∇e|  (compute via ∫_K |∇e|^2 and volumes; then take sqrt of average)
grad2_cell_int = cellwise_integral(ufl.inner(ufl.grad(e), ufl.grad(e)))

# Cell volumes via assembling 1*dx in DG0
vol_cell = cellwise_integral(ufl.as_ufl(1.0))

# Average |∇e|^2 on each cell, then sqrt
avg_grad2 = np.divide(grad2_cell_int, np.maximum(vol_cell, 1e-30))
err_grad_avg = fem.Function(W, name="err_grad_cellavg")
err_grad_avg.x.array[:] = np.sqrt(np.maximum(avg_grad2, 0.0))

# 5) (Optional) Also write the raw per-cell average of e^2 if you want a smooth scale
avg_e2 = np.divide(e2_cell_int, np.maximum(vol_cell, 1e-30))
err_cell_e2_avg = fem.Function(W, name="err_cell_e2_avg")
err_cell_e2_avg.x.array[:] = avg_e2

# 6) Write everything to a single XDMF (easy to browse in ParaView)
out = "results/err_fields.xdmf"
with io.XDMFFile(comm, out, "w") as xdmf:
    xdmf.write_mesh(V.mesh)
    xdmf.write_function(e, 0.0)
    xdmf.write_function(rel, 0.0)
    xdmf.write_function(err_cell_int, 0.0)
    xdmf.write_function(err_cell_L2, 0.0)
    xdmf.write_function(err_cell_e2_avg, 0.0)
    xdmf.write_function(err_grad_avg, 0.0)

if comm.rank == 0:
    print(f"[OK] Wrote error fields to {out}")
    print("Open in ParaView, color by:")
    print("  • err_node (CG1)           — nodal error")
    print("  • err_rel_node (CG1)       — robust relative nodal error")
    print("  • err_cell_L2 (DG0)        — sqrt(∫_K e^2 dx) per cell")
    print("  • err_grad_cellavg (DG0)   — cell-avg |∇e|")

