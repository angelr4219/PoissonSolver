
from mpi4py import MPI
import numpy as np
import argparse
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc
from ufl import dx

# --------- optional analytic reference (edit this) ----------
def phi_exact(x):
    # x is shape (3, N). Replace with your analytic rectangular-gate potential if desired.
    # For now, return zeros so relative error is skipped unless you change this.
    return np.zeros(x.shape[1], dtype=np.float64)

def has_analytic():
    # flip to True after you implement phi_exact above
    return False
# ------------------------------------------------------------

def cellwise_integral(expr, W):
    """Assemble cellwise integrals ∫_K expr dx into DG0 coefficient array."""
    v = ufl.TestFunction(W)
    form = fem.form(expr * v * dx)
    vec = assemble_vector(form)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    return vec.array  # owned (local) cell values in correct ordering

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="xin", required=True, help="Input XDMF containing mesh + solution")
    ap.add_argument("--field", dest="fname", default="phi", help="Name of solution field in XDMF (default: phi)")
    ap.add_argument("--out", dest="xout", required=True, help="Output XDMF for error fields")
    ap.add_argument("--tau", dest="tau", type=float, default=None, help="Floor for robust relative error")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD

    # --- Read mesh and solution from XDMF ---
    with io.XDMFFile(comm, args.xin, "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")
        msh.name = "mesh"
        # P1 space is the most common; adjust if your saved field used different element
        V = fem.FunctionSpace(msh, ("CG", 1))
        uh = fem.Function(V, name=args.fname)
        try:
            xdmf.read_function(uh, name=args.fname)
        except Exception:
            # Some files don't store the name; try default read
            xdmf.read_function(uh)

    # --- Build reference on V ---
    uref = fem.Function(V, name="u_ref")
    if has_analytic():
        uref.interpolate(phi_exact)
    else:
        # If you don’t have analytic yet, just copy uh so absolute errors become zero
        # (useful only to test pipeline). Replace with your reference later.
        uref.x.array[:] = uh.x.array

    # --- Nodal error (CG1) ---
    e = fem.Function(V, name="err_node")
    e.x.array[:] = uh.x.array - uref.x.array

    # --- Robust relative nodal error (CG1), optional ---
    rel = fem.Function(V, name="err_rel_node")
    abs_ref = np.abs(uref.x.array)
    if args.tau is None:
        # If no tau given, use a heuristic based on field range
        tau = 1e-6 * (abs_ref.max() - abs_ref.min() + 1e-30)
    else:
        tau = args.tau
    rel.x.array[:] = np.abs(e.x.array) / np.maximum(abs_ref, tau)

    # --- DG0 spaces for cellwise quantities ---
    W = fem.FunctionSpace(msh, ("DG", 0))

    # ∫_K e^2 dx
    e2_cell_int = cellwise_integral(ufl.inner(e, e), W)
    err_cell_int = fem.Function(W, name="err_cell_e2_int")
    err_cell_int.x.array[:] = e2_cell_int

    # sqrt(∫_K e^2 dx)
    err_cell_L2 = fem.Function(W, name="err_cell_L2")
    err_cell_L2.x.array[:] = np.sqrt(np.maximum(e2_cell_int, 0.0))

    # cell volumes
    vol_cell = cellwise_integral(ufl.as_ufl(1.0), W)

    # cell-average |∇e|
    grad2_cell_int = cellwise_integral(ufl.inner(ufl.grad(e), ufl.grad(e)), W)
    avg_grad2 = np.divide(grad2_cell_int, np.maximum(vol_cell, 1e-30))
    err_grad_avg = fem.Function(W, name="err_grad_cellavg")
    err_grad_avg.x.array[:] = np.sqrt(np.maximum(avg_grad2, 0.0))

    # optional: avg of e^2 (per-cell)
    avg_e2 = np.divide(e2_cell_int, np.maximum(vol_cell, 1e-30))
    err_cell_e2_avg = fem.Function(W, name="err_cell_e2_avg")
    err_cell_e2_avg.x.array[:] = avg_e2

    # --- write out ---
    with io.XDMFFile(comm, args.xout, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(e, 0.0)
        xdmf.write_function(rel, 0.0)
        xdmf.write_function(err_cell_int, 0.0)
        xdmf.write_function(err_cell_L2, 0.0)
        xdmf.write_function(err_cell_e2_avg, 0.0)
        xdmf.write_function(err_grad_avg, 0.0)

    if comm.rank == 0:
        print(f"[OK] Wrote error fields to {args.xout}")
        print("Open in ParaView and color by any of:")
        print("  • err_node (CG1)           — nodal error")
        print("  • err_rel_node (CG1)       — robust relative nodal error")
        print("  • err_cell_L2 (DG0)        — sqrt(∫_K e^2 dx) per cell")
        print("  • err_grad_cellavg (DG0)   — cell-avg |∇e|")
        if not has_analytic():
            print("NOTE: phi_exact() is a stub; implement it to get meaningful errors.")
