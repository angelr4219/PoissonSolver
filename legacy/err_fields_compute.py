# err_fields_compute.py
# Compare a saved FEM solution against an analytic rectangular-gate potential and write error fields.
# Usage (inside dolfinx container):
#   python3 err_fields_compute.py \
#       --in results/phi_p2a.xdmf --field phi \
#       --out results/err_fields_p2a.xdmf --tau 1e-9

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

import argparse, os, sys
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_vector

# ---- Try to import your analytic function ----
ANALYTIC = None
for modname in ("exact_rect_gates", "exact_react_gates"):
    try:
        ANALYTIC = __import__(modname)
        if hasattr(ANALYTIC, "phi_exact"):
            if rank == 0:
                print(f"[INFO] Using analytic phi_exact from {modname}.py")
            break
        ANALYTIC = None
    except Exception:
        ANALYTIC = None

def cellwise_integral(expr, W):
    v = ufl.TestFunction(W)
    form = fem.form(expr * v * ufl.dx)
    vec = assemble_vector(form)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    return vec.array

def read_mesh_and_field(xin_path, preferred_name=None):
    """Read first mesh and a scalar field from XDMF.
    Tries preferred_name, else falls back to first available."""
    if rank == 0:
        print(f"[INFO] Reading input XDMF: {xin_path}")
    if not os.path.exists(xin_path):
        raise FileNotFoundError(xin_path)

    with io.XDMFFile(comm, xin_path, "r") as xdmf:
        # Read first grid/mesh
        try:
            msh = xdmf.read_mesh()
        except TypeError:
            msh = xdmf.read_mesh(name=None)
        msh.name = "mesh"

        # Try common element degrees
        for deg in (1, 2):
            V = fem.functionspace(msh, ("CG", deg))
            uh = fem.Function(V, name=preferred_name or "u")
            try:
                if preferred_name:
                    xdmf.read_function(uh, name=preferred_name)
                else:
                    xdmf.read_function(uh)  # first scalar field
                if rank == 0:
                    arr = uh.x.array
                    print(f"[OK] Loaded field on CG{deg} "
                          f"(dofs={arr.size}, min={arr.min():.3e}, max={arr.max():.3e})")
                return msh, V, uh
            except Exception as e:
                if rank == 0:
                    print(f"[WARN] Failed to read field on CG{deg}: {e}")
        raise RuntimeError("Could not read the solution field from XDMF; "
                           "check --field name or element degree used when saving.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="xin", required=True, help="Input XDMF (mesh + solution)")
    ap.add_argument("--field", dest="fname", default="phi", help="Field name in XDMF (default: phi)")
    ap.add_argument("--out", dest="xout", required=True, help="Output XDMF for error fields")
    ap.add_argument("--tau", dest="tau", type=float, default=None, help="Floor for robust relative error")
    args = ap.parse_args()

    # Ensure output dir exists
    out_dir = os.path.dirname(args.xout) or "."
    if rank == 0 and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    comm.barrier()

    # --- load mesh and solution ---
    msh, V, uh = read_mesh_and_field(args.xin, preferred_name=args.fname)

    # --- analytic reference on V (if available) ---
    uref = fem.Function(V, name="u_ref")
    if ANALYTIC is not None:
        try:
            # expects (3, N) -> (N,)
            uref.interpolate(ANALYTIC.phi_exact)
            if rank == 0:
                print("[OK] Interpolated analytic reference onto V")
        except Exception as e:
            if rank == 0:
                print("[ERR] phi_exact interpolation failed; falling back to self-comparison:", e)
            uref.x.array[:] = uh.x.array
    else:
        if rank == 0:
            print("[WARN] No analytic module found; comparing to self (errors ~ 0).")
        uref.x.array[:] = uh.x.array

    # --- nodal error and relative error ---
    e = fem.Function(V, name="err_node")
    e.x.array[:] = uh.x.array - uref.x.array

    rel = fem.Function(V, name="err_rel_node")
    abs_ref = np.abs(uref.x.array)
    tau = args.tau if args.tau is not None else 1e-6 * (abs_ref.max() - abs_ref.min() + 1e-30)
    rel.x.array[:] = np.abs(e.x.array) / np.maximum(abs_ref, tau)

    # --- DG0 cellwise metrics ---
    W = fem.functionspace(msh, ("DG", 0))

    e2_cell_int = cellwise_integral(ufl.inner(e, e), W)
    err_cell_int = fem.Function(W, name="err_cell_e2_int")
    err_cell_int.x.array[:] = e2_cell_int

    err_cell_L2 = fem.Function(W, name="err_cell_L2")
    err_cell_L2.x.array[:] = np.sqrt(np.maximum(e2_cell_int, 0.0))

    vol_cell = cellwise_integral(ufl.as_ufl(1.0), W)

    grad2_cell_int = cellwise_integral(ufl.inner(ufl.grad(e), ufl.grad(e)), W)
    avg_grad2 = np.divide(grad2_cell_int, np.maximum(vol_cell, 1e-30))
    err_grad_avg = fem.Function(W, name="err_grad_cellavg")
    err_grad_avg.x.array[:] = np.sqrt(np.maximum(avg_grad2, 0.0))

    # --- global norms (diagnostic) ---
    l2_glob = np.sqrt(comm.allreduce(np.sum(e2_cell_int), op=MPI.SUM))
    h1_glob = np.sqrt(comm.allreduce(np.sum(grad2_cell_int), op=MPI.SUM))
    if rank == 0:
        print(f"[GLOBAL] ||e||_L2 = {l2_glob:.6e}")
        print(f"[GLOBAL] |e|_H1   = {h1_glob:.6e}")

    # --- write out (XDMF + H5) ---
    if rank == 0:
        print(f"[INFO] Writing error fields to: {args.xout}")
    with io.XDMFFile(comm, args.xout, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(e, 0.0)
        xdmf.write_function(rel, 0.0)
        xdmf.write_function(err_cell_int, 0.0)
        xdmf.write_function(err_cell_L2, 0.0)
        xdmf.write_function(err_grad_avg, 0.0)

    if rank == 0:
        h5 = os.path.splitext(args.xout)[0] + ".h5"
        print(f"[CHECK] XDMF exists: {os.path.exists(args.xout)}  H5 exists: {os.path.exists(h5)}")
        if os.path.exists(h5):
            print("[OK] Done. Open in ParaView:")
            print("     ", os.path.abspath(args.xout))
        else:
            print("[WARN] H5 sidecar missing (ParaView needs it). Check write permissions/mount and re-run.")

if __name__ == "__main__":
    main()

