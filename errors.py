
from mpi4py import MPI
import numpy as np
from dolfinx import fem, io
import ufl

def report_errors(domain, uh, exact_fun, out_prefix="results/phi_err", qdeg=4):
    """
    Minimal, stable error report:
      - builds uE by interpolation of `exact_fun`
      - computes global L2(e) and H1_seminorm(e)
      - computes Linf_dof on the function-space dofs
    Skips per-cell error fields (old PETSc path was version-sensitive).
    """
    comm = domain.comm
    V = uh.function_space
    uE = fem.Function(V, name="u_exact")
    uE.interpolate(exact_fun)

    e = uE - uh

    # Global L2 and H1 seminorm
    L2_e_sq_local  = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx(domain)))
    H1_e_sq_local  = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx(domain)))
    L2_e_sq  = comm.allreduce(L2_e_sq_local, op=MPI.SUM)
    H1_e_sq  = comm.allreduce(H1_e_sq_local, op=MPI.SUM)
    L2_e     = float(np.sqrt(L2_e_sq))
    H1_semi  = float(np.sqrt(H1_e_sq))

    # Linf on dofs
    uh_arr = uh.x.array
    uE_arr = uE.x.array
    Linf_dof_local = float(np.max(np.abs(uh_arr - uE_arr))) if uh_arr.size else 0.0
    Linf_dof = comm.allreduce(Linf_dof_local, op=MPI.MAX)

    # Write exact/solution for quick visualization (optional)
    try:
        with io.XDMFFile(comm, f"{out_prefix}.xdmf", "w") as xf:
            xf.write_mesh(domain)
            # store both fields if you like; ParaView can compare
            uh.name = "phi"
            xf.write_function(uh)
            uE.name = "u_exact"
            xf.write_function(uE)
    except Exception:
        pass

    # Dummy hotspot info so callers don’t crash
    metrics = {
        "L2": L2_e,
        "H1_seminorm": H1_semi,
        "Linf_dof": Linf_dof,
        "max_cell": {
            "local_id": None,
            "centroid": None,
            "L2_cell_norm": None,
        },
    }
    return metrics
