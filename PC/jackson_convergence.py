from mpi4py import MPI
from dolfinx import fem, io, mesh
import ufl
import numpy as np
import argparse
import csv
import pathlib

# ------------------------------------------------------------
# 1. Analytic solution for the Jackson case
# ------------------------------------------------------------

def phi_exact_callable(x: np.ndarray, params: dict) -> np.ndarray:
    """
    x: array of shape (gdim, npoints) with coordinates
    params: dict with things like q, eps1, eps2, z0, etc.

    TODO: Replace this stub with your actual analytic expression.
    For example, for a charge q at (0,0,z0) above a planar interface at z=0:
      eps1: permittivity in z>0 (where the charge sits)
      eps2: permittivity in z<0
    """
    q = params["q"]
    eps1 = params["eps1"]
    eps2 = params["eps2"]
    z0 = params["z0"]

    # positions
    X, Y, Z = x  # each is (npoints,)

    # Real charge at (0, 0, z0)
    r_real = np.sqrt(X**2 + Y**2 + (Z - z0)**2)

    # Image charge at (0, 0, -z0)
    q_img = q * (eps1 - eps2) / (eps1 + eps2)
    r_img = np.sqrt(X**2 + Y**2 + (Z + z0)**2)

    four_pi = 4.0 * np.pi

    # Potential in the upper half-space (z>0). This is a standard Jackson result.
    phi = (q / (four_pi * eps1 * r_real)) + (q_img / (four_pi * eps1 * r_img))

    return phi


# ------------------------------------------------------------
# 2. Your existing Jackson solver goes here
# ------------------------------------------------------------

def solve_jackson_case(h: float, deg: int, params: dict):
    """
    Solve your Jackson interface Poisson problem on a mesh with size ~h
    and polynomial degree 'deg'.

    Returns:
        domain: dolfinx.mesh.Mesh
        V:      dolfinx.fem.FunctionSpace
        phi:    dolfinx.fem.Function (numerical potential)

    TODO: Replace the body of this function with a call to your existing
    Jackson solver code. You probably already have something like
    'rect_gate_benchmark.py' or 'jackson_interface.py' that builds:

      - the mesh (with interface and eps1/eps2),
      - the function space V = fem.FunctionSpace(domain, ("Lagrange", deg)),
      - BCs using the analytic expression on the outer boundary,
      - the Poisson variational problem,
      - the solution phi.

    Here I just put a NotImplementedError so you won't accidentally run it
    before wiring things up.
    """
    raise NotImplementedError("Hook this up to your existing Jackson solver.")


# ------------------------------------------------------------
# 3. Relative error computation + heat map field
# ------------------------------------------------------------

def compute_relative_errors(domain: mesh.Mesh,
                            V: fem.FunctionSpace,
                            phi: fem.Function,
                            params: dict,
                            phi_floor: float = 1e-6):
    """
    Given the mesh, function space, and numerical solution phi,
    compute:

    - L2 relative error (global)
    - H1 relative error (global)
    - pointwise relative error field e_rel(x)
    - analytic solution field phi_exact(x)

    All on the same function space V (assumed Lagrange).
    """
    comm = domain.comm
    dx = ufl.Measure("dx", domain=domain)

    # Interpolate analytic solution
    phi_exact = fem.Function(V, name="phi_exact")
    phi_exact.interpolate(lambda x: phi_exact_callable(x, params))

    # UFL difference
    diff = phi - phi_exact

    # L2 relative error
    form_num_L2 = fem.form(ufl.inner(diff, diff) * dx)
    form_den_L2 = fem.form(ufl.inner(phi_exact, phi_exact) * dx)

    num_L2 = fem.assemble_scalar(form_num_L2)
    den_L2 = fem.assemble_scalar(form_den_L2)

    num_L2 = comm.allreduce(num_L2, op=MPI.SUM)
    den_L2 = comm.allreduce(den_L2, op=MPI.SUM)

    eL2_rel = np.sqrt(num_L2) / np.sqrt(den_L2)

    # H1 (seminorm) relative error
    form_num_H1 = fem.form(
        ufl.inner(ufl.grad(diff), ufl.grad(diff)) * dx
    )
    form_den_H1 = fem.form(
        ufl.inner(ufl.grad(phi_exact), ufl.grad(phi_exact)) * dx
    )

    num_H1 = fem.assemble_scalar(form_num_H1)
    den_H1 = fem.assemble_scalar(form_den_H1)

    num_H1 = comm.allreduce(num_H1, op=MPI.SUM)
    den_H1 = comm.allreduce(den_H1, op=MPI.SUM)

    eH1_rel = np.sqrt(num_H1) / np.sqrt(den_H1)

    # Pointwise relative error field for heat map
    e_rel = fem.Function(V, name="err_rel")
    num_vals = phi.x.array
    exact_vals = phi_exact.x.array
    denom_vals = np.maximum(np.abs(exact_vals), phi_floor)
    e_rel_vals = np.abs(num_vals - exact_vals) / denom_vals
    e_rel.x.array[:] = e_rel_vals

    return eL2_rel, eH1_rel, e_rel, phi_exact


# ------------------------------------------------------------
# 4. XDMF writer for phi, phi_exact, and error heat map
# ------------------------------------------------------------

def write_xdmf(domain: mesh.Mesh,
               phi: fem.Function,
               phi_exact: fem.Function,
               err_rel: fem.Function,
               outpath: pathlib.Path,
               t: float = 0.0):
    """
    Write mesh + fields to an XDMF file you can open in ParaView.
    """
    comm = domain.comm
    if comm.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)

    with io.XDMFFile(comm, outpath.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, t)
        xdmf.write_function(phi_exact, t)
        xdmf.write_function(err_rel, t)


# ------------------------------------------------------------
# 5. h- and p-refinement driver
# ------------------------------------------------------------

def run_single_case(h: float,
                    deg: int,
                    params: dict,
                    outdir: pathlib.Path,
                    phi_floor: float = 1e-6):
    """
    Run one (h, deg) case:
      - solve Jackson problem
      - compute relative errors
      - write XDMF with error heat map

    Returns:
      (eL2_rel, eH1_rel, n_cells, n_dofs, xdmf_path)
    """
    comm = MPI.COMM_WORLD
    domain, V, phi = solve_jackson_case(h, deg, params)

    # Global counts
    n_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    n_dofs_local = V.dofmap.index_map.size_local

    n_cells = comm.allreduce(n_cells_local, op=MPI.SUM)
    n_dofs = comm.allreduce(n_dofs_local, op=MPI.SUM)

    eL2_rel, eH1_rel, err_rel, phi_exact = compute_relative_errors(
        domain, V, phi, params, phi_floor=phi_floor
    )

    # Construct a filename like jackson_h8.0e-09_p2.xdmf
    base = f"jackson_h{h:.3e}_p{deg}"
    xdmf_path = outdir / f"{base}.xdmf"
    write_xdmf(domain, phi, phi_exact, err_rel, xdmf_path)

    return eL2_rel, eH1_rel, n_cells, n_dofs, xdmf_path


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    parser = argparse.ArgumentParser(
        description="Jackson interface h/p refinement with relative errors and XDMF heat maps."
    )
    parser.add_argument("--hs", type=str, default="1e-8",
                        help="Comma-separated list of target mesh sizes h, e.g. '8e-9,4e-9,2e-9'.")
    parser.add_argument("--degrees", type=str, default="1,2",
                        help="Comma-separated list of polynomial degrees, e.g. '1,2,3'.")
    parser.add_argument("--eps1", type=float, default=11.7,
                        help="Permittivity above the interface.")
    parser.add_argument("--eps2", type=float, default=3.9,
                        help="Permittivity below the interface.")
    parser.add_argument("--q", type=float, default=1.602176634e-19,
                        help="Point charge (C).")
    parser.add_argument("--z0", type=float, default=1e-8,
                        help="z-position of the point charge (m).")
    parser.add_argument("--phi-floor", type=float, default=1e-6,
                        help="Floor for |phi_exact| in relative error.")
    parser.add_argument("--outdir", type=str, default="results/jackson_convergence",
                        help="Output directory for XDMF and CSV.")
    parser.add_argument("--csv-name", type=str, default="jackson_errors.csv",
                        help="CSV filename inside outdir.")
    args = parser.parse_args()

    hs = [float(s) for s in args.hs.split(",")]
    degrees = [int(s) for s in args.degrees.split(",")]
    outdir = pathlib.Path(args.outdir)

    # Parameters passed to analytic solution (and optionally your solver)
    params = dict(
        q=args.q,
        eps1=args.eps1,
        eps2=args.eps2,
        z0=args.z0,
    )

    # CSV on rank 0
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / args.csv_name
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            ["h", "deg", "n_cells", "n_dofs", "eL2_rel", "eH1_rel", "xdmf_file"]
        )
    else:
        writer = None
        csv_file = None

    # Loop over h and p
    for h in hs:
        for deg in degrees:
            if rank == 0:
                print(f"=== Running h={h:.3e}, p={deg} ===", flush=True)

            eL2_rel, eH1_rel, n_cells, n_dofs, xdmf_path = run_single_case(
                h, deg, params, outdir, phi_floor=args.phi_floor
            )

            if rank == 0:
                writer.writerow(
                    [h, deg, n_cells, n_dofs, eL2_rel, eH1_rel, xdmf_path.name]
                )
                print(
                    f"h={h:.3e}, p={deg}: "
                    f"cells={n_cells}, dofs={n_dofs}, "
                    f"eL2_rel={eL2_rel:.3e}, eH1_rel={eH1_rel:.3e}"
                )

    if rank == 0 and csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
