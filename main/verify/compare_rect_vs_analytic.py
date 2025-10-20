"""Compare a FEniCSx solution to the analytic rectangular‑gate potential.

This script loads a numerical solution from an XDMF file, constructs the
corresponding analytic potential on exactly the same finite element mesh and
function space, and then computes relative errors per degree of freedom and
per cell.  Results are written in XDMF/CSV form for inspection and plotting.

Author: PoissonSolver verification suite
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Sequence

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
import ufl

from verify.analytic_rect import phi0_rect


def parse_xs(xs_str: str, a: float) -> List[float]:
    """Parse a comma‑separated list of gate centres.

    Entries may be numeric or multiples of ``a`` using e.g. '-2a'.
    """
    xs_list: List[float] = []
    for item in xs_str.split(','):
        item = item.strip()
        if not item:
            continue
        if 'a' in item:
            coeff_str = item.replace('a', '')
            if coeff_str in ('', '+'):
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)
            xs_list.append(coeff * a)
        else:
            xs_list.append(float(item))
    return xs_list


def parse_csv_float(seq: str) -> List[float]:
    """Parse a comma‑separated string into a list of floats."""
    return [float(item.strip()) for item in seq.split(',') if item.strip()]


def gather_global_max(value: float, comm: MPI.Intracomm) -> float:
    """Compute the global maximum of a scalar across all ranks."""
    return comm.allreduce(value, op=MPI.MAX)


def gather_global_sum(value: float, comm: MPI.Intracomm) -> float:
    """Compute the global sum of a scalar across all ranks."""
    return comm.allreduce(value, op=MPI.SUM)


def assemble_relative_error_cellwise(
    phi_h: fem.Function,
    phi_ref: fem.Function,
    tau: float,
) -> fem.Function:
    """Assemble the cellwise relative error via L2 projection onto DG0.

    r(x) = |φ_h(x) − φ_ref(x)| / max(|φ_ref(x)|, τ)
    and then projected onto a DG0 space (one value per cell).
    """
    mesh = phi_h.function_space.mesh
    V0 = fem.functionspace(mesh, ("DG", 0))
    r_expr = ufl.abs(phi_h - phi_ref) / ufl.max_value(ufl.abs(phi_ref), tau)
    u = ufl.TrialFunction(V0)
    v = ufl.TestFunction(V0)
    a_proj = fem.form(u * v * ufl.dx)
    L_proj = fem.form(r_expr * v * ufl.dx)
    r_cell = fem.Function(V0, name="rel_error_cell")
    problem = fem.petsc.LinearProblem(
        a_proj,
        L_proj,
        u=r_cell,
        petsc_options={"ksp_type": "preonly", "pc_type": "jacobi"},
    )
    problem.solve()
    return r_cell


def write_function_xdmf(
    mesh,
    func: fem.Function,
    path: str,
    comm: MPI.Intracomm,
) -> None:
    """Write a function (and its mesh) to an XDMF file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.XDMFFile(comm, path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(func)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure relative error of a FEniCSx solution versus the analytic rectangular‑gate potential",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phi-xdmf", required=True, help="XDMF file containing the numerical solution φ_h (with mesh and function)")
    parser.add_argument("--a", type=float, required=True, help="Half side‑length of each gate (metres)")
    parser.add_argument("--xs", type=str, required=True, help="Comma‑separated gate centre positions (metres); entries may include 'a'")
    parser.add_argument("--Vs", type=str, required=True, help="Comma‑separated gate voltages (volts), same length as xs")
    parser.add_argument("--tau", type=float, default=1e-9, help="Floor τ for the denominator in the relative error")
    parser.add_argument("--out-stem", type=str, default=None, help="Output file stem; defaults to results/verify/<basename(phi)>")
    parser.add_argument("--z-from-mesh-midplane", action="store_true", help="Use nodal z from the mesh instead of a constant z")
    parser.add_argument("--z", dest="z_const", type=float, default=None, help="Constant z plane (metres) when not using mesh z")
    parser.add_argument(
        "--sample-line",
        nargs=5,
        type=float,
        metavar=("Y", "Z", "X0", "X1", "N"),
        help="Sample along a line y=Y, z=Z from x=X0 to X1 with N points; writes an additional CSV",
    )
    parser.add_argument("--field-name", type=str, default="phi", help="Name of the dataset inside the XDMF file")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Parse gate parameters
    xs_vals = parse_xs(args.xs, args.a)
    vs_vals = parse_csv_float(args.Vs)
    if len(xs_vals) != len(vs_vals):
        if rank == 0:
            raise ValueError(f"Number of gate centres ({len(xs_vals)}) does not match number of gate voltages ({len(vs_vals)})")
        return

    # Determine output stem
    base_name = os.path.splitext(os.path.basename(args.phi_xdmf))[0]
    out_stem = args.out_stem or os.path.join("results", "verify", base_name)

    # Load numerical solution and mesh
    with io.XDMFFile(comm, args.phi_xdmf, "r") as xdmf:
        try:
            phi_h = xdmf.read_function(name=args.field_name)
        except Exception:
            phi_h = xdmf.read_function()
        mesh = phi_h.function_space.mesh
        V = phi_h.function_space

    # Require a scalar Lagrange space
    if phi_h.value_rank != 0:
        if rank == 0:
            raise RuntimeError("Only scalar Lagrange spaces are supported in this verification script.")
        return

    # Build analytic reference on same DOFs
    phi_ref = fem.Function(V, name="phi_ref")
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
    x_coords = dof_coords[:, 0]
    y_coords = dof_coords[:, 1]
    if args.z_from_mesh_midplane:
        z_coords = dof_coords[:, 2]
    else:
        if args.z_const is None:
            if rank == 0:
                raise ValueError("--z must be provided if --z-from-mesh-midplane is not set")
            return
        z_coords = np.full_like(x_coords, args.z_const)
    ref_vals = phi0_rect(x_coords, y_coords, z_coords, args.a, xs_vals, vs_vals)
    phi_ref.x.array[:] = ref_vals

    # Compute nodal relative error
    phi_h_arr = phi_h.x.array
    phi_ref_arr = phi_ref.x.array
    denom = np.maximum(np.abs(phi_ref_arr), args.tau)
    r_dof = np.abs(phi_h_arr - phi_ref_arr) / denom
    rel_error_dof = fem.Function(V, name="rel_error_dof")
    rel_error_dof.x.array[:] = r_dof

    # Compute cellwise relative error via projection
    rel_error_cell = assemble_relative_error_cellwise(phi_h, phi_ref, args.tau)

    # Global reductions
    local_max_dof = r_dof.max(initial=0.0) if r_dof.size > 0 else 0.0
    global_max_dof = gather_global_max(local_max_dof, comm)
    local_sum_sq_dof = float(np.sum(r_dof * r_dof))
    global_sum_sq_dof = gather_global_sum(local_sum_sq_dof, comm)
    global_l2_dof = np.sqrt(global_sum_sq_dof)

    cell_arr = rel_error_cell.x.array
    local_max_cell = cell_arr.max(initial=0.0) if cell_arr.size > 0 else 0.0
    global_max_cell = gather_global_max(local_max_cell, comm)
    local_sum_cell = float(np.sum(cell_arr))
    local_count_cell = float(cell_arr.size)
    global_sum_cell = gather_global_sum(local_sum_cell, comm)
    global_count_cell = gather_global_sum(local_count_cell, comm)
    global_mean_cell = global_sum_cell / global_count_cell if global_count_cell > 0 else 0.0

    local_dofs = float(phi_h_arr.size)
    global_dofs = gather_global_sum(local_dofs, comm)
    num_cells_local = float(mesh.topology.index_map(mesh.topology.dim).size_local)
    num_cells_global = gather_global_sum(num_cells_local, comm)

    # Rank 0 writes outputs
    if rank == 0:
        out_dir = os.path.dirname(out_stem)
        os.makedirs(out_dir, exist_ok=True)
        cell_path = f"{out_stem}_rel_error_cellwise.xdmf"
        write_function_xdmf(mesh, rel_error_cell, cell_path, comm)
        nodal_path = f"{out_stem}_rel_error_nodal.xdmf"
        write_function_xdmf(mesh, rel_error_dof, nodal_path, comm)
        summary_path = f"{out_stem}_summary.csv"
        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "phi_xdmf", "a", "xs", "Vs", "tau",
                    "max_rel_cell", "mean_rel_cell",
                    "Linf_rel_dof", "L2_rel_dof",
                    "n_cells", "n_dofs",
                ])
            writer.writerow([
                args.phi_xdmf,
                args.a,
                ",".join(f"{x:.6g}" for x in xs_vals),
                ",".join(f"{v:.6g}" for v in vs_vals),
                args.tau,
                f"{global_max_cell:.6e}",
                f"{global_mean_cell:.6e}",
                f"{global_max_dof:.6e}",
                f"{global_l2_dof:.6e}",
                int(num_cells_global),
                int(global_dofs),
            ])
        print(f"[REL-DOF] Linf={global_max_dof:.6e} L2={global_l2_dof:.6e}")
        print(f"[REL-CELL] max={global_max_cell:.6e} mean={global_mean_cell:.6e}")

    # Sample along a line if requested
    if args.sample_line is not None:
        y_line, z_line, x0, x1, n = args.sample_line
        n_int = int(n)
        xs_line = np.linspace(x0, x1, n_int)
        ys_line = np.full(n_int, y_line)
        zs_line = np.full(n_int, z_line)
        phi_ref_line = phi0_rect(xs_line, ys_line, zs_line, args.a, xs_vals, vs_vals)
        pts = np.vstack((xs_line, ys_line, zs_line))
        values = np.zeros(n_int, dtype=float)
        phi_h.eval(values, pts)
        delta = values - phi_ref_line
        denom_line = np.maximum(np.abs(phi_ref_line), args.tau)
        r_line = np.abs(delta) / denom_line
        if rank == 0:
            line_path = f"{out_stem}_line_sample.csv"
            with open(line_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "phi_h", "phi_ref", "delta", "rel_error"])
                for xi, ph, pr, d, r in zip(xs_line, values, phi_ref_line, delta, r_line):
                    writer.writerow([f"{xi:.6g}", f"{ph:.6e}", f"{pr:.6e}", f"{d:.6e}", f"{r:.6e}"])


if __name__ == "__main__":
    main()
