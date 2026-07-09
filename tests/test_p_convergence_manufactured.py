"""
Test: p-refinement convergence on a smooth 3D manufactured solution
====================================================================
Proves exponential (spectral) p-convergence on the same smooth 3D manufactured
Poisson solution. Uses a fixed coarse mesh so the discretisation error dominates
at all p values tested.

Exact solution on [0,1]^3:
  u_exact = sin(π x) * sin(π y) * sin(π z)
  f = 3π² * u_exact

Fixed mesh: n=4 (--quick, default) or n=6 (--full).
Degrees tested: [1, 2, 3, 4, 5].

Expected: each p step should reduce L2 by a large factor for smooth solutions.
The p=1→2 step should give at least 10× L2 reduction on a coarse mesh.

Run:
    ./run_dolfinx.sh tests/test_p_convergence_manufactured.py [--quick|--full] [options]
"""
import argparse
import csv
import os
import time

import numpy as np
from mpi4py import MPI

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_case(n: int, deg: int, outdir: str, write_xdmf: bool) -> dict:
    """Solve the manufactured Poisson problem on an n×n×n box at degree deg."""

    # --- Mesh ---
    domain = mesh.create_box(
        COMM,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        n=[n, n, n],
        cell_type=mesh.CellType.tetrahedron,
    )

    V = fem.functionspace(domain, ("Lagrange", deg))
    x = ufl.SpatialCoordinate(domain)

    # Manufactured solution and source (Lx=Ly=Lz=1)
    u_exact_ufl = (
        ufl.sin(np.pi * x[0])
        * ufl.sin(np.pi * x[1])
        * ufl.sin(np.pi * x[2])
    )
    f_ufl = 3.0 * np.pi**2 * u_exact_ufl

    # --- Dirichlet BC: u = u_exact on all 6 faces ---
    u_D = fem.Function(V)
    u_D.name = "phi_exact"
    u_D.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    facets = mesh.locate_entities_boundary(
        domain,
        domain.topology.dim - 1,
        lambda X: np.full(X.shape[1], True, dtype=bool),
    )
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
    bc = fem.dirichletbc(u_D, dofs)

    # --- Variational form ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx

    # Use LU for p >= 3 (CG may converge slowly for high-order systems)
    if deg >= 3:
        petsc_opts = {"ksp_type": "preonly", "pc_type": "lu"}
    else:
        petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}

    problem = LinearProblem(
        a,
        L,
        petsc_options_prefix=f"pconv_n{n}_p{deg}_",
        bcs=[bc],
        petsc_options=petsc_opts,
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    t1 = time.perf_counter()
    uh.name = "phi"

    # --- Exact solution Function for error computation ---
    ue = fem.Function(V)
    ue.name = "phi_exact"
    ue.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )

    # Error as UFL expression
    e_ufl = uh - ue

    abs_L2 = float(
        np.sqrt(COMM.allreduce(
            fem.assemble_scalar(fem.form(e_ufl**2 * ufl.dx)),
            op=MPI.SUM,
        ))
    )
    abs_H1semi = float(
        np.sqrt(COMM.allreduce(
            fem.assemble_scalar(
                fem.form(ufl.inner(ufl.grad(e_ufl), ufl.grad(e_ufl)) * ufl.dx)
            ),
            op=MPI.SUM,
        ))
    )

    ndofs = V.dofmap.index_map.size_global
    ncells = domain.topology.index_map(domain.topology.dim).size_global

    # --- Optional XDMF output ---
    if write_xdmf:
        os.makedirs(outdir, exist_ok=True)
        xdmf_path = os.path.join(outdir, f"p_conv_n{n}_p{deg}.xdmf")

        # Interpolate error into a Function for XDMF
        e_fn = fem.Function(V)
        e_fn.name = "error"
        e_fn.interpolate(
            fem.Expression(e_ufl, V.element.interpolation_points)
        )

        # dolfinx 0.10: XDMF only accepts degree-1 output; interpolate if needed
        if deg > 1:
            V1 = fem.functionspace(domain, ("Lagrange", 1))
            uh_out = fem.Function(V1, name="phi")
            uh_out.interpolate(uh)
            e_out = fem.Function(V1, name="error")
            e_out.interpolate(e_fn)
        else:
            uh_out = uh
            e_out  = e_fn

        with io.XDMFFile(COMM, xdmf_path, "w") as xf:
            xf.write_mesh(domain)
            xf.write_function(uh_out)
            xf.write_function(e_out)

    return {
        "deg": deg,
        "n": n,
        "cells": ncells,
        "dofs": ndofs,
        "abs_L2": abs_L2,
        "abs_H1semi": abs_H1semi,
        "time_s": t1 - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="p-convergence test for 3D manufactured Poisson problem"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Use n=4 (default)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Use n=6",
    )
    parser.add_argument(
        "--outdir",
        default="tests/test_p_convergence_manufactured/output",
        help="Output directory for XDMF and CSV files",
    )
    parser.add_argument("--write-xdmf", action="store_true",
                        help="Write solution + error fields to XDMF")
    parser.add_argument("--save-csv", action="store_true",
                        help="Write results table to outdir/p_convergence_results.csv")
    parser.add_argument("--max-dofs", type=int, default=500_000, metavar="INT",
                        help="Skip p values above this DOF count (default: 500000)")
    args = parser.parse_args()

    n_mesh = 6 if args.full else 4
    degrees = [1, 2, 3, 4, 5]

    results = []
    skipped = []

    for deg in degrees:
        r = solve_case(n_mesh, deg, args.outdir, args.write_xdmf)

        if r["dofs"] > args.max_dofs:
            skipped.append(deg)
            if COMM.rank == 0:
                print(
                    f"  Skipping p={deg}: DOFs {r['dofs']:,} > --max-dofs {args.max_dofs:,}"
                )
            continue

        results.append(r)

    if COMM.rank == 0:
        print(f"\n=== p-convergence: 3D manufactured solution, mesh n={n_mesh} ===")
        print(
            f"{'p':>4} {'dofs':>10} {'abs_L2':>14} {'abs_H1semi':>14} "
            f"{'L2_reduction_vs_prev':>22} {'time_s':>8}"
        )
        print("-" * 80)

        rows = []
        L2_reduction_p1_p2 = float("nan")

        for i, r in enumerate(results):
            if i == 0:
                reduction_str = "—"
                reduction_val = float("nan")
            else:
                prev_L2 = results[i - 1]["abs_L2"]
                curr_L2 = r["abs_L2"]
                if curr_L2 > 0:
                    reduction_val = prev_L2 / curr_L2
                else:
                    reduction_val = float("inf")
                reduction_str = f"{reduction_val:.1f}x"

                # Capture the p=1→2 reduction for PASS/FAIL
                if results[i - 1]["deg"] == 1 and r["deg"] == 2:
                    L2_reduction_p1_p2 = reduction_val

            print(
                f"{r['deg']:>4} {r['dofs']:>10,} {r['abs_L2']:>14.3e} "
                f"{r['abs_H1semi']:>14.3e} {reduction_str:>22} {r['time_s']:>8.2f}"
            )

            rows.append({
                "p": r["deg"],
                "n": r["n"],
                "cells": r["cells"],
                "dofs": r["dofs"],
                "abs_L2": r["abs_L2"],
                "abs_H1semi": r["abs_H1semi"],
                "L2_reduction_vs_prev": reduction_val,
                "time_s": r["time_s"],
            })

        # --- CSV ---
        if args.save_csv:
            os.makedirs(args.outdir, exist_ok=True)
            csv_path = os.path.join(args.outdir, "p_convergence_results.csv")
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["p", "n", "cells", "dofs", "abs_L2", "abs_H1semi",
                                "L2_reduction_vs_prev", "time_s"],
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nResults written to {csv_path}")

        # --- Summary ---
        print(
            "\nExpected: each p step should reduce L2 by a large factor for smooth solutions."
        )
        print(
            "If errors plateau: mesh is too fine for higher p — reduce n or use --full for n=4."
        )

        # --- PASS / FAIL ---
        if np.isnan(L2_reduction_p1_p2):
            print("FAIL: could not compute p=1→2 L2 reduction (missing results).")
        elif L2_reduction_p1_p2 >= 10.0:
            print(f"PASS  (p=1→2 L2 reduction = {L2_reduction_p1_p2:.1f}x ≥ 10×)")
        else:
            print(
                f"FAIL  (p=1→2 L2 reduction = {L2_reduction_p1_p2:.1f}x < 10× required; "
                "mesh may be too fine — try reducing n)"
            )


if __name__ == "__main__":
    main()
