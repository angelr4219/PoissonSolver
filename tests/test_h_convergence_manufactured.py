"""
Test: h-refinement convergence on a smooth 3D manufactured solution
=====================================================================
Proves P1 (and optionally P2) FEM convergence rates on a smooth 3D manufactured
Poisson solution. This separates discretisation error from all other effects.

Exact solution on [0,1]^3:
  u_exact = sin(π x) * sin(π y) * sin(π z)
  f = 3π² * u_exact

Dirichlet BCs: u = u_exact on all 6 faces.

Expected convergence rates
  P1: L2 ≈ 2.0, H1semi ≈ 1.0
  P2: L2 ≈ 3.0, H1semi ≈ 2.0

Run:
    ./run_dolfinx.sh tests/test_h_convergence_manufactured.py [--quick|--full] [options]
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
    """Solve the manufactured Poisson problem on an n×n×n box and return metrics."""

    Lx = Ly = Lz = 1.0

    # --- Mesh ---
    domain = mesh.create_box(
        COMM,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        n=[n, n, n],
        cell_type=mesh.CellType.tetrahedron,
    )

    V = fem.functionspace(domain, ("Lagrange", deg))
    x = ufl.SpatialCoordinate(domain)

    # Manufactured solution and source
    u_exact_ufl = (
        ufl.sin(np.pi * x[0] / Lx)
        * ufl.sin(np.pi * x[1] / Ly)
        * ufl.sin(np.pi * x[2] / Lz)
    )
    coeff = np.pi**2 * (1.0 / Lx**2 + 1.0 / Ly**2 + 1.0 / Lz**2)
    f_ufl = coeff * u_exact_ufl

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

    problem = LinearProblem(
        a,
        L,
        petsc_options_prefix=f"hconv_n{n}_p{deg}_",
        bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    t1 = time.perf_counter()
    uh.name = "phi"

    # Error against the EXACT UFL expression (not the P1 interpolant).
    # Using uh - u_exact_ufl gives the true H1 seminorm convergence rate (≈1 for P1).
    # Using uh - Ih(u_exact) would give rate ≈2 for both norms due to Galerkin
    # orthogonality cancellation — a superconvergence artifact, not the real rate.
    e_ufl = uh - u_exact_ufl

    abs_L2 = float(
        np.sqrt(COMM.allreduce(
            fem.assemble_scalar(fem.form(e_ufl**2 * ufl.dx)),
            op=MPI.SUM,
        ))
    )
    norm_L2 = float(
        np.sqrt(COMM.allreduce(
            fem.assemble_scalar(fem.form(u_exact_ufl**2 * ufl.dx)),
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
    norm_H1semi = float(
        np.sqrt(COMM.allreduce(
            fem.assemble_scalar(
                fem.form(ufl.inner(ufl.grad(u_exact_ufl), ufl.grad(u_exact_ufl)) * ufl.dx)
            ),
            op=MPI.SUM,
        ))
    )

    ndofs = V.dofmap.index_map.size_global
    ncells = domain.topology.index_map(domain.topology.dim).size_global

    rel_L2 = abs_L2 / norm_L2 if norm_L2 > 0 else float("nan")
    rel_H1semi = abs_H1semi / norm_H1semi if norm_H1semi > 0 else float("nan")

    # --- Optional XDMF output ---
    if write_xdmf:
        os.makedirs(outdir, exist_ok=True)
        xdmf_path = os.path.join(outdir, f"h_conv_n{n}_p{deg}.xdmf")

        # Interpolate error into a Function for XDMF
        e_fn = fem.Function(V)
        e_fn.name = "error"
        e_fn.interpolate(
            fem.Expression(e_ufl, V.element.interpolation_points)
        )

        with io.XDMFFile(COMM, xdmf_path, "w") as xf:
            xf.write_mesh(domain)
            xf.write_function(uh)
            xf.write_function(e_fn)

    return {
        "n": n,
        "cells": ncells,
        "dofs": ndofs,
        "abs_L2": abs_L2,
        "rel_L2": rel_L2,
        "abs_H1semi": abs_H1semi,
        "rel_H1semi": rel_H1semi,
        "time_s": t1 - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="h-convergence test for 3D manufactured Poisson problem"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Use n=[4,8,16,32] (default)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Use n=[4,8,16,32,64]",
    )
    parser.add_argument("--p", type=int, default=1, metavar="INT",
                        help="Polynomial degree (default: 1)")
    parser.add_argument(
        "--outdir",
        default="tests/test_h_convergence_manufactured/output",
        help="Output directory for XDMF and CSV files",
    )
    parser.add_argument("--write-xdmf", action="store_true",
                        help="Write solution + error fields to XDMF")
    parser.add_argument("--save-csv", action="store_true",
                        help="Write results table to outdir/h_convergence_results.csv")
    parser.add_argument("--max-dofs", type=int, default=2_000_000, metavar="INT",
                        help="Skip cases above this DOF count (default: 2000000)")
    args = parser.parse_args()

    if args.full:
        n_values = [4, 8, 16, 32, 64]
    else:
        n_values = [4, 8, 16, 32]

    deg = args.p

    # Expected rates
    if deg == 1:
        expected_L2_rate = 2.0
        expected_H1_rate = 1.0
    else:
        # General: P_p → O(h^{p+1}) in L2, O(h^p) in H1semi
        expected_L2_rate = float(deg + 1)
        expected_H1_rate = float(deg)

    results = []
    skipped = []

    for n in n_values:
        # Estimate DOFs before building (rough: (n*p+1)^3 for hex, less for tet)
        est_dofs = (n * deg + 1) ** 3
        if est_dofs > args.max_dofs:
            skipped.append(n)
            if COMM.rank == 0:
                print(f"  Skipping n={n}: estimated DOFs {est_dofs:,} > --max-dofs {args.max_dofs:,}")
            continue

        r = solve_case(n, deg, args.outdir, args.write_xdmf)

        if r["dofs"] > args.max_dofs:
            skipped.append(n)
            if COMM.rank == 0:
                print(f"  Skipping n={n}: actual DOFs {r['dofs']:,} > --max-dofs {args.max_dofs:,}")
            continue

        results.append(r)

    if COMM.rank == 0:
        print(f"\n=== h-convergence: P{deg} FEM on 3D manufactured solution ===")
        print(f"{'n':>5} {'cells':>10} {'dofs':>10} {'rel_L2':>12} {'rel_H1semi':>12} "
              f"{'L2_rate':>9} {'H1_rate':>9} {'time_s':>8}")
        print("-" * 82)

        L2_rates = []
        H1_rates = []
        rows = []

        for i, r in enumerate(results):
            if i == 0:
                L2_rate_str = "—"
                H1_rate_str = "—"
                L2_rate_val = float("nan")
                H1_rate_val = float("nan")
            else:
                prev = results[i - 1]
                ratio = np.log(r["n"] / prev["n"])
                L2_rate_val = np.log(prev["rel_L2"] / r["rel_L2"]) / ratio
                H1_rate_val = np.log(prev["rel_H1semi"] / r["rel_H1semi"]) / ratio
                L2_rate_str = f"{L2_rate_val:.2f}"
                H1_rate_str = f"{H1_rate_val:.2f}"
                L2_rates.append(L2_rate_val)
                H1_rates.append(H1_rate_val)

            print(
                f"{r['n']:>5} {r['cells']:>10,} {r['dofs']:>10,} "
                f"{r['rel_L2']:>12.3e} {r['rel_H1semi']:>12.3e} "
                f"{L2_rate_str:>9} {H1_rate_str:>9} {r['time_s']:>8.2f}"
            )

            rows.append({
                "n": r["n"],
                "cells": r["cells"],
                "dofs": r["dofs"],
                "rel_L2": r["rel_L2"],
                "rel_H1semi": r["rel_H1semi"],
                "L2_rate": L2_rate_val,
                "H1_rate": H1_rate_val,
                "time_s": r["time_s"],
            })

        # --- CSV ---
        if args.save_csv:
            os.makedirs(args.outdir, exist_ok=True)
            csv_path = os.path.join(args.outdir, "h_convergence_results.csv")
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["n", "cells", "dofs", "rel_L2", "rel_H1semi",
                                "L2_rate", "H1_rate", "time_s"],
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nResults written to {csv_path}")

        # --- Expected rates summary ---
        print(f"\nExpected: P1 → L2 rate ≈ 2.0, H1semi rate ≈ 1.0")
        print(f"         P2 → L2 rate ≈ 3.0, H1semi rate ≈ 2.0")

        # --- PASS / FAIL ---
        tol = 0.2
        if len(L2_rates) == 0:
            print("FAIL: not enough refinement steps to compute rates.")
        else:
            failures = []
            for i, (lr, hr) in enumerate(zip(L2_rates, H1_rates)):
                if abs(lr - expected_L2_rate) > tol:
                    failures.append(
                        f"  Step {i+1}: L2_rate={lr:.2f}, expected {expected_L2_rate:.1f} "
                        f"(|diff|={abs(lr - expected_L2_rate):.2f} > {tol})"
                    )
                if abs(hr - expected_H1_rate) > tol:
                    failures.append(
                        f"  Step {i+1}: H1_rate={hr:.2f}, expected {expected_H1_rate:.1f} "
                        f"(|diff|={abs(hr - expected_H1_rate):.2f} > {tol})"
                    )

            if failures:
                print("FAIL")
                for msg in failures:
                    print(msg)
            else:
                print("PASS")


if __name__ == "__main__":
    main()
