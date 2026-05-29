"""
point_charge_sweep.py — sweep runner for point-charge-in-vacuum verification.

CLI args (lengths in nm, converted to metres internally):
  --L_nm, --h_fine_nm, --h_coarse_nm, --case_id, --out
"""

import argparse
import csv
import os
import time
import math
import numpy as np

from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem

from poisson.refinement import build_refined_mesh_3d, RefinementBox

COMM = MPI.COMM_WORLD


def _allreduce_sum(comm, local_val: int) -> int:
    arr = np.array(local_val, dtype=np.int64)
    result = np.zeros(1, dtype=np.int64)
    comm.Allreduce(arr, result, op=MPI.SUM)
    return int(result[0])


def _make_functionspace(msh, element):
    try:
        return fem.functionspace(msh, element)
    except AttributeError:
        return fem.FunctionSpace(msh, element)


def main():
    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Point-charge sweep for Poisson solver")
    parser.add_argument("--L_nm",        type=float, required=True)
    parser.add_argument("--h_fine_nm",   type=float, required=True)
    parser.add_argument("--h_coarse_nm", type=float, required=True)
    parser.add_argument("--case_id",     type=int,   default=0)
    parser.add_argument("--out",         type=str,   default="results/sweep.csv")
    args = parser.parse_args()

    nm = 1e-9
    L        = args.L_nm        * nm
    h_fine   = args.h_fine_nm   * nm
    h_coarse = args.h_coarse_nm * nm

    Lx = Ly = Lz = L
    charge_z = L / 2.0
    box_size = L / 5.0

    ref_box = RefinementBox(
        cx=0.0, cy=0.0, cz=charge_z,
        lx=box_size, ly=box_size, lz=box_size,
        h_fine=h_fine,
    )

    # ── Build mesh ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    msh = build_refined_mesh_3d(COMM, Lx, Ly, Lz, h_coarse, boxes=[ref_box], rank=0)
    t_mesh = time.perf_counter() - t0

    V = _make_functionspace(msh, ("Lagrange", 1))

    n_cells = _allreduce_sum(COMM, msh.topology.index_map(msh.topology.dim).size_local)
    n_dofs  = int(V.dofmap.index_map.size_global) * int(V.dofmap.index_map_bs)

    # ── Variational form ──────────────────────────────────────────────────────
    q_charge = 1.0
    eps0     = 1.0
    sigma    = 3.0 * h_fine

    x    = ufl.SpatialCoordinate(msh)
    r2   = x[0]**2 + x[1]**2 + (x[2] - charge_z)**2
    rho_expr = (q_charge / (math.sqrt(math.pi) * sigma)**3) * ufl.exp(-r2 / sigma**2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a     = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    Lform = (rho_expr / eps0) * v * ufl.dx

    # ── Boundary conditions ───────────────────────────────────────────────────
    atol = h_fine * 0.1

    def on_boundary(x):
        return (
            np.isclose(x[0], -Lx / 2, atol=atol) | np.isclose(x[0], Lx / 2, atol=atol) |
            np.isclose(x[1], -Ly / 2, atol=atol) | np.isclose(x[1], Ly / 2, atol=atol) |
            np.isclose(x[2],  0.0,    atol=atol) | np.isclose(x[2], Lz,     atol=atol)
        )

    def phi_exact_np(x):
        dist = np.sqrt(x[0]**2 + x[1]**2 + (x[2] - charge_z)**2)
        return (q_charge / (4.0 * math.pi * eps0)) / np.maximum(dist, 1e-30)

    boundary_facets = dmesh.locate_entities_boundary(msh, msh.topology.dim - 1, on_boundary)
    boundary_dofs   = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
    phi_bc = fem.Function(V)
    phi_bc.interpolate(phi_exact_np)
    bc = fem.dirichletbc(phi_bc, boundary_dofs)

    # ── Solve ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    petsc_opts = {"ksp_type": "cg", "ksp_rtol": 1e-10,
                  "pc_type": "hypre", "pc_hypre_type": "boomeramg"}
    try:
        problem = LinearProblem(a, Lform, bcs=[bc],
                                petsc_options=petsc_opts,
                                petsc_options_prefix="sweep_")
    except TypeError:
        problem = LinearProblem(a, Lform, bcs=[bc], petsc_options=petsc_opts)
    phi = problem.solve()
    t_solve = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    # ── Error at probe points ─────────────────────────────────────────────────
    r_probe = L / 3.0
    cz      = charge_z
    probe_pts = [
        ( r_probe, 0.0, cz), (-r_probe, 0.0, cz),
        (0.0,  r_probe, cz), (0.0, -r_probe, cz),
        (0.0, 0.0, cz + r_probe), (0.0, 0.0, cz - r_probe),
    ]
    phi_exact_probe = (q_charge / (4.0 * math.pi * eps0)) / r_probe

    mean_rel_err = float("nan")
    try:
        from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
        pts_array  = np.array(probe_pts, dtype=np.float64)
        tree       = bb_tree(msh, msh.topology.dim)
        collisions = compute_collisions_points(tree, pts_array)
        cells      = compute_colliding_cells(msh, collisions, pts_array)
        rel_errs   = []
        for i in range(len(probe_pts)):
            c_slice = cells.array[cells.offsets[i]: cells.offsets[i + 1]]
            if c_slice.size == 0:
                continue
            val = phi.eval(pts_array[i: i + 1], c_slice[:1])
            rel_errs.append(abs(float(val.flat[0]) - phi_exact_probe) / phi_exact_probe)
        if rel_errs:
            mean_rel_err = float(np.mean(rel_errs))
    except Exception:
        mean_rel_err = float("nan")

    # ── CSV output ────────────────────────────────────────────────────────────
    if COMM.rank == 0:
        csv_path = args.out
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        fieldnames = ["case_id", "L_nm", "h_fine_nm", "h_coarse_nm",
                      "n_cells", "n_dofs", "t_mesh_s", "t_solve_s", "t_total_s", "mean_rel_err"]
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "case_id": args.case_id, "L_nm": args.L_nm,
                "h_fine_nm": args.h_fine_nm, "h_coarse_nm": args.h_coarse_nm,
                "n_cells": n_cells, "n_dofs": n_dofs,
                "t_mesh_s": round(t_mesh, 4), "t_solve_s": round(t_solve, 4),
                "t_total_s": round(t_total, 4), "mean_rel_err": mean_rel_err,
            })

        err_str = f"{mean_rel_err:.3e}" if not math.isnan(mean_rel_err) else "nan"
        print(
            f"[case {args.case_id:02d}] L={args.L_nm:.0f}nm "
            f"h_fine={args.h_fine_nm:.0f}nm h_coarse={args.h_coarse_nm:.0f}nm "
            f"| cells={n_cells} dofs={n_dofs} "
            f"| mesh={t_mesh:.2f}s solve={t_solve:.2f}s total={t_total:.2f}s "
            f"| err={err_str}"
        )


if __name__ == "__main__":
    main()
