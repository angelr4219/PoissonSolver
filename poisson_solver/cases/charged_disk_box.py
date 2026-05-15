from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh

from ..analytic import eval_function_on_points
from ..common import COMM, RANK, global_minmax, make_function_space
from ..materials import build_multi_disk_box_fields
from ..mesh_builders import gmsh_build_box_with_refinement_field, read_gmsh_mesh
from ..mpc import create_periodic_mpc
from ..output import function_for_xdmf, write_mesh_and_functions, write_pre_fields
from ..solver import solve_scalar_problem


def _flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


def _parse_layers(layer_args):
    if not layer_args:
        return []
    layers = []
    for raw in layer_args:
        parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid --layer entry '{raw}'. Expected zmin,zmax,epsr")
        zmin, zmax, epsr = map(float, parts)
        layers.append((zmin, zmax, epsr))
    return layers


def _parse_disks(args):
    """
    New format:  --disk xc,yc,zc,R,t,Q
    Old fallback: --R --t --Q --z0 (single centered disk)
    """
    if args.disk:
        disks = []
        for raw in args.disk:
            parts = [p.strip() for p in str(raw).split(",")]
            if len(parts) != 6:
                raise ValueError(
                    f"Invalid --disk entry '{raw}'. Expected xc,yc,zc,R,t,Q"
                )
            xc, yc, zc, R, t, Q = map(float, parts)
            disks.append((xc, yc, zc, R, t, Q))
        return disks

    return [(0.0, 0.0, float(args.z0), float(args.R), float(args.t), float(args.Q))]


def _build_bcs(V, msh, args, _zmin, _zmax):
    """Return (bcs, mpc).  mpc is None for Dirichlet-only mode."""
    fdim = msh.topology.dim - 1
    periodic = args.periodic

    if periodic == "none":
        boundary_facets = mesh.locate_entities_boundary(
            msh, fdim, lambda X: np.full(X.shape[1], True, dtype=bool)
        )
        bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)]
        return bcs, None

    # Periodic in x, y, or xy: Dirichlet only on z-faces.
    tol = 1e-10 * max(1.0, abs(_zmin), abs(_zmax), args.Lz)

    def on_zfaces(X):
        return np.isclose(X[2], _zmin, atol=tol) | np.isclose(X[2], _zmax, atol=tol)

    bc_dofs = fem.locate_dofs_geometrical(V, on_zfaces)
    bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)]

    mpc = create_periodic_mpc(V, direction=periodic, bcs=bcs)
    return bcs, mpc


def _gauss_law_charge(msh, epsilon, phi) -> float:
    """
    Compute total charge via Gauss's law:
        Q = -∮_∂Ω ε ∇φ · n̂ dA
    For a well-resolved mesh this converges to Q_input.
    """
    n = ufl.FacetNormal(msh)
    form = fem.form(-epsilon * ufl.dot(ufl.grad(phi), n) * ufl.ds)
    q_local = float(fem.assemble_scalar(form))
    return COMM.allreduce(q_local, op=MPI.SUM)


def _probe_z(msh, phi, epsilon, rho, _zmin, _zmax, n_pts):
    """
    Evaluate phi, eps_r, and rho along the z-axis at (x=0, y=0).
    Returns a (n_pts, 4) array: [z, phi, eps_r, rho]  (rank 0 only, None elsewhere).
    """
    EPS0 = 8.8541878128e-12
    zs = np.linspace(_zmin, _zmax, n_pts)
    pts = np.column_stack([
        np.zeros(n_pts),
        np.zeros(n_pts),
        zs,
    ]).astype(np.float64)

    result_phi = eval_function_on_points(phi, pts, COMM)
    result_eps = eval_function_on_points(epsilon, pts, COMM)
    result_rho = eval_function_on_points(rho, pts, COMM)

    if RANK != 0:
        return None

    phi_vals, _ = result_phi
    eps_vals, _ = result_eps
    rho_vals, _ = result_rho

    eps_r = eps_vals / EPS0  # convert absolute → relative
    return np.column_stack([zs, phi_vals, eps_r, rho_vals])


def register_parser(subparsers) -> None:
    ap = subparsers.add_parser(
        "charged_disk_box",
        help="3D box with one or more charged disks and distance-based mesh refinement.",
    )
    ap.set_defaults(func=run)

    # --- disk specification ---
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--t", type=float, default=4e-9)
    ap.add_argument("--Q", type=float, default=1.602176634e-19)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument(
        "--disk", action="append", default=[],
        help="Charged disk as xc,yc,zc,R,t,Q. Repeatable.",
    )

    # --- output ---
    ap.add_argument("--out", type=str, default="disk_Q1e")

    # --- solver ---
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--epsr", type=float, default=12.0)

    # --- box ---
    ap.add_argument("--Lx", type=float, default=600e-9)
    ap.add_argument("--Ly", type=float, default=600e-9)
    ap.add_argument("--Lz", type=float, default=400e-9)
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)

    # --- mesh refinement ---
    ap.add_argument("--n_diam", type=int, default=28)
    ap.add_argument("--refine_band_R", type=float, default=8.0)
    ap.add_argument("--far_h_factor", type=float, default=25.0)

    # --- dielectric layers ---
    ap.add_argument(
        "--layer", action="append", default=[],
        help="Dielectric layer as zmin,zmax,epsr. Repeatable.",
    )

    # --- boundary conditions ---
    ap.add_argument(
        "--periodic", choices=["none", "x", "y", "xy"], default="none",
        help="Periodic BCs via dolfinx_mpc. Dirichlet applied only on z-faces.",
    )

    # --- probing & output control ---
    ap.add_argument("--probe_n", type=int, default=301,
                    help="Number of z-axis probe points for CSV output.")
    ap.add_argument("--gate_scaffold", action="store_true")
    ap.add_argument("--hard_exit", action="store_true")
    ap.add_argument("--skip_write", action="store_true")


def run(args) -> None:
    t_start = time.perf_counter()

    outdir = Path(args.out)
    msh_path = outdir / "mesh.msh"

    disks = _parse_disks(args)
    layers = _parse_layers(args.layer)

    ref_R = max(d[3] for d in disks)
    ref_z0 = disks[0][2]

    _zmin = float(args.zmin) if args.zmin is not None else -0.5 * args.Lz
    _zmax = float(args.zmax) if args.zmax is not None else +0.5 * args.Lz

    _flush_print("[stage] build mesh file")
    info = gmsh_build_box_with_refinement_field(
        comm=COMM,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        R=ref_R, z0=ref_z0,
        n_diam=args.n_diam,
        refine_band_R=args.refine_band_R,
        far_h_factor=args.far_h_factor,
        msh_path=msh_path,
        xmin=args.xmin, xmax=args.xmax,
        ymin=args.ymin, ymax=args.ymax,
        zmin=args.zmin, zmax=args.zmax,
    )

    if RANK == 0:
        _flush_print("=== CASE: charged_disk_box ===")
        _flush_print(f"periodic = {args.periodic}")
        _flush_print(f"number of disks = {len(disks)}")
        for i, (xc, yc, zc, R, t, Q) in enumerate(disks, start=1):
            _flush_print(
                f"  disk[{i}] xc={xc:.3e} yc={yc:.3e} zc={zc:.3e} "
                f"R={R:.3e} t={t:.3e} Q={Q:.3e}"
            )
        _flush_print(f"n_diam={args.n_diam} -> h_disk={info['h_disk']:.3e} m")
        _flush_print(f"h_far ={info['h_far']:.3e} m")
        _flush_print(
            f"box x:[{info['xmin']:.3e},{info['xmax']:.3e}] "
            f"y:[{info['ymin']:.3e},{info['ymax']:.3e}] "
            f"z:[{info['zmin']:.3e},{info['zmax']:.3e}]"
        )
        if layers:
            _flush_print(f"layers = {layers}")
        else:
            _flush_print(f"uniform epsr = {args.epsr}")

    _flush_print("[stage] read mesh")
    msh, cell_tags, facet_tags = read_gmsh_mesh(msh_path, COMM, gdim=3)

    _flush_print("[stage] build fields")
    rho, epsilon, region_id, inside, rho_total_assigned = build_multi_disk_box_fields(
        msh=msh,
        disks=disks,
        epsr_bulk=args.epsr,
        layers=layers if layers else None,
    )

    if not args.skip_write:
        _flush_print("[stage] write pre_fields")
        write_pre_fields(COMM, msh, outdir, rho, epsilon, region_id)

    V = make_function_space(msh, ("CG", args.p))

    _flush_print("[stage] build BCs")
    bcs, mpc = _build_bcs(V, msh, args, _zmin, _zmax)

    t_solve_start = time.perf_counter()
    _flush_print("[stage] solve")
    phi, _ksp = solve_scalar_problem(
        V, epsilon, bcs=bcs,
        rhs_expr=rho,
        name="phi_p",
        petsc_options_prefix="charged_disk_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 4000,
        },
        mpc=mpc,
    )
    t_solve = time.perf_counter() - t_solve_start

    gmin, gmax = global_minmax(phi)
    n_charged_local = int(np.count_nonzero(inside))
    n_charged_global = COMM.allreduce(n_charged_local, op=MPI.SUM)

    # --- Gauss law charge verification ---
    _flush_print("[stage] Gauss law charge check")
    Q_gauss = _gauss_law_charge(msh, epsilon, phi)
    Q_target = sum(d[5] for d in disks)
    charge_err_pct = abs(Q_gauss - Q_target) / abs(Q_target) * 100.0 if Q_target != 0 else 0.0

    if RANK == 0:
        _flush_print("=== Solve summary ===")
        _flush_print(f"solve_time   = {t_solve:.2f} s")
        _flush_print(f"p_degree     = {args.p}")
        _flush_print(f"charged_cells= {n_charged_global}")
        _flush_print(f"Q_target     = {Q_target:.6e} C")
        _flush_print(f"rho_assigned = {rho_total_assigned:.6e} C")
        _flush_print(f"Q_gauss      = {Q_gauss:.6e} C")
        _flush_print(f"charge_err   = {charge_err_pct:.4f} %")
        _flush_print(f"phi min/max  = {gmin:.6e} / {gmax:.6e} V")

    # --- Z-axis probe ---
    _flush_print("[stage] z-axis probe")
    probe_data = _probe_z(msh, phi, epsilon, rho, _zmin, _zmax, args.probe_n)

    # --- Write outputs ---
    if not args.skip_write:
        _flush_print("[stage] write mesh_fields")
        phi_out = function_for_xdmf(phi, msh)
        write_mesh_and_functions(
            COMM, msh, outdir / "mesh_fields.xdmf",
            [phi_out, rho, epsilon, region_id],
        )

    if RANK == 0:
        # Save z-probe CSV
        csv_path = outdir / "z_probe.csv"
        np.savetxt(
            csv_path,
            probe_data,
            delimiter=",",
            header="z_m,phi_V,eps_r,rho_Cm3",
            comments="",
        )
        _flush_print(f"Wrote probe CSV: {csv_path}")

        # Save stats JSON
        stats = {
            "periodic": args.periodic,
            "p_degree": args.p,
            "n_diam": args.n_diam,
            "disks": [
                {"xc": d[0], "yc": d[1], "zc": d[2],
                 "R": d[3], "t": d[4], "Q": d[5]}
                for d in disks
            ],
            "layers": [
                {"zmin": l[0], "zmax": l[1], "epsr": l[2]}
                for l in layers
            ],
            "Q_target_C": Q_target,
            "rho_assigned_C": rho_total_assigned,
            "Q_gauss_C": Q_gauss,
            "charge_err_pct": charge_err_pct,
            "phi_min_V": gmin,
            "phi_max_V": gmax,
            "n_charged_cells": n_charged_global,
            "solve_time_s": t_solve,
            "total_time_s": time.perf_counter() - t_start,
            "h_disk_m": info["h_disk"],
            "h_far_m": info["h_far"],
        }
        stats_path = outdir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        _flush_print(f"Wrote stats:    {stats_path}")

    if args.gate_scaffold and RANK == 0:
        _flush_print(f"[gate_id scaffold] facet_tags present: {facet_tags is not None}")

    if RANK == 0:
        _flush_print("[stage] done")

    if args.hard_exit:
        _flush_print("[stage] barrier → hard exit")
        try:
            COMM.Barrier()
        except Exception:
            pass
        os._exit(0)
