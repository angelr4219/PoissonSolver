from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
from petsc4py import PETSc
from dolfinx import fem

from ..analytic import eval_function_on_points, phi_square_gates_row
from ..common import COMM, RANK, global_minmax, make_function_space
from ..materials import build_piecewise_z_epsilon
from ..mesh_builders import build_box_mesh
from ..mpc import create_periodic_mpc
from ..output import function_for_xdmf, write_mesh_and_functions
from ..solver import solve_scalar_problem


def _parse_float_list(raw):
    out = []
    for token in raw:
        pieces = str(token).split(",")
        for piece in pieces:
            piece = piece.strip()
            if piece:
                out.append(float(piece))
    return out


def _flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


def register_parser(subparsers) -> None:
    ap = subparsers.add_parser(
        "square_gate_stack",
        help="3D square-gate stacked box benchmark without gmsh/OCC.",
    )
    ap.set_defaults(func=run)

    ap.add_argument("--outdir", type=str, default="results/realistic_gates")
    ap.add_argument("--basename", type=str, default="gate_stack")

    ap.add_argument("--Lx", type=float, default=300e-9)
    ap.add_argument("--Ly", type=float, default=300e-9)
    ap.add_argument("--H", type=float, default=200e-9)
    ap.add_argument("--tox", type=float, default=0.0)
    ap.add_argument("--a", type=float, default=35e-9)

    ap.add_argument("--gate_xs", type=str, nargs="+", default=["-70e-9", "0.0", "70e-9"])
    ap.add_argument("--gate_ys", type=str, nargs="+", default=["0.0", "0.0", "0.0"])
    ap.add_argument("--gate_Vs", type=str, nargs="+", default=["0.25", "0.10", "0.25"])

    ap.add_argument("--h", type=float, default=10e-9)
    ap.add_argument("--deg", type=int, default=1)

    ap.add_argument("--eps_ox", type=float, default=3.9)
    ap.add_argument("--eps_si", type=float, default=11.7)

    ap.add_argument(
        "--boundary_mode",
        choices=["dirichlet", "periodic_x", "periodic_y"],
        default="dirichlet",
    )
    ap.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")
    ap.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")

    ap.add_argument("--probe_line", type=int, default=1)
    ap.add_argument("--probe_n", type=int, default=401)
    ap.add_argument("--probe_y", type=float, default=0.0)
    ap.add_argument("--probe_z", type=float, default=35e-9)

    ap.add_argument("--hard_exit", action="store_true")
    ap.add_argument("--skip_write", action="store_true")
    ap.add_argument("--skip_probe", action="store_true")


def run(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xs = _parse_float_list(args.gate_xs)
    ys = _parse_float_list(args.gate_ys)
    Vs = _parse_float_list(args.gate_Vs)

    assert len(xs) == len(ys) == len(Vs), (
        f"gate_xs, gate_ys, gate_Vs must have the same length, got "
        f"{len(xs)}, {len(ys)}, {len(Vs)}"
    )

    _flush_print("[stage] build_box_mesh")
    domain, nx, ny, nz, x0, y0 = build_box_mesh(
        COMM, args.Lx, args.Ly, args.H, args.h
    )

    _flush_print("[stage] build function spaces and epsilon")
    V = make_function_space(domain, ("CG", args.deg))
    eps = build_piecewise_z_epsilon(domain, args.tox, args.eps_ox, args.eps_si)

    a = args.a
    tol = 1e-12
    H = args.H
    Lx = args.Lx
    Ly = args.Ly

    def on_top(x):
        return np.isclose(x[2], 0.0, atol=1e-14)

    def in_gate_i(i):
        def marker(x):
            return (
                on_top(x)
                & (np.abs(x[0] - xs[i]) <= a + tol)
                & (np.abs(x[1] - ys[i]) <= a + tol)
            )
        return marker

    def on_top_ground(x):
        inside_any = np.zeros(x.shape[1], dtype=bool)
        for i in range(len(xs)):
            inside_any |= (
                (np.abs(x[0] - xs[i]) <= a + tol)
                & (np.abs(x[1] - ys[i]) <= a + tol)
            )
        return on_top(x) & (~inside_any)

    def on_bottom(x):
        return np.isclose(x[2], H, atol=1e-14)

    def on_x_sides(x):
        on_x0 = np.isclose(x[0], x0, atol=1e-14)
        on_x1 = np.isclose(x[0], x0 + Lx, atol=1e-14)
        return on_x0 | on_x1

    def on_y_sides(x):
        on_y0 = np.isclose(x[1], y0, atol=1e-14)
        on_y1 = np.isclose(x[1], y0 + Ly, atol=1e-14)
        return on_y0 | on_y1

    _flush_print("[stage] build BCs")
    bcs = []
    for i, Vg in enumerate(Vs):
        dofs = fem.locate_dofs_geometrical(V, in_gate_i(i))
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    dofs = fem.locate_dofs_geometrical(V, on_top_ground)
    bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if args.bc_bottom == "dirichlet0":
        dofs = fem.locate_dofs_geometrical(V, on_bottom)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    mpc = None
    if args.boundary_mode == "dirichlet":
        if args.bc_sides == "dirichlet0":
            dofs = fem.locate_dofs_geometrical(V, lambda x: on_x_sides(x) | on_y_sides(x))
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))
    elif args.boundary_mode == "periodic_x":
        if args.bc_sides == "dirichlet0":
            dofs = fem.locate_dofs_geometrical(V, on_y_sides)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))
        mpc = create_periodic_mpc(V, "x", bcs=bcs)
    elif args.boundary_mode == "periodic_y":
        if args.bc_sides == "dirichlet0":
            dofs = fem.locate_dofs_geometrical(V, on_x_sides)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))
        mpc = create_periodic_mpc(V, "y", bcs=bcs)
    else:
        raise RuntimeError(f"Unhandled boundary mode: {args.boundary_mode}")

    _flush_print("[stage] solve")
    phi, _ksp = solve_scalar_problem(
        V,
        eps,
        bcs=bcs,
        rhs_expr=None,
        name="phi",
        petsc_options_prefix="gate_stack_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-10,
            "ksp_max_it": 20000,
        },
        mpc=mpc,
    )

    gmin, gmax = global_minmax(phi)

    if RANK == 0:
        _flush_print("=== CASE: square_gate_stack ===")
        _flush_print(f"Lx={Lx:.3e} Ly={Ly:.3e} H={H:.3e} tox={args.tox:.3e}")
        _flush_print(f"mesh: nx={nx} ny={ny} nz={nz}  (h~{args.h:.3e}), deg={args.deg}")
        _flush_print(f"eps_ox={args.eps_ox} eps_si={args.eps_si}")
        _flush_print(f"gate_xs={xs}")
        _flush_print(f"gate_ys={ys}")
        _flush_print(f"gate_Vs={Vs}")
        _flush_print(f"boundary_mode={args.boundary_mode}  bottom={args.bc_bottom}  sides={args.bc_sides}")
        _flush_print(f"phi min={gmin:.6e}  max={gmax:.6e}")

    x_out = outdir / f"{args.basename}.xdmf"

    if not args.skip_write:
        _flush_print("[stage] function_for_xdmf")
        phi_out = function_for_xdmf(phi, domain)

        _flush_print("[stage] write_mesh_and_functions")
        write_mesh_and_functions(COMM, domain, x_out, [phi_out, eps])

        if RANK == 0:
            _flush_print(f"Wrote: {x_out}")

    if args.probe_line and not args.skip_probe:
        _flush_print("[stage] probe evaluation")
        y = args.probe_y
        z = args.probe_z
        xs_line = np.linspace(-Lx / 2, Lx / 2, args.probe_n)
        pts = np.column_stack(
            [xs_line, y * np.ones_like(xs_line), z * np.ones_like(xs_line)]
        ).astype(np.float64)

        vals = eval_function_on_points(phi, pts, COMM)
        if RANK == 0:
            csv_path = outdir / f"{args.basename}_probe.csv"
            np.savetxt(
                csv_path,
                np.column_stack([xs_line, vals]),
                delimiter=",",
                header="x,phi_num",
                comments="",
            )
            _flush_print(f"Wrote probe CSV: {csv_path}")

            if args.tox == 0.0 and args.boundary_mode == "dirichlet":
                phi_an = phi_square_gates_row(xs_line, y, z, args.a, xs, Vs)
                pad = int(0.10 * args.probe_n)
                sl = slice(pad, -pad if pad > 0 else None)
                err = vals - phi_an
                maxerr = np.nanmax(np.abs(err[sl]))
                rms = math.sqrt(np.nanmean(err[sl] ** 2))
                _flush_print(f"Analytic comparison at y={y:.3e}, z={z:.3e} (middle 80% of line):")
                _flush_print(f"  max|err| = {maxerr:.6e}")
                _flush_print(f"  RMS err  = {rms:.6e}")

                csv_path2 = outdir / f"{args.basename}_probe_analytic.csv"
                np.savetxt(
                    csv_path2,
                    np.column_stack([xs_line, vals, phi_an, err]),
                    delimiter=",",
                    header="x,phi_num,phi_analytic,err",
                    comments="",
                )
                _flush_print(f"Wrote probe+analytic CSV: {csv_path2}")

    if RANK == 0:
        _flush_print("[stage] finished main body")

    if args.hard_exit:
        _flush_print("[stage] barrier before hard exit")
        try:
            COMM.Barrier()
        except Exception:
            pass
        os._exit(0)
