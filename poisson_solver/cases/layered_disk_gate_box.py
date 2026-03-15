from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def _flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


def _degree_compat(deg_or_callable) -> int:
    try:
        return int(deg_or_callable())
    except TypeError:
        return int(deg_or_callable)


def function_for_xdmf(phi: fem.Function, msh) -> fem.Function:
    mesh_deg = _degree_compat(msh.geometry.cmap.degree)
    sol_deg = _degree_compat(phi.function_space.ufl_element().degree)
    if sol_deg == mesh_deg:
        return phi

    Vout = fem.functionspace(msh, ("CG", mesh_deg))
    phi_out = fem.Function(Vout, name=phi.name)
    try:
        phi_out.interpolate(phi)
    except Exception:
        expr = fem.Expression(phi, Vout.element.interpolation_points())
        phi_out.interpolate(expr)
    return phi_out


def _global_minmax(phi: fem.Function):
    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)
    return gmin, gmax


def _parse_layers(layer_args):
    if not layer_args:
        return [
            (0.0, 50e-9, 13.05),
            (50e-9, 55e-9, 11.7),
            (55e-9, 255e-9, 13.05),
        ]

    layers = []
    for raw in layer_args:
        parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid --layer '{raw}'. Expected zmin,zmax,epsr")
        zmin, zmax, epsr = map(float, parts)
        layers.append((zmin, zmax, epsr))
    return layers


def _build_box_mesh(Lx, Ly, H, h, cell_type: str):
    x0 = -Lx / 2.0
    y0 = -Ly / 2.0

    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H / h)))

    if cell_type == "hex":
        ct = mesh.CellType.hexahedron
    elif cell_type == "tetra":
        ct = mesh.CellType.tetrahedron
    else:
        raise ValueError(f"Unsupported cell_type '{cell_type}'")

    msh = mesh.create_box(
        COMM,
        [
            np.array([x0, y0, 0.0], dtype=np.float64),
            np.array([x0 + Lx, y0 + Ly, H], dtype=np.float64),
        ],
        [nx, ny, nz],
        cell_type=ct,
    )
    return msh, nx, ny, nz, x0, y0


def _build_layered_epsilon_and_regions(msh, layers, default_epsr=None):
    eps0 = 8.8541878128e-12
    Q0 = fem.functionspace(msh, ("DG", 0))

    epsilon = fem.Function(Q0, name="epsilon")
    region_id = fem.Function(Q0, name="region_id")

    if default_epsr is None:
        epsilon.x.array[:] = 0.0
        region_id.x.array[:] = 0.0
    else:
        epsilon.x.array[:] = eps0 * float(default_epsr)
        region_id.x.array[:] = 1.0

    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    zc = x[:, 2]

    for i, (zmin, zmax, epsr) in enumerate(layers, start=1):
        mask = (zc >= float(zmin)) & (zc < float(zmax))
        epsilon.x.array[mask] = eps0 * float(epsr)
        region_id.x.array[mask] = float(i)

    return epsilon, region_id


def register_parser(subparsers) -> None:
    ap = subparsers.add_parser(
        "layered_disk_gate_box",
        help="Layered dielectric box with bottom Dirichlet and top disk-gate Dirichlet patch.",
    )
    ap.set_defaults(func=run)

    ap.add_argument("--outdir", type=str, default="results/layered_disk_gate_box")
    ap.add_argument("--basename", type=str, default="layered_disk_gate_box")

    ap.add_argument("--Lx", type=float, default=500e-9)
    ap.add_argument("--Ly", type=float, default=500e-9)
    ap.add_argument("--H", type=float, default=255e-9)

    ap.add_argument("--h", type=float, default=5e-9)
    ap.add_argument("--deg", type=int, default=1)
    ap.add_argument("--cell_type", choices=["hex", "tetra"], default="hex")

    ap.add_argument("--disk_xc", type=float, default=0.0)
    ap.add_argument("--disk_yc", type=float, default=0.0)
    ap.add_argument("--disk_R", type=float, default=10e-9)

    ap.add_argument("--bottom_voltage", type=float, default=12.0)
    ap.add_argument("--disk_voltage", type=float, default=1.0)

    ap.add_argument("--top_rest_bc", choices=["none", "dirichlet0"], default="none")
    ap.add_argument("--bc_sides", choices=["none", "dirichlet0"], default="none")

    ap.add_argument(
        "--layer",
        action="append",
        default=[],
        help="Repeatable dielectric layer as zmin,zmax,epsr",
    )

    ap.add_argument("--hard_exit", action="store_true")
    ap.add_argument("--skip_write", action="store_true")


def run(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    layers = _parse_layers(args.layer)

    _flush_print("[stage] build structured box mesh")
    msh, nx, ny, nz, x0, y0 = _build_box_mesh(
        Lx=args.Lx,
        Ly=args.Ly,
        H=args.H,
        h=args.h,
        cell_type=args.cell_type,
    )

    if RANK == 0:
        _flush_print("=== CASE: layered_disk_gate_box ===")
        _flush_print(f"box = {args.Lx:.3e} x {args.Ly:.3e} x {args.H:.3e} m")
        _flush_print(f"mesh = nx={nx}, ny={ny}, nz={nz}, cell_type={args.cell_type}, deg={args.deg}")
        _flush_print(f"disk = center=({args.disk_xc:.3e}, {args.disk_yc:.3e}), R={args.disk_R:.3e}")
        _flush_print(f"bottom_voltage = {args.bottom_voltage}")
        _flush_print(f"disk_voltage   = {args.disk_voltage}")
        _flush_print(f"top_rest_bc = {args.top_rest_bc}, bc_sides = {args.bc_sides}")
        _flush_print(f"layers = {layers}")

    _flush_print("[stage] build layered epsilon")
    epsilon, region_id = _build_layered_epsilon_and_regions(msh, layers)

    V = fem.functionspace(msh, ("CG", args.deg))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(epsilon * ufl.grad(u), ufl.grad(v)) * ufl.dx
    zero = fem.Constant(msh, PETSc.ScalarType(0.0))
    L = zero * v * ufl.dx

    tol = 1e-14
    bcs = []

    def on_bottom(X):
        return np.isclose(X[2], 0.0, atol=tol)

    def on_top(X):
        return np.isclose(X[2], args.H, atol=tol)

    def in_top_disk(X):
        r2 = (X[0] - args.disk_xc) ** 2 + (X[1] - args.disk_yc) ** 2
        return on_top(X) & (r2 <= args.disk_R ** 2 + tol)

    def on_top_rest(X):
        r2 = (X[0] - args.disk_xc) ** 2 + (X[1] - args.disk_yc) ** 2
        return on_top(X) & (r2 > args.disk_R ** 2 + tol)

    def on_sides(X):
        on_x0 = np.isclose(X[0], -args.Lx / 2.0, atol=tol)
        on_x1 = np.isclose(X[0],  args.Lx / 2.0, atol=tol)
        on_y0 = np.isclose(X[1], -args.Ly / 2.0, atol=tol)
        on_y1 = np.isclose(X[1],  args.Ly / 2.0, atol=tol)
        return on_x0 | on_x1 | on_y0 | on_y1

    _flush_print("[stage] build BCs")
    dofs_bottom = fem.locate_dofs_geometrical(V, on_bottom)
    bcs.append(fem.dirichletbc(PETSc.ScalarType(args.bottom_voltage), dofs_bottom, V))
    if RANK == 0:
        _flush_print(f"bottom dofs = {len(dofs_bottom)}")

    dofs_disk = fem.locate_dofs_geometrical(V, in_top_disk)
    bcs.append(fem.dirichletbc(PETSc.ScalarType(args.disk_voltage), dofs_disk, V))
    if RANK == 0:
        _flush_print(f"disk dofs = {len(dofs_disk)}")

    if args.top_rest_bc == "dirichlet0":
        dofs_top_rest = fem.locate_dofs_geometrical(V, on_top_rest)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs_top_rest, V))
        if RANK == 0:
            _flush_print(f"top-rest dofs = {len(dofs_top_rest)}")

    if args.bc_sides == "dirichlet0":
        dofs_sides = fem.locate_dofs_geometrical(V, on_sides)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs_sides, V))
        if RANK == 0:
            _flush_print(f"side dofs = {len(dofs_sides)}")

    _flush_print("[stage] solve")
    phi = fem.Function(V, name="phi")
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        u=phi,
        petsc_options_prefix="layered_disk_gate_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 20000,
        },
    )
    phi = problem.solve()
    phi.name = "phi"

    gmin, gmax = _global_minmax(phi)
    if RANK == 0:
        _flush_print(f"phi(min/max) = {gmin:.6e}, {gmax:.6e}")

    if not args.skip_write:
        _flush_print("[stage] function_for_xdmf")
        phi_out = function_for_xdmf(phi, msh)

        xdmf_path = outdir / f"{args.basename}.xdmf"
        _flush_print("[stage] write output")

        with io.XDMFFile(COMM, str(xdmf_path), "w", encoding=io.XDMFFile.Encoding.HDF5) as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(phi_out)
            xdmf.write_function(epsilon)
            xdmf.write_function(region_id)

        if RANK == 0:
            h5_path = outdir / f"{args.basename}.h5"
            _flush_print(f"Wrote: {xdmf_path}")
            _flush_print(f"Wrote: {h5_path}")

    if RANK == 0:
        _flush_print("[stage] finished main body")

    if args.hard_exit:
        _flush_print("[stage] barrier before hard exit")
        try:
            COMM.Barrier()
        except Exception:
            pass
        os._exit(0)
