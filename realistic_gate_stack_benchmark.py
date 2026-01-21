#!/usr/bin/env python3
import argparse
import math
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh
import ufl

# --- solver API compatibility ---
try:
    from dolfinx.fem.petsc import LinearProblem
except Exception as e:
    raise RuntimeError(
        "This DOLFINx build does not provide dolfinx.fem.petsc.LinearProblem.\n"
        "Run inside container:\n"
        "  python3 -c \"import dolfinx.fem.petsc as p; print(dir(p))\"\n"
        "and paste the output."
    ) from e


def g_uvz(u, v, z):
    denom = z * np.sqrt(u*u + v*v + z*z)
    return (1.0 / (2.0 * np.pi)) * np.arctan2(u * v, denom)


def phi_square_gates_row(x, y, z, a, xs, Vs):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    out = np.zeros_like(x, dtype=float)
    for xi, Vi in zip(xs, Vs):
        out -= Vi * (
            g_uvz(a - xi + x, a + y, z) +
            g_uvz(a - xi + x, a - y, z) +
            g_uvz(a + xi - x, a + y, z) +
            g_uvz(a + xi - x, a - y, z)
        )
    return out


def eval_function_on_points(u: fem.Function, points: np.ndarray, comm: MPI.Comm):
    from dolfinx import geometry

    msh = u.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, points)
    cells = geometry.compute_colliding_cells(msh, cand, points)

    vals_local = np.full((points.shape[0],), np.nan, dtype=np.float64)
    for i in range(points.shape[0]):
        cell_candidates = cells.links(i)
        if len(cell_candidates) > 0:
            cell = cell_candidates[0]
            v = np.zeros((1,), dtype=np.float64)
            u.eval(v, points[i:i+1], np.array([cell], dtype=np.int32))
            vals_local[i] = v[0]

    all_vals = comm.gather(vals_local, root=0)
    if comm.rank != 0:
        return None

    all_vals = np.stack(all_vals, axis=0)
    out = np.full((points.shape[0],), np.nan, dtype=np.float64)
    for j in range(points.shape[0]):
        col = all_vals[:, j]
        finite = col[np.isfinite(col)]
        out[j] = finite[0] if finite.size > 0 else np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="3D square-gate stacked box benchmark (NO gmsh/OCC).")
    parser.add_argument("--outdir", type=str, default="results/realistic_gates")
    parser.add_argument("--basename", type=str, default="gate_stack")

    # Geometry
    parser.add_argument("--Lx", type=float, default=300e-9)
    parser.add_argument("--Ly", type=float, default=300e-9)
    parser.add_argument("--H",  type=float, default=200e-9)
    parser.add_argument("--tox", type=float, default=0.0, help="oxide thickness for eps(z); does NOT create an interface mesh.")
    parser.add_argument("--a", type=float, default=35e-9, help="half-size of square gate. gate size = 2a.")
    parser.add_argument("--gate_xs", type=float, nargs="+", default=[-70e-9, 0.0, 70e-9])
    parser.add_argument("--gate_ys", type=float, nargs="+", default=[0.0, 0.0, 0.0])
    parser.add_argument("--gate_Vs", type=float, nargs="+", default=[0.25, 0.10, 0.25])

    # Mesh/FEM
    parser.add_argument("--h", type=float, default=10e-9)
    parser.add_argument("--deg", type=int, default=1)

    # Materials
    parser.add_argument("--eps_ox", type=float, default=3.9)
    parser.add_argument("--eps_si", type=float, default=11.7)

    # BC switches
    parser.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")
    parser.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")

    # Probe
    parser.add_argument("--probe_line", type=int, default=1)
    parser.add_argument("--probe_n", type=int, default=401)
    parser.add_argument("--probe_y", type=float, default=0.0)
    parser.add_argument("--probe_z", type=float, default=35e-9)

    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    os.makedirs(args.outdir, exist_ok=True)

    # ---- build box mesh ----
    Lx, Ly, H = args.Lx, args.Ly, args.H
    x0, y0 = -Lx/2.0, -Ly/2.0

    nx = max(2, int(math.ceil(Lx / args.h)))
    ny = max(2, int(math.ceil(Ly / args.h)))
    nz = max(2, int(math.ceil(H  / args.h)))

    domain = mesh.create_box(
        comm,
        [np.array([x0, y0, 0.0], dtype=np.float64),
         np.array([x0 + Lx, y0 + Ly, H], dtype=np.float64)],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron
    )

    V = fem.functionspace(domain, ("CG", args.deg))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ---- eps(z) on DG0 ----
    eps0 = 8.8541878128e-12
    Q = fem.functionspace(domain, ("DG", 0))
    eps = fem.Function(Q)
    tox = args.tox

    def eps_interp(x):
        z = x[2]
        if tox > 0:
            return np.where(z < tox, eps0 * args.eps_ox, eps0 * args.eps_si)
        else:
            return (eps0 * args.eps_si) * np.ones_like(z)

    eps.interpolate(eps_interp)

    a_form = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    # ---- Dirichlet BCs via coordinate tests ----
    xs = list(args.gate_xs)
    ys = list(args.gate_ys)
    Vs = list(args.gate_Vs)
    assert len(xs) == len(ys) == len(Vs)
    a = args.a
    tol = 1e-12

    def on_top(x):
        return np.isclose(x[2], 0.0, atol=1e-14)

    def in_gate_i(i):
        def marker(x):
            return (on_top(x) &
                    (np.abs(x[0] - xs[i]) <= a + tol) &
                    (np.abs(x[1] - ys[i]) <= a + tol))
        return marker

    def on_top_ground(x):
        if not np.all(on_top(x)):
            return np.zeros(x.shape[1], dtype=bool)
        inside_any = np.zeros(x.shape[1], dtype=bool)
        for i in range(len(xs)):
            inside_any |= ((np.abs(x[0] - xs[i]) <= a + tol) &
                           (np.abs(x[1] - ys[i]) <= a + tol))
        return on_top(x) & (~inside_any)

    def on_bottom(x):
        return np.isclose(x[2], H, atol=1e-14)

    def on_sides(x):
        on_x0 = np.isclose(x[0], x0, atol=1e-14)
        on_x1 = np.isclose(x[0], x0 + Lx, atol=1e-14)
        on_y0 = np.isclose(x[1], y0, atol=1e-14)
        on_y1 = np.isclose(x[1], y0 + Ly, atol=1e-14)
        return on_x0 | on_x1 | on_y0 | on_y1

    bcs = []
    for i, Vg in enumerate(Vs):
        dofs = fem.locate_dofs_geometrical(V, in_gate_i(i))
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    dofs = fem.locate_dofs_geometrical(V, on_top_ground)
    bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if args.bc_bottom == "dirichlet0":
        dofs = fem.locate_dofs_geometrical(V, on_bottom)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if args.bc_sides == "dirichlet0":
        dofs = fem.locate_dofs_geometrical(V, on_sides)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    # ---- solve ----
    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": "1.0e-10",
        "ksp_max_it": "20000",
    }
    problem = LinearProblem(a_form, L_form, bcs=bcs, petsc_options=petsc_opts, petsc_options_prefix="gate_stack_")
    uh = problem.solve()
    uh.name = "phi"

    local_min = np.min(uh.x.array) if uh.x.array.size else np.inf
    local_max = np.max(uh.x.array) if uh.x.array.size else -np.inf
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print("=== realistic_gate_stack_benchmark (no-gmsh) ===")
        print(f"Lx={Lx:.3e} Ly={Ly:.3e} H={H:.3e} tox={args.tox:.3e}")
        print(f"mesh: nx={nx} ny={ny} nz={nz}  (h~{args.h:.3e}), deg={args.deg}")
        print(f"eps_ox={args.eps_ox} eps_si={args.eps_si}")
        print(f"BC: top_ground=0, bottom={args.bc_bottom}, sides={args.bc_sides}")
        print(f"phi min={gmin:.6e}  max={gmax:.6e}")

    with io.XDMFFile(comm, os.path.join(args.outdir, f"{args.basename}.xdmf"), "w") as xdmf:
        xdmf.write_mesh(domain)

        # XDMF in this DOLFINx build requires function degree == mesh geometry degree.
        # Mesh geometry is degree-1, so write a CG1 copy for visualization.
        V1 = fem.functionspace(domain, ("CG", 1))
        uh1 = fem.Function(V1)
        uh1.name = "phi"
        uh1.interpolate(uh)
        xdmf.write_function(uh1)

        eps_out = fem.Function(Q)
        eps_out.x.array[:] = eps.x.array[:]
        eps_out.name = "eps_abs"
        xdmf.write_function(eps_out)

    if args.probe_line:
        y = args.probe_y
        z = args.probe_z
        xs_line = np.linspace(-Lx/2, Lx/2, args.probe_n)
        pts = np.column_stack([xs_line, y*np.ones_like(xs_line), z*np.ones_like(xs_line)]).astype(np.float64)

        vals = eval_function_on_points(uh, pts, comm)
        if comm.rank == 0:
            csv_path = os.path.join(args.outdir, f"{args.basename}_probe.csv")
            np.savetxt(csv_path, np.column_stack([xs_line, vals]), delimiter=",", header="x,phi_num", comments="")
            print(f"Wrote probe CSV: {csv_path}")

            if args.tox == 0.0:
                phi_an = phi_square_gates_row(xs_line, y, z, args.a, args.gate_xs, args.gate_Vs)
                pad = int(0.10 * args.probe_n)
                sl = slice(pad, -pad if pad > 0 else None)
                err = vals - phi_an
                maxerr = np.nanmax(np.abs(err[sl]))
                rms = math.sqrt(np.nanmean(err[sl]**2))
                print(f"Analytic comparison at y={y:.3e}, z={z:.3e} (middle 80% of line):")
                print(f"  max|err| = {maxerr:.6e}")
                print(f"  RMS err  = {rms:.6e}")

                csv_path2 = os.path.join(args.outdir, f"{args.basename}_probe_analytic.csv")
                np.savetxt(csv_path2, np.column_stack([xs_line, vals, phi_an, err]),
                           delimiter=",", header="x,phi_num,phi_analytic,err", comments="")
                print(f"Wrote probe+analytic CSV: {csv_path2}")

    if comm.rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
