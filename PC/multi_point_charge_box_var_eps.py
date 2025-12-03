#!/usr/bin/env python3
"""
Multiple point charges in a box with variable permittivity ε(r).

PDE:

  -∇·(ε(r) ∇φ) = ρ  in Ω = box

with

  ε(r) = ε0 * eps_r(r),

and eps_r(r) defined as a DG0 field from **cell tags**:

  - Two regions split along z = split_z:
      region 0 (z < split_z): eps_r = epsr_bot
      region 1 (z >= split_z): eps_r = epsr_top

Charge model: sum of Gaussian-smeared point charges:

  ρ(r) = Σ_i q_i / ((σ_i √(2π))^3) * exp(-|r - x0_i|^2 / (2σ_i^2))

BC options:

  --bc dirichlet : φ = 0 on all faces
  --bc neumann   : natural Neumann (ε ∂φ/∂n = 0), with charge-neutralized RHS
  --bc periodic  : periodic in x using dolfinx_mpc if available, else fallback to full Dirichlet

Usage examples
--------------

1) Single charge, two-layer ε(r):

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 PC/multi_point_charge_box_var_eps.py \
      --L=1e-7 --h=5e-9 \
      --epsr-bot=3.9 --epsr-top=11.7 --split-z=0.0 \
      --q=1.602176634e-19 \
      --x0="0.0,0.0,0.0" \
      --sigma=5e-9 \
      --deg=1 \
      --bc=dirichlet \
      --outdir=results/multi_point_charge_box_var_eps

2) Two charges, same two-layer ε(r):

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 PC/multi_point_charge_box_var_eps.py \
      --L=1e-7 --h=5e-9 \
      --epsr-bot=3.9 --epsr-top=11.7 --split-z=0.0 \
      --charges="1.602e-19,-2e-8,0,0,5e-9;1.602e-19,2e-8,0,0,5e-9" \
      --deg=1 \
      --bc=dirichlet \
      --outdir=results/multi_point_charge_box_var_eps_dipole
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem as FemLinearProblem
import ufl

# Try to import dolfinx_mpc for periodic BCs
try:
    from dolfinx_mpc import LinearProblem as MPCLinearProblem, MultiPointConstraint
    HAS_MPC = True
except ImportError:
    HAS_MPC = False

COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiple point charges in a box with variable ε(r)")

    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length (domain is [-L/2, L/2]^3) [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for φ")

    # Permittivity: two regions split along z = split_z
    p.add_argument("--epsr-bot", type=float, default=1.0,
                   help="Relative permittivity below split_z (z < split_z)")
    p.add_argument("--epsr-top", type=float, default=1.0,
                   help="Relative permittivity above split_z (z >= split_z)")
    p.add_argument("--split-z", type=float, default=0.0,
                   help="z-plane where permittivity changes (global coords)")

    # Single-charge (fallback) interface
    p.add_argument("--q", type=float, default=1.602176634e-19,
                   help="Total charge for single-charge mode [C]")
    p.add_argument("--x0", type=str, default="0.0,0.0,0.0",
                   help="Single charge position 'x,y,z' in meters")
    p.add_argument("--sigma", type=float, default=5e-9,
                   help="Gaussian width σ for single-charge mode [m]")

    # Multi-charge interface
    p.add_argument(
        "--charges",
        type=str,
        default=None,
        help=("Semicolon-separated list of charges: "
              "'q,x,y,z,sigma; q,x,y,z,sigma; ...' (SI units)")
    )

    p.add_argument("--bc", type=str, default="dirichlet",
                   help="BC type: 'dirichlet', 'neumann', or 'periodic'")
    p.add_argument("--outdir", type=str, default="results/multi_point_charge_box_var_eps",
                   help="Base output directory")

    return p.parse_args()


# ---------- Geometry & Mesh ----------

def build_box(L: float, h: float) -> mesh.Mesh:
    """
    Build a 3D box mesh over [-L/2, L/2]^3 with approx cell size ~ h.
    """
    nx = max(2, int(np.round(L / h)))
    ny = nx
    nz = nx

    p0 = np.array([-L / 2, -L / 2, -L / 2], dtype=np.double)
    p1 = np.array([+L / 2, +L / 2, +L / 2], dtype=np.double)

    domain = mesh.create_box(
        COMM,
        points=[p0, p1],
        n=[nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


# ---------- Charges & ρ construction ----------

def parse_charges(args: argparse.Namespace):
    """
    Return a list of charges, each dict:
      {"q": float, "x0": np.array([x,y,z]), "sigma": float}

    Priority:
      - If args.charges is not None: parse multi-charge string
      - Else: use single-charge args.q, args.x0, args.sigma
    """
    charges = []

    if args.charges is not None:
        groups = [g.strip() for g in args.charges.split(";") if g.strip()]
        for g in groups:
            parts = [p.strip() for p in g.split(",")]
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid charge spec '{g}'; expected 5 comma-separated values: q,x,y,z,sigma"
                )
            q = float(parts[0])
            x0 = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.double)
            sigma = float(parts[4])
            charges.append({"q": q, "x0": x0, "sigma": sigma})
    else:
        x0 = np.array([float(s) for s in args.x0.split(",")], dtype=np.double)
        if x0.shape != (3,):
            raise ValueError("x0 must be 'x,y,z' with three components")
        charges.append({"q": args.q, "x0": x0, "sigma": args.sigma})

    if len(charges) == 0:
        raise ValueError("No charges specified.")

    return charges


def make_rho_function(charges, V: fem.FunctionSpace) -> fem.Function:
    """
    Build ρ(x) as a sum of Gaussians:

      ρ_i(x) = q_i / ((σ_i√(2π))^3) * exp(-|x - x0_i|^2 / (2σ_i^2))
      ρ(x)   = Σ_i ρ_i(x)
    """
    qs = np.array([c["q"] for c in charges], dtype=np.double)
    sigmas = np.array([c["sigma"] for c in charges], dtype=np.double)
    x0s = np.array([c["x0"] for c in charges], dtype=np.double)  # (n, 3)

    prefacs = qs / ((sigmas * np.sqrt(2.0 * np.pi)) ** 3)

    def rho_fun(x):
        # x: (gdim, npoints)
        npts = x.shape[1]
        values = np.zeros(npts, dtype=np.double)
        for i in range(len(charges)):
            dx = x.T - x0s[i]
            r2 = np.sum(dx * dx, axis=1)
            values += prefacs[i] * np.exp(-0.5 * r2 / (sigmas[i] ** 2))
        return values

    rho = fem.Function(V, name="rho")
    rho.interpolate(rho_fun)
    return rho


# ---------- Variable permittivity via cell tags + DG0 ----------

def build_eps_r(domain: mesh.Mesh, epsr_bot: float, epsr_top: float, split_z: float):
    """
    Build:

      - cell_tags (dim, indices, values): 0 = bottom (z < split_z), 1 = top (z >= split_z)
      - DG0 scalar function eps_r: epsr_bot in region 0, epsr_top in region 1
    """
    dim = domain.topology.dim
    gdim = domain.geometry.dim
    assert dim == 3 and gdim == 3

    # Locate cells by centroid z
    tol = 1e-12 * max(1.0, abs(split_z))

    def in_bot(x):
        return x[2] < split_z - tol

    def in_top(x):
        return x[2] > split_z + tol

    # Note: cells whose centroids are within ±tol of split_z will fall into whichever
    # inequality you decide; here we just allow the tiny band to go to "top" by default.
    cells_bot = mesh.locate_entities(domain, dim, in_bot)
    cells_top = mesh.locate_entities(domain, dim, in_top)

    # Any remaining cells (on the plane) -> top region
    num_cells = domain.topology.index_map(dim).size_local
    all_cells = np.arange(num_cells, dtype=np.int32)
    mask_assigned = np.zeros(num_cells, dtype=bool)
    mask_assigned[cells_bot] = True
    mask_assigned[cells_top] = True
    cells_plane = all_cells[~mask_assigned]
    if cells_plane.size > 0:
        if RANK == 0:
            print(f"[INFO] {cells_plane.size} cells lie near split_z; assigning to TOP region.")
        cells_top = np.concatenate([cells_top, cells_plane])

    indices = np.concatenate([cells_bot, cells_top])
    values = np.concatenate([
        np.full(cells_bot.size, 0, dtype=np.int32),
        np.full(cells_top.size, 1, dtype=np.int32),
    ])

    cell_tags = mesh.meshtags(domain, dim, indices, values)

    # DG0 for eps_r
    V_eps = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps_r = fem.Function(V_eps, name="epsr")

    # Initialize to epsr_bot and then overwrite top region
    eps_r.x.array[:] = epsr_bot
    dofs_top = fem.locate_dofs_topological(V_eps, dim, cells_top)
    eps_r.x.array[dofs_top] = epsr_top

    return cell_tags, eps_r


# ---------- Main solve ----------

def main():
    t_total_start = perf_counter()
    args = parse_args()

    bc_label = args.bc.lower()
    if bc_label not in ("dirichlet", "neumann", "periodic"):
        raise ValueError("bc must be 'dirichlet', 'neumann', or 'periodic'")

    # Build mesh
    domain = build_box(args.L, args.h)
    dim = domain.topology.dim
    gdim = domain.geometry.dim

    dx = ufl.dx(domain)

    # Function spaces
    V = fem.functionspace(domain, ("Lagrange", args.deg))
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0, (gdim,)))

    # Permittivity field from cell tags
    cell_tags, eps_r = build_eps_r(domain, args.epsr_bot, args.epsr_top, args.split_z)
    eps = EPS0 * eps_r  # UFL expression

    # Charges and ρ
    charges = parse_charges(args)
    rho = make_rho_function(charges, V)

    # Total charge and volume
    volume = fem.assemble_scalar(fem.form(1.0 * dx))
    total_q = fem.assemble_scalar(fem.form(rho * dx))
    total_q = COMM.allreduce(total_q, op=MPI.SUM)

    if RANK == 0:
        print("=== Multiple point charges with variable ε(r) ===")
        print(f"L           = {args.L:.3e} m, h ~ {args.h:.3e} m")
        print(f"deg         = {args.deg}")
        print(f"epsr_bot    = {args.epsr_bot:.3g}")
        print(f"epsr_top    = {args.epsr_top:.3g}")
        print(f"split_z     = {args.split_z:.3e} m")
        print(f"Base outdir = {args.outdir}")
        print(f"Requested BC type: {bc_label}")
        print(f"HAS_MPC     = {HAS_MPC}")
        print("Charges:")
        for i, c in enumerate(charges):
            print(f"  {i}: q = {c['q']:.6e} C, x0 = {c['x0']}, sigma = {c['sigma']:.3e} m")
        print(f"Total charge from ρ: {total_q:.6e} C, volume = {volume:.6e} m^3")

    # Weak form base: -∇·(ε(r) ∇φ) = ρ
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

    bcs = []
    petsc_opts = {}
    bc_physics = ""
    rho_rhs = None
    use_mpc = False
    mpc = None

    # ---- Dirichlet ----
    if bc_label == "dirichlet":
        bc_physics = "dirichlet_phi_0_on_boundary"
        L_form = rho * v * dx

        facet_indices = mesh.locate_entities_boundary(
            domain, dim - 1,
            lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        bc_dofs = fem.locate_dofs_topological(V, dim - 1, facet_indices)
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)
        bcs = [bc]

        petsc_opts = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
        }

    # ---- Neumann ----
    elif bc_label == "neumann":
        # Need ∫ρ dV = 0 for solvability → subtract mean of ρ
        rho_mean = total_q / volume
        rho_tilde_expr = rho - rho_mean

        if RANK == 0:
            print(f"Neumann: rho_mean = {rho_mean:.6e} C/m^3 (subtracted)")

        L_form = rho_tilde_expr * v * dx
        bcs = []
        bc_physics = "neumann_zero_flux_all_faces_charge_neutral_with_zero_mean_phi"

        petsc_opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        # Visible DG0-ish rhs for inspection: ρ̃ = ρ - <ρ>
        rho_rhs = fem.Function(V, name="rho_rhs")
        rhs_interp = fem.Expression(
            rho_tilde_expr,
            V.element.interpolation_points
        )
        rho_rhs.interpolate(rhs_interp)

    # ---- Periodic in x ----
    elif bc_label == "periodic":
        tol = 10 * np.finfo(float).eps * max(1.0, abs(args.L))

        x_min = -args.L / 2.0
        x_max = +args.L / 2.0
        y_min = -args.L / 2.0
        y_max = +args.L / 2.0
        z_min = -args.L / 2.0
        z_max = +args.L / 2.0

        def periodic_boundary(x):
            return np.isclose(x[0], x_max, atol=tol)

        def periodic_relation(x):
            out = np.copy(x)
            out[0] -= args.L
            return out

        def dirichlet_boundary_pbc(x):
            return np.logical_or.reduce([
                np.isclose(x[1], y_min, atol=tol),
                np.isclose(x[1], y_max, atol=tol),
                np.isclose(x[2], z_min, atol=tol),
                np.isclose(x[2], z_max, atol=tol),
            ])

        L_form = rho * v * dx

        if HAS_MPC:
            facets = mesh.locate_entities_boundary(domain, dim - 1, dirichlet_boundary_pbc)
            bc_dofs = fem.locate_dofs_topological(V, dim - 1, facets)
            bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)
            bcs = [bc]

            mpc = MultiPointConstraint(V)
            mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcs)
            mpc.finalize()
            use_mpc = True

            bc_physics = "periodic_in_x_dirichlet_on_yz"

            petsc_opts = {
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-14,
            }
        else:
            if RANK == 0:
                print("[WARN] dolfinx_mpc not available; falling back to Dirichlet on all faces.")
            bc_physics = "dirichlet_phi_0_on_boundary (periodic label, no MPC)"

            facets = mesh.locate_entities_boundary(
                domain, dim - 1,
                lambda x: np.full(x.shape[1], True, dtype=bool)
            )
            bc_dofs = fem.locate_dofs_topological(V, dim - 1, facets)
            bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)
            bcs = [bc]

            petsc_opts = {
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-14,
            }

    if RANK == 0:
        print(f"PDE BC physics: {bc_physics}")

    # Solve system
    t_solve_start = perf_counter()
    if use_mpc:
        problem = MPCLinearProblem(
            a, L_form, mpc,
            bcs=bcs,
            petsc_options=petsc_opts,
            petsc_options_prefix="mpcbox_eps_",
        )
    else:
        problem = FemLinearProblem(
            a, L_form,
            bcs=bcs,
            petsc_options=petsc_opts,
            petsc_options_prefix="mpcbox_eps_",
        )

    phi = problem.solve()
    t_solve_end = perf_counter()
    phi.name = "phi"
    solve_time = t_solve_end - t_solve_start

    if RANK == 0:
        try:
            its = problem.solver.getIterationNumber()
            print(f"KSP iterations: {its}")
        except Exception:
            pass
        print(f"Solve time (problem + solve): {solve_time:.4f} s")

    # Neumann: gauge-fix φ to zero mean if finite
    if bc_label == "neumann":
        phi_mean = fem.assemble_scalar(fem.form(phi * dx)) / volume
        phi_mean = COMM.allreduce(phi_mean, op=MPI.SUM)
        if RANK == 0:
            print(f"Neumann: computed mean(phi) = {phi_mean:.6e} V")
        if np.isfinite(phi_mean):
            phi.x.array[:] -= phi_mean
            if RANK == 0:
                print("Neumann: subtracted mean(phi) to enforce zero-mean gauge.")
        else:
            if RANK == 0:
                print("Neumann: mean(phi) non-finite; skipping gauge fix.")

    # E field = -∇φ projected to DG0 vector space
    E_expr = -ufl.grad(phi)
    E = fem.Function(W, name="E")
    interp_points = W.element.interpolation_points
    E_interpolator = fem.Expression(E_expr, interp_points)
    E.interpolate(E_interpolator)

    # ---------- Output ----------
    base_outdir = Path(args.outdir)
    if RANK == 0:
        base_outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_outdir / f"run_{timestamp}_{bc_label}"
    if RANK == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    t_total_end = perf_counter()
    total_time = t_total_end - t_total_start

    if RANK == 0:
        params = {
            "L": args.L,
            "h": args.h,
            "deg": args.deg,
            "epsr_bot": args.epsr_bot,
            "epsr_top": args.epsr_top,
            "split_z": args.split_z,
            "bc_label": bc_label,
            "bc_physics": bc_physics,
            "t_solve_sec": solve_time,
            "t_total_sec": total_time,
            "charges": [
                {
                    "q": float(c["q"]),
                    "x0": c["x0"].tolist(),
                    "sigma": float(c["sigma"]),
                }
                for c in charges
            ],
        }
        with (run_dir / "params.json").open("w") as f:
            json.dump(params, f, indent=2)
        print(f"Total runtime (main): {total_time:.4f} s")
        print(f"Wrote params.json in {run_dir}")

    xdmf_path = run_dir / "multi_point_charge_box_var_eps.xdmf"
    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_tags)
        xdmf.write_function(eps_r)
        xdmf.write_function(phi)
        xdmf.write_function(E)
        xdmf.write_function(rho)
        if rho_rhs is not None:
            xdmf.write_function(rho_rhs)

    if RANK == 0:
        print(f"Wrote {xdmf_path}")


if __name__ == "__main__":
    main()
def build_eps_r(domain: mesh.Mesh, epsr_bot: float, epsr_top: float, split_z: float):
    """
    Build:

      - cell_tags (dim, indices, values): 0 = bottom (z < split_z), 1 = top (z >= split_z)
      - DG0 scalar function eps_r: epsr_bot in region 0, epsr_top in region 1
    """
    dim = domain.topology.dim
    gdim = domain.geometry.dim
    assert dim == 3 and gdim == 3

    # IMPORTANT: ensure cell connectivity (dim->dim) exists for locate_dofs_topological
    domain.topology.create_connectivity(dim, dim)

    # Locate cells by centroid z
    tol = 1e-12 * max(1.0, abs(split_z))

    def in_bot(x):
        return x[2] < split_z - tol

    def in_top(x):
        return x[2] > split_z + tol

    cells_bot = mesh.locate_entities(domain, dim, in_bot)
    cells_top = mesh.locate_entities(domain, dim, in_top)

    # Any remaining cells (on the plane) -> top region
    num_cells = domain.topology.index_map(dim).size_local
    all_cells = np.arange(num_cells, dtype=np.int32)
    mask_assigned = np.zeros(num_cells, dtype=bool)
    mask_assigned[cells_bot] = True
    mask_assigned[cells_top] = True
    cells_plane = all_cells[~mask_assigned]
    if cells_plane.size > 0:
        if RANK == 0:
            print(f"[INFO] {cells_plane.size} cells lie near split_z; assigning to TOP region.")
        cells_top = np.concatenate([cells_top, cells_plane])

    indices = np.concatenate([cells_bot, cells_top])
    values = np.concatenate([
        np.full(cells_bot.size, 0, dtype=np.int32),
        np.full(cells_top.size, 1, dtype=np.int32),
    ])

    cell_tags = mesh.meshtags(domain, dim, indices, values)

    # DG0 for eps_r
    V_eps = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps_r = fem.Function(V_eps, name="epsr")

    # Initialize to epsr_bot and then overwrite top region
    eps_r.x.array[:] = epsr_bot
    dofs_top = fem.locate_dofs_topological(V_eps, dim, cells_top)
    eps_r.x.array[dofs_top] = epsr_top

    return cell_tags, eps_r
def build_eps_r(domain: mesh.Mesh, epsr_bot: float, epsr_top: float, split_z: float):
    """
    Build:

      - cell_tags (dim, indices, values): 0 = bottom (z < split_z), 1 = top (z >= split_z)
      - DG0 scalar function eps_r: epsr_bot in region 0, epsr_top in region 1

    For DG0, we avoid locate_dofs_topological and assign eps_r by
    directly indexing with cell indices (1 dof per cell).
    """
    dim = domain.topology.dim
    gdim = domain.geometry.dim
    assert dim == 3 and gdim == 3

    # Locate cells by centroid z
    tol = 1e-12 * max(1.0, abs(split_z))

    def in_bot(x):
        return x[2] < split_z - tol

    def in_top(x):
        return x[2] > split_z + tol

    cells_bot = mesh.locate_entities(domain, dim, in_bot)
    cells_top = mesh.locate_entities(domain, dim, in_top)

    # Any remaining cells (on the plane) -> top region
    num_cells = domain.topology.index_map(dim).size_local
    all_cells = np.arange(num_cells, dtype=np.int32)
    mask_assigned = np.zeros(num_cells, dtype=bool)
    mask_assigned[cells_bot] = True
    mask_assigned[cells_top] = True
    cells_plane = all_cells[~mask_assigned]
    if cells_plane.size > 0:
        if RANK == 0:
            print(f"[INFO] {cells_plane.size} cells lie near split_z; assigning to TOP region.")
        cells_top = np.concatenate([cells_top, cells_plane])

    indices = np.concatenate([cells_bot, cells_top])
    values = np.concatenate([
        np.full(cells_bot.size, 0, dtype=np.int32),
        np.full(cells_top.size, 1, dtype=np.int32),
    ])

    cell_tags = mesh.meshtags(domain, dim, indices, values)

    # DG0 for eps_r: 1 dof per cell, dof index == cell index in serial
    V_eps = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps_r = fem.Function(V_eps, name="epsr")

    # Initialize all cells to epsr_bot
    eps_r.x.array[:] = epsr_bot

    # Overwrite cells tagged as TOP (value == 1) to epsr_top
    mask_top = (cell_tags.values == 1)
    cells_top_only = cell_tags.indices[mask_top]
    eps_r.x.array[cells_top_only] = epsr_top

    return cell_tags, eps_r
