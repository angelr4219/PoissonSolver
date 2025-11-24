#!/usr/bin/env python3
"""
Multiple point charges in a box (uniform permittivity) with selectable BC type.

PDE:

  -∇·(ε ∇φ) = ρ  in Ω = box

BC options:

  --bc dirichlet : φ = 0 on all faces (grounded box)
  --bc neumann   : ε ∂φ/∂n = 0 on all faces (natural), with ∫ρ dV = 0 enforced
                   and φ gauge fixed (if possible) to have zero mean.
  --bc periodic  : periodic in x using dolfinx_mpc (identify x=-L/2 with x=+L/2),
                   Dirichlet φ=0 on y,z faces. If dolfinx_mpc is not available,
                   falls back to Dirichlet on all faces with a warning.

ρ is a SUM of Gaussian-regularized point charges:

  ρ(x) = Σ_i q_i / ((σ_i√(2π))^3) * exp(-|x - x0_i|^2 / (2σ_i^2))

Each charge i has (q_i, x0_i, σ_i).

Ways to specify charges:

  1) Multi-charge interface (recommended):
     --charges "q,x,y,z,sigma; q,x,y,z,sigma; ..."

     Example (two charges, +e and -e):
       --charges "1.602e-19,0,0,0,5e-9;-1.602e-19,2e-8,0,0,5e-9"

  2) Single charge (backwards compatible with point_charge_box.py):
     --q 1.602176634e-19 --x0 "0,0,0" --sigma 5e-9

Each run writes to:

  <outdir>/run_<timestamp>_<bc_label>/

Files:
  - multi_point_charge_box.xdmf  (mesh + phi + E + rho [+ rho_rhs for Neumann])
  - params.json                  (all CLI args + bc_physics + charges + timing)
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

# ---------- Constants ----------
COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiple point charges in a box")

    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length (domain is [-L/2, L/2]^3) [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--epsr", type=float, default=1.0,
                   help="Relative permittivity ε_r (uniform)")
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
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for φ")
    p.add_argument("--bc", type=str, default="dirichlet",
                   help="BC type: 'dirichlet', 'neumann', or 'periodic'")
    p.add_argument("--outdir", type=str, default="results/multi_point_charge_box",
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


# ---------- Charge parsing & ρ construction ----------

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
        # Format: "q,x,y,z,sigma; q,x,y,z,sigma; ..."
        groups = [g.strip() for g in args.charges.split(";") if g.strip()]
        for g in groups:
            parts = [p.strip() for p in g.split(",")]
            if len(parts) != 5:
                raise ValueError(f"Invalid charge spec '{g}'; expected 5 comma-separated values: q,x,y,z,sigma")
            q = float(parts[0])
            x0 = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.double)
            sigma = float(parts[4])
            charges.append({"q": q, "x0": x0, "sigma": sigma})
    else:
        # Single-charge fallback
        x0 = np.array([float(s) for s in args.x0.split(",")], dtype=np.double)
        if x0.shape != (3,):
            raise ValueError("x0 must be 'x,y,z' with three components")
        charges.append({"q": args.q, "x0": x0, "sigma": args.sigma})

    if len(charges) == 0:
        raise ValueError("No charges specified.")

    return charges


def make_rho_function(charges, V: fem.FunctionSpace) -> fem.Function:
    """
    Build ρ(x) as a sum of Gaussians, one per charge:

      ρ_i(x) = q_i / ((σ_i√(2π))^3) * exp(-|x - x0_i|^2 / (2σ_i^2))

      ρ(x) = Σ_i ρ_i(x)
    """

    # Precompute prefactors and centers in arrays for speed
    qs = np.array([c["q"] for c in charges], dtype=np.double)
    sigmas = np.array([c["sigma"] for c in charges], dtype=np.double)
    x0s = np.array([c["x0"] for c in charges], dtype=np.double)  # shape (n,3)

    prefacs = qs / ((sigmas * np.sqrt(2.0 * np.pi)) ** 3)

    def rho_fun(x):
        # x: shape (gdim, npoints)
        npts = x.shape[1]
        values = np.zeros(npts, dtype=np.double)

        # For each charge, accumulate its Gaussian
        for i in range(len(charges)):
            dx = x.T - x0s[i]   # (npoints, 3)
            r2 = np.sum(dx * dx, axis=1)
            values += prefacs[i] * np.exp(-0.5 * r2 / (sigmas[i] ** 2))
        return values

    rho = fem.Function(V, name="rho")
    rho.interpolate(rho_fun)
    return rho


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
    assert dim == 3

    # Measure attached to this domain
    dx = ufl.dx(domain)

    # Function spaces
    V = fem.functionspace(domain, ("Lagrange", args.deg))
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0, (gdim,)))

    # Material: uniform ε
    eps = EPS0 * args.epsr

    # Parse charges
    charges = parse_charges(args)

    if RANK == 0:
        print("=== Multiple point charges in box ===")
        print(f"L      = {args.L:.3e} m, h ~ {args.h:.3e} m")
        print(f"epsr   = {args.epsr:.3g}")
        print(f"deg    = {args.deg}")
        print(f"Base outdir = {args.outdir}")
        print(f"Requested BC type: {bc_label}")
        print(f"HAS_MPC = {HAS_MPC}")
        print("Charges:")
        for i, c in enumerate(charges):
            print(f"  {i}: q = {c['q']:.6e} C, x0 = {c['x0']}, sigma = {c['sigma']:.3e} m")

    # Source ρ (sum of Gaussians)
    rho = make_rho_function(charges, V)

    # Total charge and volume (for Neumann)
    volume = fem.assemble_scalar(fem.form(1.0 * dx))
    total_q = fem.assemble_scalar(fem.form(rho * dx))
    if RANK == 0:
        print(f"Total charge from ρ: {total_q:.6e} C, volume = {volume:.6e} m^3")

    # Weak form base
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

    bcs = []
    petsc_opts = {}
    bc_physics = ""
    rho_rhs = None  # will hold neutralized source for Neumann only
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
        # Pure Neumann: ε ∂φ/∂n = 0 on all faces (natural BC).
        # For solvability, need ∫ρ dV = 0 → subtract mean charge density.
        rho_mean = total_q / volume
        rho_tilde_expr = rho - rho_mean  # UFL expression (ρ - <ρ>)

        if RANK == 0:
            print(f"Neumann: rho_mean = {rho_mean:.6e} C/m^3 (subtracted)")

        L_form = rho_tilde_expr * v * dx
        bcs = []  # no Dirichlet BCs
        bc_physics = "neumann_zero_flux_all_faces_charge_neutral_with_zero_mean_phi"

        # Use LU for robustness on Neumann system
        petsc_opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        # Make a visible Function for the *actual* RHS: ρ̃ = ρ - <ρ>
        rho_rhs = fem.Function(V, name="rho_rhs")
        rhs_interp = fem.Expression(
            rho_tilde_expr,
            V.element.interpolation_points
        )
        rho_rhs.interpolate(rhs_interp)

    # ---- Periodic in x (with MPC) ----
    elif bc_label == "periodic":
        tol = 10 * np.finfo(float).eps * max(1.0, abs(args.L))

        x_min = -args.L / 2.0
        x_max = +args.L / 2.0
        y_min = -args.L / 2.0
        y_max = +args.L / 2.0
        z_min = -args.L / 2.0
        z_max = +args.L / 2.0

        def periodic_boundary(x):
            # Slave side: x ≈ x_max
            return np.isclose(x[0], x_max, atol=tol)

        def periodic_relation(x):
            # Map x_max face back to x_min face: x' = x - L
            out = np.copy(x)
            out[0] -= args.L
            return out

        def dirichlet_boundary_pbc(x):
            # Dirichlet on y, z faces only; leave x faces for periodic coupling
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

    # Solve linear system (time this!)
    t_solve_start = perf_counter()
    if use_mpc:
        problem = MPCLinearProblem(
            a, L_form, mpc,
            bcs=bcs,
            petsc_options=petsc_opts,
            petsc_options_prefix="mpcbox_",
        )
    else:
        problem = FemLinearProblem(
            a, L_form,
            bcs=bcs,
            petsc_options=petsc_opts,
            petsc_options_prefix="mpcbox_",
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

    # For Neumann: try to gauge-fix φ to have zero mean safely
    if bc_label == "neumann":
        phi_mean = fem.assemble_scalar(fem.form(phi * dx)) / volume
        if RANK == 0:
            print(f"Neumann: computed mean(phi) = {phi_mean:.6e} V")

        if np.isfinite(phi_mean):
            phi.x.array[:] -= phi_mean
            if RANK == 0:
                print("Neumann: subtracted mean(phi) to enforce zero-mean gauge.")
        else:
            if RANK == 0:
                print("Neumann: mean(phi) is not finite; skipping gauge fix. "
                      "E-field (grad phi) is still physically meaningful.")

    # Electric field E = -∇φ projected to DG0 vector space
    E_expr = -ufl.grad(phi)
    E = fem.Function(W, name="E")

    interp_points = W.element.interpolation_points
    E_interpolator = fem.Expression(E_expr, interp_points)
    E.interpolate(E_interpolator)

    # ---------- Output: timestamped run dir + params ----------
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

    # Save parameters as JSON (rank 0)
    if RANK == 0:
        params = {
            "L": args.L,
            "h": args.h,
            "epsr": args.epsr,
            "deg": args.deg,
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

    # XDMF output
    xdmf_path = run_dir / "multi_point_charge_box.xdmf"
    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(E)
        xdmf.write_function(rho)
        if rho_rhs is not None:
            xdmf.write_function(rho_rhs)

    if RANK == 0:
        print(f"Wrote {xdmf_path}")


if __name__ == "__main__":
    main()
