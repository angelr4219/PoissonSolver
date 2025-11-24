#!/usr/bin/env python3
"""
Multiple point charges in a box with variable permittivity ε(r), two layers in z.

PDE:
    -∇·(ε(r) ∇φ) = ρ

with
    ε(r) = ε0 * epsr(r),
    epsr(r) = epsr_bot for z < split_z,
              epsr_top for z >= split_z.

Charge model: sum of Gaussian-smeared point charges:
    ρ(r) = Σ_i q_i / ((σ_i sqrt(2π))^3) * exp(-|r - x0_i|^2 / (2 σ_i^2))

Boundary conditions: Dirichlet, φ = 0 on all faces.

Outputs:
    - phi      (CG)
    - E        (DG0 vector, approximate -∇φ)
    - rho      (CG)
    - epsr_dg0 (DG0 scalar, interpolated eps_r field)
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-point charge box with variable ε(z)")

    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length (domain is [-L/2, L/2]^3) [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for φ")

    # Permittivity profile: two layers in z
    p.add_argument("--epsr-bot", type=float, default=1.0,
                   help="Relative permittivity below split_z (z < split_z)")
    p.add_argument("--epsr-top", type=float, default=1.0,
                   help="Relative permittivity above split_z (z >= split_z)")
    p.add_argument("--split-z", type=float, default=0.0,
                   help="z-plane where permittivity changes (global coords)")

    # Single-charge mode
    p.add_argument("--q", type=float, default=1.602176634e-19,
                   help="Total charge [C] (single-charge mode)")
    p.add_argument("--x0", type=str, default="0.0,0.0,0.0",
                   help="Single charge position 'x,y,z' [m]")
    p.add_argument("--sigma", type=float, default=5e-9,
                   help="Gaussian width σ [m]")

    # Multi-charge mode
    p.add_argument(
        "--charges",
        type=str,
        default=None,
        help=("Semicolon-separated list of charges: "
              "'q,x,y,z,sigma; q,x,y,z,sigma; ...' (SI units)")
    )

    p.add_argument("--outdir", type=str, default="results/multi_point_charge_box_var_eps_simple",
                   help="Output directory")

    return p.parse_args()


# ---------- Geometry & Mesh ----------

def build_box(L: float, h: float) -> mesh.Mesh:
    nx = max(2, int(round(L / h)))
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
    Return a list of charges:
        {"q": float, "x0": np.array([x,y,z]), "sigma": float}
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
    ρ(x) = Σ_i q_i / ((σ_i√(2π))^3) * exp(-|x - x0_i|^2 / (2σ_i^2))
    """
    qs = np.array([c["q"] for c in charges], dtype=np.double)
    sigmas = np.array([c["sigma"] for c in charges], dtype=np.double)
    x0s = np.array([c["x0"] for c in charges], dtype=np.double)  # (n,3)

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


# ---------- Main solve ----------

def main():
    args = parse_args()
    t0 = perf_counter()

    domain = build_box(args.L, args.h)
    dim = domain.topology.dim
    gdim = domain.geometry.dim
    assert dim == 3 and gdim == 3

    dx = ufl.dx(domain)

    # Spaces
    V = fem.functionspace(domain, ("Lagrange", args.deg))                 # φ
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0, (gdim,))) # E (vector DG0)
    V_eps = fem.functionspace(domain, ("Discontinuous Lagrange", 0))      # epsr (DG0 scalar)

    # Charges + ρ
    charges = parse_charges(args)
    rho = make_rho_function(charges, V)

    # Total charge
    total_q = fem.assemble_scalar(fem.form(rho * dx))
    total_q = COMM.allreduce(total_q, op=MPI.SUM)

    # Variable epsr(z) as a UFL expression
    x = ufl.SpatialCoordinate(domain)
    epsr_expr = ufl.conditional(
        ufl.lt(x[2], args.split_z),
        args.epsr_bot,
        args.epsr_top,
    )
    eps_expr = EPS0 * epsr_expr

    # Interpolate epsr_expr to DG0 for visualization
    epsr_dg0 = fem.Function(V_eps, name="epsr")
    epsr_interpolator = fem.Expression(
        epsr_expr,
        V_eps.element.interpolation_points
    )
    epsr_dg0.interpolate(epsr_interpolator)

    if RANK == 0:
        print("=== Multi-point charge with variable ε(z) ===")
        print(f"L           = {args.L:.3e} m, h ~ {args.h:.3e} m")
        print(f"deg         = {args.deg}")
        print(f"epsr_bot    = {args.epsr_bot:.3g}")
        print(f"epsr_top    = {args.epsr_top:.3g}")
        print(f"split_z     = {args.split_z:.3e} m")
        print(f"outdir      = {args.outdir}")
        print("Charges:")
        for i, c in enumerate(charges):
            print(f"  {i}: q = {c['q']:.6e} C, x0 = {c['x0']}, sigma = {c['sigma']:.3e} m")
        print(f"Total charge from ρ: {total_q:.6e} C")

    # Weak form: -∇·(ε(z) ∇φ) = ρ, φ = 0 on boundary
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = eps_expr * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_form = rho * v * dx

    # Dirichlet BC on all faces
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

    problem = LinearProblem(
        a, L_form,
        bcs=bcs,
        petsc_options=petsc_opts,
        petsc_options_prefix="var_eps_simple_",
    )

    t_solve0 = perf_counter()
    phi = problem.solve()
    t_solve1 = perf_counter()
    phi.name = "phi"

    if RANK == 0:
        try:
            its = problem.solver.getIterationNumber()
            print(f"KSP iterations: {its}")
        except Exception:
            pass
        print(f"Solve time (problem + solve): {t_solve1 - t_solve0:.4f} s")

    # E field = -∇φ projected to DG0 vector space
    E_expr = -ufl.grad(phi)
    E = fem.Function(W, name="E")
    E_interp = fem.Expression(E_expr, W.element.interpolation_points)
    E.interpolate(E_interp)

    # Output
    outdir = Path(args.outdir)
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    xdmf_path = outdir / "multi_point_charge_box_var_eps_simple.xdmf"
    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(epsr_dg0)
        xdmf.write_function(phi)
        xdmf.write_function(E)
        xdmf.write_function(rho)

    t1 = perf_counter()
    if RANK == 0:
        print(f"Wrote {xdmf_path}")
        print(f"Total runtime: {t1 - t0:.4f} s")


if __name__ == "__main__":
    main()
