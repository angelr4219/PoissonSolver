#!/usr/bin/env python3
"""
Multiple point charges in a box with variable permittivity ε(z), up to N layers.

PDE:
    -∇·(ε(z) ∇φ) = ρ

with
    ε(z) = ε0 * epsr(z),

and epsr(z) defined as:
    - either a simple 2-layer profile using --epsr-bot, --epsr-top, --split-z
    - or an N-layer profile via --layers.

Layer specification (N-layer mode)
----------------------------------

Use:

  --layers="epsr1,zmax1; epsr2,zmax2; ...; epsrN,zmaxN"

This defines layers in z as:

  layer 1:  z ∈ [ -L/2,      zmax1 )
  layer 2:  z ∈ [ zmax1,     zmax2 )
  ...
  layer N:  z ∈ [ zmax{N-1}, zmaxN )

Typically you set zmaxN = +L/2 to cover the whole domain.

Example: 3 layers

  L = 1e-7  →  z ∈ [ -5e-8, 5e-8 ]
  --layers="3.9,-1e-8; 7.5,1e-8; 11.7,5e-8"

means

  layer 1: epsr = 3.9,  z ∈ [ -5e-8, -1e-8 )
  layer 2: epsr = 7.5,  z ∈ [ -1e-8,  1e-8 )
  layer 3: epsr = 11.7, z ∈ [  1e-8,  5e-8 )

If --layers is NOT provided, we fall back to a 2-layer profile:

  z < split_z      → epsr = epsr_bot
  z >= split_z     → epsr = epsr_top

Charge model: sum of Gaussian-smeared point charges:
    ρ(r) = Σ_i q_i / ((σ_i sqrt(2π))^3) * exp(-|r - x0_i|^2 / (2 σ_i^2))

Boundary conditions: Dirichlet, φ = 0 on all faces.

Outputs:
    - phi       (CG)
    - E         (DG0 vector, approximate -∇φ)
    - rho       (CG)
    - epsr_dg0  (DG0 scalar, epsr(z) sampled at cell centers)
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
    p = argparse.ArgumentParser(description="Multi-point charge box with N-layer ε(z)")

    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length (domain is [-L/2, L/2]^3) [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for φ")

    # 2-layer fallback profile
    p.add_argument("--epsr-bot", type=float, default=1.0,
                   help="Relative permittivity below split_z (z < split_z) if --layers is not used")
    p.add_argument("--epsr-top", type=float, default=1.0,
                   help="Relative permittivity above split_z (z >= split_z) if --layers is not used")
    p.add_argument("--split-z", type=float, default=0.0,
                   help="z-plane where 2-layer permittivity changes (used if --layers is omitted)")

    # N-layer profile
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help=("Optional N-layer spec: "
              "\"epsr1,zmax1; epsr2,zmax2; ...; epsrN,zmaxN\". "
              "zmax in meters, with zmaxN typically = L/2.")
    )

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

    p.add_argument("--outdir", type=str, default="results/multi_point_charge_box_var_eps_layers",
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


# ---------- Layers & εr(z) ----------

def parse_layers(L: float, args: argparse.Namespace):
    """
    Build a list of layer dicts:

        {"zmin": float, "zmax": float, "epsr": float}

    If args.layers is given, use N-layer mode with
        --layers="epsr1,zmax1; epsr2,zmax2; ...; epsrN,zmaxN"
    else, fall back to 2-layer profile.
    """
    z_lo = -L / 2.0
    z_hi = +L / 2.0
    layers = []

    if args.layers is None:
        # Simple 2-layer fallback
        layers.append({"zmin": z_lo,        "zmax": args.split_z, "epsr": args.epsr_bot})
        layers.append({"zmin": args.split_z, "zmax": z_hi,        "epsr": args.epsr_top})
        mode = "2-layer fallback (epsr_bot/epsr_top/split_z)"
    else:
        groups = [g.strip() for g in args.layers.split(";") if g.strip()]
        if len(groups) == 0:
            raise ValueError("Empty --layers string; either omit it or provide at least one layer spec.")
        zmin = z_lo
        for g in groups:
            parts = [p.strip() for p in g.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid layer spec '{g}'; expected 'epsr,zmax' (two comma-separated values)."
                )
            epsr = float(parts[0])
            zmax = float(parts[1])
            if zmax <= zmin:
                raise ValueError(
                    f"Layer zmax={zmax} must be > previous zmin={zmin}. "
                    "Ensure layers are ordered and non-overlapping."
                )
            layers.append({"zmin": zmin, "zmax": zmax, "epsr": epsr})
            zmin = zmax

        mode = "N-layer mode from --layers"
        # Warn if last zmax != +L/2
        tol = 1e-12 * max(1.0, abs(z_hi))
        if abs(zmin - z_hi) > tol and RANK == 0:
            print(
                f"[WARN] Last layer zmax={zmin:.3e} != L/2={z_hi:.3e}. "
                "Cells above last zmax will have epsr=0 (unphysical)."
            )

    return layers, mode


def build_epsr_expression_and_dg0(domain: mesh.Mesh,
                                  layers,
                                  V_eps: fem.FunctionSpace):
    """
    Given a list of layer dicts with fields zmin,zmax,epsr, build:

      - epsr_expr: UFL expression epsr(z) using nested conditionals
      - epsr_dg0: DG0 Function with epsr(z) interpolated at cell centers
    """
    x = ufl.SpatialCoordinate(domain)
    z = x[2]

    # Build piecewise epsr(z) via nested ufl.conditional
    # Start from 0.0 and layer-by-layer override
    epsr_expr = ufl.as_ufl(0.0)
    for layer in reversed(layers):
        zmin = layer["zmin"]
        zmax = layer["zmax"]
        epsr_val = layer["epsr"]

        cond = ufl.And(ufl.ge(z, zmin), ufl.lt(z, zmax))
        epsr_expr = ufl.conditional(cond, epsr_val, epsr_expr)

    epsr_dg0 = fem.Function(V_eps, name="epsr")
    epsr_interpolator = fem.Expression(
        epsr_expr,
        V_eps.element.interpolation_points
    )
    epsr_dg0.interpolate(epsr_interpolator)

    return epsr_expr, epsr_dg0


# ---------- Charges & ρ ----------

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
    V_eps = fem.functionspace(domain, ("Discontinuous Lagrange", 0))      # epsr(z) DG0

    # Layer structure and epsr(z)
    layers, mode = parse_layers(args.L, args)
    epsr_expr, epsr_dg0 = build_epsr_expression_and_dg0(domain, layers, V_eps)
    eps_expr = EPS0 * epsr_expr

    # Charges + ρ
    charges = parse_charges(args)
    rho = make_rho_function(charges, V)

    # Total charge
    total_q = fem.assemble_scalar(fem.form(rho * dx))
    total_q = COMM.allreduce(total_q, op=MPI.SUM)

    if RANK == 0:
        print("=== Multi-point charge with N-layer ε(z) ===")
        print(f"L           = {args.L:.3e} m, h ~ {args.h:.3e} m")
        print(f"deg         = {args.deg}")
        print(f"Layer mode  = {mode}")
        print("Layers (zmin, zmax, epsr):")
        for i, Lr in enumerate(layers):
            print(f"  {i}: [{Lr['zmin']:.3e}, {Lr['zmax']:.3e}) → epsr = {Lr['epsr']:.3g}")
        print(f"outdir      = {args.outdir}")
        print("Charges:")
        for i, c in enumerate(charges):
            print(f"  {i}: q = {c['q']:.6e} C, x0 = {c['x0']}, sigma = {c['sigma']:.3e} m")
        print(f"Total charge from ρ: {total_q:.6e} C")

    # Weak form: -∇·(ε(z) ∇φ) = ρ, φ = 0 on boundary (Dirichlet)
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
        petsc_options_prefix="var_eps_layers_",
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

    xdmf_path = outdir / "multi_point_charge_box_var_eps_layers.xdmf"
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
