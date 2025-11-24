#!/usr/bin/env python3
"""
Point charge in a grounded box (uniform permittivity).

Solves: -∇·(ε ∇φ) = ρ  in Ω = box
        φ = 0 on ∂Ω

ρ is a Gaussian-regularized point charge at x0 with total charge q.

Outputs:
  - phi (scalar potential)
  - E   (electric field = -∇phi)

Usage (example):

  python point_charge_box.py \
    --L 1e-7 --h 5e-9 \
    --epsr 1.0 \
    --q 1.602176634e-19 \
    --x0 "0.0,0.0,0.0" \
    --sigma 5e-9 \
    --deg 1 \
    --outdir results/point_charge_box

"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# ---------- Constants ----------
COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point charge in a grounded box")

    p.add_argument("--L", type=float, default=1e-7,
                   help="Box side length (domain is [-L/2, L/2]^3) [m]")
    p.add_argument("--h", type=float, default=5e-9,
                   help="Target cell size [m]")
    p.add_argument("--epsr", type=float, default=1.0,
                   help="Relative permittivity ε_r (uniform)")
    p.add_argument("--q", type=float, default=1.602176634e-19,
                   help="Total charge [C] (default = +e)")
    p.add_argument("--x0", type=str, default="0.0,0.0,0.0",
                   help="Charge position 'x,y,z' in meters")
    p.add_argument("--sigma", type=float, default=5e-9,
                   help="Gaussian width σ [m]")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for φ")
    p.add_argument("--outdir", type=str, default="results/point_charge_box",
                   help="Output directory")

    return p.parse_args()


# ---------- Geometry & Mesh ----------

def build_box(L: float, h: float) -> mesh.Mesh:
    """
    Build a 3D box mesh over [-L/2, L/2]^3 with approx cell size ~ h.
    """
    # Number of cells per direction
    nx = max(2, int(np.round(L / h)))
    ny = nx
    nz = nx

    p0 = np.array([-L / 2, -L / 2, -L / 2], dtype=np.double)
    p1 = np.array([+L / 2, +L / 2, +L / 2], dtype=np.double)

    domain = mesh.create_box(
        COMM,
        points=[p0, p1],
        n=[nx, ny, nz],                 # <-- key change: use `n` instead of `cells`
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


# ---------- Source term: Gaussian "point" charge ----------

def make_rho_function(domain: mesh.Mesh,
                      q: float,
                      x0: np.ndarray,
                      sigma: float,
                      V: fem.FunctionSpace) -> fem.Function:
    """
    Build ρ(x) as a Gaussian centered at x0 with total charge q:

      ρ(x) = q / ((σ√(2π))^3) * exp(-|x - x0|^2 / (2σ^2))
    """
    x0 = np.asarray(x0, dtype=np.double)
    sigma = float(sigma)
    q = float(q)

    prefac = q / ((sigma * np.sqrt(2.0 * np.pi)) ** 3)

    def rho_fun(x):
        # x has shape (gdim, npoints)
        dx = x.T - x0  # (npoints, gdim)
        r2 = np.sum(dx * dx, axis=1)  # (npoints,)
        values = prefac * np.exp(-0.5 * r2 / sigma**2)
        return values

    rho = fem.Function(V, name="rho")
    rho.interpolate(rho_fun)

    # Optionally: check approximate total charge (debug)
    # total_q = COMM.allreduce(fem.assemble_scalar(fem.form(rho * ufl.dx(domain))), op=MPI.SUM)
    # if RANK == 0:
    #     print(f"Approx total charge from ρ: {total_q:.6e} C (target {q:.6e} C)")

    return rho


# ---------- Main solve ----------

def main():
    args = parse_args()

    if RANK == 0:
        print("=== Point charge in grounded box ===")
        print(f"L = {args.L:.3e} m, h ~ {args.h:.3e} m")
        print(f"epsr = {args.epsr:.3g}, q = {args.q:.6e} C")
        print(f"x0 = {args.x0}, sigma = {args.sigma:.3e} m")
        print(f"deg = {args.deg}, outdir = {args.outdir}")

    # Parse x0
    x0 = np.array([float(s) for s in args.x0.split(",")], dtype=np.double)
    if x0.shape != (3,):
        raise ValueError("x0 must be 'x,y,z' with three components")

    # Build mesh
    domain = build_box(args.L, args.h)
    dim = domain.topology.dim
    assert dim == 3

    # Measure (attach to this domain)
    dx = ufl.dx(domain)

    # Function spaces (use new dolfinx API: fem.functionspace)
    V = fem.functionspace(domain, ("CG", args.deg))

    # Vector DG0 space: encode vector shape in the element tuple
    gdim = domain.geometry.dim
    W = fem.functionspace(domain, ("DG", 0, gdim))

    # Material: uniform ε
    eps = EPS0 * args.epsr

    # Source ρ
    rho = make_rho_function(domain, args.q, x0, args.sigma, V)

    # Weak form: -∇·(ε∇φ) = ρ with φ = 0 on ∂Ω
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    # Dirichlet BC φ = 0 on entire boundary
    # Mark boundary facets via a simple "all facets" indicator
    facet_indices = mesh.locate_entities_boundary(
        domain, dim - 1,
        lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, dim - 1, facet_indices)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
    }

    problem = LinearProblem(
        a, L_form,
        bcs=[bc],
        petsc_options=petsc_opts,
        petsc_options_prefix="pcbox_",
    )


    phi = problem.solve()
    phi.name = "phi"

    if RANK == 0:
        its = problem.solver.getIterationNumber()
        print(f"KSP iterations: {its}")

    # Electric field E = -∇φ projected to DG0
    E_expr = -ufl.grad(phi)
    E = fem.Function(W, name="E")

    E_interpolator = fem.Expression(
        E_expr,
        W.element.interpolation_points()
    )
    E.interpolate(E_interpolator)

    # ---------- Output ----------
    outdir = Path(args.outdir)
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    COMM.barrier()

    xdmf_path = outdir / "point_charge_box.xdmf"
    with io.XDMFFile(COMM, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(E)
        xdmf.write_function(rho)

    if RANK == 0:
        print(f"Wrote {xdmf_path}")


if __name__ == "__main__":
    main()
