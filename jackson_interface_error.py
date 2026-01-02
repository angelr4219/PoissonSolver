#!/usr/bin/env python3
"""
Jackson planar dielectric interface benchmark with relative error heat map.

Problem:
  - Planar interface at z = 0
  - eps(z) = eps1 for z > 0, eps2 for z < 0
  - Single point charge q at (0,0,z0), with z0 > 0
  - PDE: -∇·(eps(z) ∇φ) = ρ   in box Ω
  - BC: φ = φ_exact (image-charge solution) on ∂Ω

This script:

  1. Builds the mesh (box).
  2. Solves the Poisson problem for φ_h.
  3. Evaluates the analytic Jackson solution φ_exact on the mesh.
  4. Computes relative L2 and H1(semi) errors vs φ_exact.
  5. Builds a pointwise relative error field:

       err_rel(x) = |φ_h - φ_exact| / max(|φ_exact|, φ_floor)

  6. Writes XDMF with φ, φ_exact, err_rel for ParaView.

Usage example (inside dolfinx Docker):

  /dolfinx-env/bin/python3 jackson_interface_error.py \\
    --h 8e-9 --deg 2 \\
    --eps1r 11.7 --eps2r 3.9 \\
    --q 1.602176634e-19 --z0 1e-8 --sigma 5e-9 \\
    --Lx 1e-7 --Ly 1e-7 --Lz 1e-7 \\
    --outdir results/jackson_single
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc  # ok to keep, even if not used directly
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
RANK = COMM.rank

# Vacuum permittivity
EPS0 = 8.8541878128e-12  # F/m


# ------------------------------------------------------------
# Analytic Jackson solution: image charges at planar interface
# ------------------------------------------------------------

def phi_exact_callable(x: np.ndarray, params: dict) -> np.ndarray:
    """
    Analytic potential for a point charge q at (0,0,z0) above a planar
    dielectric interface at z=0, following Jackson.

    eps1_abs: absolute permittivity in region z>0 (where the charge sits)
    eps2_abs: absolute permittivity in region z<0
    z0:      position of charge (z>0)

    In upper half-space z>0:
      φ1 = 1/(4π eps1) [ q / R_real + q' / R_image ]
      q' = q * (eps1 - eps2) / (eps1 + eps2)
      R_real  = |r - (0,0,z0)|
      R_image = |r - (0,0,-z0)|

    In lower half-space z<0:
      φ2 = 1/(4π eps2) [ q'' / R_real ]
      q'' = 2 eps2 / (eps1 + eps2) * q
    """
    q = params["q"]
    eps1 = params["eps1_abs"]
    eps2 = params["eps2_abs"]
    z0 = params["z0"]

    X, Y, Z = x  # each shape: (npoints,)

    # Distances to real and image charge
    r_real = np.sqrt(X**2 + Y**2 + (Z - z0)**2)
    r_img = np.sqrt(X**2 + Y**2 + (Z + z0)**2)

    # Avoid division by zero at the charge location
    r_real = np.maximum(r_real, 1e-15)
    r_img = np.maximum(r_img, 1e-15)

    # Image charges
    q_img = q * (eps1 - eps2) / (eps1 + eps2)
    q2 = 2.0 * eps2 / (eps1 + eps2) * q

    four_pi = 4.0 * np.pi
    phi = np.zeros_like(X)

    mask_up = Z >= 0.0
    mask_lo = ~mask_up

    # Upper half-space
    if np.any(mask_up):
        phi_up = (q / (four_pi * eps1 * r_real[mask_up])
                  + q_img / (four_pi * eps1 * r_img[mask_up]))
        phi[mask_up] = phi_up

    # Lower half-space
    if np.any(mask_lo):
        phi_lo = q2 / (four_pi * eps2 * r_real[mask_lo])
        phi[mask_lo] = phi_lo

    return phi


# ------------------------------------------------------------
# Mesh + coefficients + solve
# ------------------------------------------------------------

def build_box(Lx: float, Ly: float, Lz: float, h: float) -> mesh.Mesh:
    """
    Build a 3D box mesh over [-Lx/2, Lx/2] x [-Ly/2, Ly/2] x [-Lz/2, Lz/2]
    with approx cell size ~ h.
    """
    nx = max(4, int(round(Lx / h)))
    ny = max(4, int(round(Ly / h)))
    nz = max(4, int(round(Lz / h)))

    p0 = np.array([-Lx / 2, -Ly / 2, -Lz / 2], dtype=np.double)
    p1 = np.array([+Lx / 2, +Ly / 2, +Lz / 2], dtype=np.double)

    domain = mesh.create_box(
        COMM,
        points=[p0, p1],
        n=[nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


def make_eps_function(domain: mesh.Mesh,
                      eps1_abs: float,
                      eps2_abs: float) -> fem.Function:
    """
    Piecewise-constant eps(z): eps1_abs for z>0, eps2_abs for z<0.
    Represented as DG0 scalar function.
    """
    Veps = fem.functionspace(domain, ("DG", 0))
    eps_fun = fem.Function(Veps, name="eps")

    def eps_expr(x):
        z = x[2]
        vals = np.where(z >= 0.0, eps1_abs, eps2_abs)
        return vals

    eps_fun.interpolate(eps_expr)
    return eps_fun


def make_rho_function(domain: mesh.Mesh,
                      q: float,
                      z0: float,
                      sigma: float,
                      V: fem.FunctionSpace) -> fem.Function:
    """
    Gaussian-regularized point charge at (0,0,z0) with total charge q:

      ρ(x) = q / ((σ√(2π))^3) * exp(-|x - x0|^2 / (2σ^2))
    """
    x0 = np.array([0.0, 0.0, z0], dtype=np.double)
    prefac = q / ((sigma * np.sqrt(2.0 * np.pi)) ** 3)

    def rho_expr(x):
        # x: shape (gdim, npoints)
        dx = x.T - x0  # (npoints, 3)
        r2 = np.sum(dx * dx, axis=1)
        return prefac * np.exp(-0.5 * r2 / (sigma ** 2))

    rho = fem.Function(V, name="rho")
    rho.interpolate(rho_expr)
    return rho


def solve_jackson(domain: mesh.Mesh,
                  deg: int,
                  params: dict):
    """
    Solve Jackson planar interface Poisson on given domain, with CG degree 'deg'.

    Returns:
      V (FunctionSpace), phi (FEM solution), rho (source), eps_fun (DG0 permittivity)
    """
    dx = ufl.dx(domain)
    V = fem.functionspace(domain, ("Lagrange", deg))

    # Coefficients
    eps_fun = make_eps_function(domain, params["eps1_abs"], params["eps2_abs"])
    rho_fun = make_rho_function(domain, params["q"], params["z0"],
                                params["sigma"], V)

    # Trial/test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear + linear forms: -∇·(eps ∇u) = rho
    a = ufl.inner(eps_fun * ufl.grad(u), ufl.grad(v)) * dx
    L = rho_fun * v * dx

    # Dirichlet BCs: φ = φ_exact on the whole boundary
    def boundary_all(x):
        return np.full(x.shape[1], True, dtype=bool)

    facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, boundary_all)
    bcdofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)

    phi_exact_bc = fem.Function(V, name="phi_exact_bc")
    phi_exact_bc.interpolate(lambda x: phi_exact_callable(x, params))

    bc = fem.dirichletbc(phi_exact_bc, bcdofs)

    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
    }

    problem = LinearProblem(
        a, L,
        bcs=[bc],
        petsc_options=petsc_opts,
        petsc_options_prefix="jackson_single_",
    )

    phi = problem.solve()
    phi.name = "phi"

    if RANK == 0:
        try:
            its = problem.solver.getIterationNumber()
            print(f"KSP iterations = {its}")
        except Exception:
            pass

    return V, phi, rho_fun, eps_fun


# ------------------------------------------------------------
# Error calculation + relative error field
# ------------------------------------------------------------

def compute_relative_errors(domain: mesh.Mesh,
                            V: fem.FunctionSpace,
                            phi: fem.Function,
                            params: dict,
                            phi_floor: float = 1e-6):
    """
    Compute relative L2 and H1(semi) errors vs analytic solution,
    plus a pointwise relative error field.
    """
    comm = domain.comm
    dx = ufl.dx(domain)

    # Analytic solution in same space
    phi_exact = fem.Function(V, name="phi_exact")
    phi_exact.interpolate(lambda x: phi_exact_callable(x, params))

    diff = phi - phi_exact

    # L2
    form_num_L2 = fem.form(ufl.inner(diff, diff) * dx)
    form_den_L2 = fem.form(ufl.inner(phi_exact, phi_exact) * dx)

    num_L2 = fem.assemble_scalar(form_num_L2)
    den_L2 = fem.assemble_scalar(form_den_L2)
    num_L2 = comm.allreduce(num_L2, op=MPI.SUM)
    den_L2 = comm.allreduce(den_L2, op=MPI.SUM)

    eL2_rel = np.sqrt(num_L2) / np.sqrt(den_L2)

    # H1 seminorm
    form_num_H1 = fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * dx)
    form_den_H1 = fem.form(ufl.inner(ufl.grad(phi_exact), ufl.grad(phi_exact)) * dx)

    num_H1 = fem.assemble_scalar(form_num_H1)
    den_H1 = fem.assemble_scalar(form_den_H1)
    num_H1 = comm.allreduce(num_H1, op=MPI.SUM)
    den_H1 = comm.allreduce(den_H1, op=MPI.SUM)

    eH1_rel = np.sqrt(num_H1) / np.sqrt(den_H1)

    # Pointwise relative error field
    err_rel = fem.Function(V, name="err_rel")
    num_vals = phi.x.array
    exact_vals = phi_exact.x.array
    denom_vals = np.maximum(np.abs(exact_vals), phi_floor)
    err_rel.x.array[:] = np.abs(num_vals - exact_vals) / denom_vals

    return eL2_rel, eH1_rel, err_rel, phi_exact


# ------------------------------------------------------------
# XDMF output (phi, phi_exact, err_rel), using CG1 for output
# ------------------------------------------------------------

def write_xdmf(domain: mesh.Mesh,
               phi: fem.Function,
               phi_exact: fem.Function,
               err_rel: fem.Function,
               rho: fem.Function,
               eps_fun: fem.Function,
               outpath: Path):
    """
    Write mesh + fields to XDMF (ParaView-ready).

    Dolfinx requires the function degree to match the mesh geometry degree.
    The mesh geometry is degree 1, so for p>1 we interpolate all scalar
    fields down to CG1 for visualization only.
    """
    comm = domain.comm
    if comm.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)

    # CG1 scalar space for output
    V1 = fem.functionspace(domain, ("Lagrange", 1))

    def to_V1(f: fem.Function, name: str) -> fem.Function:
        expr = fem.Expression(f, V1.element.interpolation_points)
        f1 = fem.Function(V1, name=name)
        f1.interpolate(expr)
        return f1

    phi1 = to_V1(phi, "phi")
    phi_exact1 = to_V1(phi_exact, "phi_exact")
    err_rel1 = to_V1(err_rel, "err_rel")

    # For eps (DG0) and rho (already in V), we can also interpolate to V1
    eps1 = to_V1(eps_fun, "eps")
    rho1 = to_V1(rho, "rho")

    with io.XDMFFile(comm, outpath.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi1)
        xdmf.write_function(phi_exact1)
        xdmf.write_function(err_rel1)
        xdmf.write_function(rho1)
        xdmf.write_function(eps1)


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Jackson planar interface with relative error heat map."
    )
    parser.add_argument("--h", type=float, default=8e-9,
                        help="Target mesh size h.")
    parser.add_argument("--deg", type=int, default=1,
                        help="CG polynomial degree for φ.")
    parser.add_argument("--eps1r", type=float, default=11.7,
                        help="Relative permittivity above interface (z>0).")
    parser.add_argument("--eps2r", type=float, default=3.9,
                        help="Relative permittivity below interface (z<0).")
    parser.add_argument("--q", type=float, default=1.602176634e-19,
                        help="Point charge (C).")
    parser.add_argument("--z0", type=float, default=1e-8,
                        help="Charge z-position (m), must be > 0.")
    parser.add_argument("--sigma", type=float, default=5e-9,
                        help="Gaussian sigma for regularized charge (m).")
    parser.add_argument("--Lx", type=float, default=1e-7,
                        help="Domain length in x.")
    parser.add_argument("--Ly", type=float, default=1e-7,
                        help="Domain length in y.")
    parser.add_argument("--Lz", type=float, default=1e-7,
                        help="Domain length in z.")
    parser.add_argument("--phi-floor", type=float, default=1e-6,
                        help="Floor for |phi_exact| in relative error.")
    parser.add_argument("--outdir", type=str, default="results/jackson_single",
                        help="Output directory.")
    parser.add_argument("--basename", type=str, default="jackson_interface",
                        help="Base name for XDMF file (without extension).")

    args = parser.parse_args()

    eps1_abs = EPS0 * args.eps1r
    eps2_abs = EPS0 * args.eps2r

    params = dict(
        q=args.q,
        z0=args.z0,
        sigma=args.sigma,
        eps1_abs=eps1_abs,
        eps2_abs=eps2_abs,
    )

    if RANK == 0:
        print("=== Jackson planar interface (single run) ===")
        print(f"h    = {args.h:.3e}")
        print(f"deg  = {args.deg}")
        print(f"eps1r (z>0) = {args.eps1r}, eps2r (z<0) = {args.eps2r}")
        print(f"q = {args.q:.6e} C, z0 = {args.z0:.3e} m, sigma = {args.sigma:.3e} m")
        print(f"Box: Lx={args.Lx:.3e}, Ly={args.Ly:.3e}, Lz={args.Lz:.3e}")
        print(f"Output dir = {args.outdir}, basename = {args.basename}")

    # Build mesh
    domain = build_box(args.Lx, args.Ly, args.Lz, args.h)

    # Solve
    V, phi, rho_fun, eps_fun = solve_jackson(domain, args.deg, params)

    # Error vs analytic
    eL2_rel, eH1_rel, err_rel, phi_exact = compute_relative_errors(
        domain, V, phi, params, phi_floor=args.phi_floor
    )

    if RANK == 0:
        print(f"Relative L2 error (vs analytic) = {eL2_rel:.3e}")
        print(f"Relative H1 error (vs analytic) = {eH1_rel:.3e}")

    # Output
    outdir = Path(args.outdir)
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    xdmf_path = outdir / f"{args.basename}.xdmf"
    write_xdmf(domain, phi, phi_exact, err_rel, rho_fun, eps_fun, xdmf_path)

    if RANK == 0:
        print(f"Wrote XDMF to {xdmf_path}")


if __name__ == "__main__":
    main()

# --- Override write_xdmf: keep eps as DG0 (hard jump), only interpolate scalar fields ---
def write_xdmf(domain: mesh.Mesh,
               phi: fem.Function,
               phi_exact: fem.Function,
               err_rel: fem.Function,
               rho: fem.Function,
               eps_fun: fem.Function,
               outpath: Path):
    """
    Write mesh + fields to XDMF (ParaView-ready).

    - φ, φ_exact, err_rel, ρ: interpolated to CG1 for output (to match geometry degree).
    - eps_fun: written as DG0 (piecewise-constant), so the z=0 interface stays sharp.
    """
    comm = domain.comm
    if comm.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)

    # CG1 scalar space for output
    V1 = fem.functionspace(domain, ("Lagrange", 1))

    def to_V1(f: fem.Function, name: str) -> fem.Function:
        expr = fem.Expression(f, V1.element.interpolation_points)
        f1 = fem.Function(V1, name=name)
        f1.interpolate(expr)
        return f1

    phi1       = to_V1(phi, "phi")
    phi_exact1 = to_V1(phi_exact, "phi_exact")
    err_rel1   = to_V1(err_rel, "err_rel")
    rho1       = to_V1(rho, "rho")

    with io.XDMFFile(comm, outpath.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi1)
        xdmf.write_function(phi_exact1)
        xdmf.write_function(err_rel1)
        xdmf.write_function(rho1)
        # <-- key change: write eps_fun directly (DG0), no smoothing
        xdmf.write_function(eps_fun)
