#!/usr/bin/env python3
"""
Jackson point-charge benchmark at a planar dielectric interface.

Geometry:
  - Box: [-Lx/2, Lx/2] x [-Ly/2, Ly/2] x [-Lz/2, Lz/2]
  - Interface at z = 0
  - eps(z) = eps1_abs for z > 0
             eps2_abs for z < 0

Source:
  - Gaussian-regularized point charge at x0 = (0, 0, z0) with total charge q,
    width sigma:

      rho(x) = q / ((sigma*sqrt(2*pi))^3) * exp(-|x - x0|^2 / (2*sigma^2))

PDE:
  -div(eps(z) grad(phi)) = rho(x)  in Omega
   phi = phi_exact_Jackson        on ∂Omega

Analytic reference (Jackson):
  - Standard image-charge solution for a *delta* point charge above a planar
    dielectric interface (infinite half-spaces). We use this as phi_exact:

    For z > 0:
      phi = (1/(4*pi*eps1)) * [ q / R_real + q' / R_img ]
      q' = q (eps1 - eps2)/(eps1 + eps2)

    For z < 0:
      phi = (1/(4*pi*eps2)) * [ q'' / R_real ]
      q'' = q * 2*eps2/(eps1 + eps2)

Because the PDE uses a Gaussian rho while phi_exact assumes a delta source,
they disagree very close to the charge, but match well away from it. This is
the physics-oriented Jackson benchmark, not the manufactured-solution one.
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # vacuum permittivity (SI)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Jackson point-charge at planar dielectric interface."
    )
    p.add_argument("--Lx", type=float, default=1e-7, help="Box length in x [m]")
    p.add_argument("--Ly", type=float, default=1e-7, help="Box length in y [m]")
    p.add_argument("--Lz", type=float, default=1e-7, help="Box length in z [m]")
    p.add_argument("--h", type=float, default=8e-9, help="Target cell size [m]")

    p.add_argument("--eps1r", type=float, default=11.7,
                   help="Relative permittivity for z>0 (e.g. Si)")
    p.add_argument("--eps2r", type=float, default=3.9,
                   help="Relative permittivity for z<0 (e.g. oxide)")

    p.add_argument("--q", type=float, default=1.602176634e-19,
                   help="Point charge [C]")
    p.add_argument("--z0", type=float, default=1e-8,
                   help="z-position of the charge [m] ( > 0 )")
    p.add_argument("--sigma", type=float, default=5e-9,
                   help="Gaussian width sigma [m]")

    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for phi")
    p.add_argument("--outdir", type=str, default="results/jackson_pointcharge",
                   help="Output directory")
    p.add_argument("--basename", type=str, default="jackson_pc",
                   help="Base name for XDMF file")

    return p.parse_args()


# ---------- Mesh ----------

def build_box(Lx: float, Ly: float, Lz: float, h: float) -> mesh.Mesh:
    """
    Build a 3D box mesh over [-Lx/2,Lx/2] x [-Ly/2,Ly/2] x [-Lz/2,Lz/2]
    with approx cell size ~ h.
    """
    nx = max(2, int(np.round(Lx / h)))
    ny = max(2, int(np.round(Ly / h)))
    nz = max(2, int(np.round(Lz / h)))

    p0 = np.array([-Lx / 2, -Ly / 2, -Lz / 2], dtype=np.double)
    p1 = np.array([+Lx / 2, +Ly / 2, +Lz / 2], dtype=np.double)

    domain = mesh.create_box(
        COMM,
        points=[p0, p1],
        n=[nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


# ---------- Permittivity (DG0 for visualization) ----------

def make_eps_function(domain: mesh.Mesh, eps1_abs: float, eps2_abs: float) -> fem.Function:
    """
    DG0 function eps_fun(x) with a jump at z=0:

      eps_fun = eps1_abs for z>=0
              = eps2_abs for z<0
    """
    Veps = fem.functionspace(domain, ("DG", 0))
    eps_fun = fem.Function(Veps, name="eps")

    def eps_expr(x):
        Z = x[2]
        vals = np.where(Z >= 0.0, eps1_abs, eps2_abs)
        return vals

    eps_fun.interpolate(eps_expr)
    return eps_fun


# ---------- Gaussian "point" charge ----------

def make_rho_gaussian(V: fem.FunctionSpace,
                      q: float,
                      x0: np.ndarray,
                      sigma: float) -> fem.Function:
    """
    Build ρ(x) as a Gaussian centered at x0 with total charge q:

      ρ(x) = q / ((σ√(2π))^3) * exp(-|x - x0|^2 / (2σ^2))
    """
    rho = fem.Function(V, name="rho")

    x0 = np.asarray(x0, dtype=np.double)
    sigma = float(sigma)
    q = float(q)
    prefac = q / ((sigma * np.sqrt(2.0 * np.pi)) ** 3)

    def rho_fun(x):
        # x: shape (gdim, npoints)
        dx = x.T - x0  # (npoints, 3)
        r2 = np.sum(dx * dx, axis=1)
        values = prefac * np.exp(-0.5 * r2 / sigma**2)
        return values

    rho.interpolate(rho_fun)
    return rho


# ---------- Jackson analytic solution ----------

def phi_exact_callable(x: np.ndarray, params: dict) -> np.ndarray:
    """
    Jackson analytic solution for a point charge q at (0,0,z0)
    above a planar dielectric interface at z=0 between eps1 (z>0)
    and eps2 (z<0).

    Uses the standard image-charge construction:

      For z > 0:
        phi = (1/(4*pi*eps1)) [ q / R_real + q' / R_img ]
        q'  = q (eps1 - eps2)/(eps1 + eps2)

      For z < 0:
        phi = (1/(4*pi*eps2)) [ q'' / R_real ]
        q'' = q * 2*eps2/(eps1 + eps2)
    """
    q = params["q"]
    eps1 = params["eps1_abs"]
    eps2 = params["eps2_abs"]
    z0 = params["z0"]

    X, Y, Z = x  # each shape (npoints,)

    # Distances to real and image charges
    r_real = np.sqrt(X**2 + Y**2 + (Z - z0)**2)
    r_img = np.sqrt(X**2 + Y**2 + (Z + z0)**2)

    # Avoid division by zero exactly at the charge
    r_real = np.maximum(r_real, 1e-15)
    r_img = np.maximum(r_img, 1e-15)

    # Image charges
    q_img = q * (eps1 - eps2) / (eps1 + eps2)      # q'
    q2 = q * (2.0 * eps2) / (eps1 + eps2)          # q''

    four_pi = 4.0 * np.pi
    phi = np.zeros_like(X)

    mask_up = Z >= 0.0
    mask_lo = ~mask_up

    if np.any(mask_up):
        phi_up = (q / (four_pi * eps1 * r_real[mask_up])
                  + q_img / (four_pi * eps1 * r_img[mask_up]))
        phi[mask_up] = phi_up

    if np.any(mask_lo):
        phi_lo = q2 / (four_pi * eps2 * r_real[mask_lo])
        phi[mask_lo] = phi_lo

    return phi


# ---------- Main solve + error ----------

def main():
    args = parse_args()

    if RANK == 0:
        print("=== Jackson point-charge interface benchmark ===")
        print(f"Lx={args.Lx:.3e}, Ly={args.Ly:.3e}, Lz={args.Lz:.3e}, h~{args.h:.3e}")
        print(f"eps1r={args.eps1r}, eps2r={args.eps2r}, deg={args.deg}")
        print(f"q={args.q:.3e} C, z0={args.z0:.3e} m, sigma={args.sigma:.3e} m")
        print(f"outdir={args.outdir}")

    # Absolute permittivities
    eps1_abs = EPS0 * args.eps1r
    eps2_abs = EPS0 * args.eps2r

    # Build mesh
    domain = build_box(args.Lx, args.Ly, args.Lz, args.h)
    dim = domain.topology.dim
    assert dim == 3
    dx = ufl.dx(domain)

    # Function space for phi
    V = fem.functionspace(domain, ("Lagrange", args.deg))

    # Permittivity as DG0 function (for output)
    eps_fun = make_eps_function(domain, eps1_abs, eps2_abs)

    # Permittivity as UFL expression in the PDE
    x = ufl.SpatialCoordinate(domain)
    eps_ufl = ufl.conditional(
        ufl.ge(x[2], 0.0),
        eps1_abs,
        eps2_abs,
    )

    # Gaussian-regularized point charge
    x0 = np.array([0.0, 0.0, args.z0], dtype=np.double)
    rho = make_rho_gaussian(V, args.q, x0, args.sigma)

    # Analytic Jackson solution interpolated to V
    params = dict(
        q=args.q,
        eps1_abs=eps1_abs,
        eps2_abs=eps2_abs,
        z0=args.z0,
    )

    phi_exact = fem.Function(V, name="phi_exact")

    # Quick debug info
    if RANK == 0:
        print(f"phi_exact: min={phi_exact.x.array.min():.3e}, "
              f"max={phi_exact.x.array.max():.3e}")
        print(f"rho:       min={rho.x.array.min():.3e}, "
              f"max={rho.x.array.max():.3e}")

    # Weak form: -div(eps grad(phi)) = rho
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_ufl * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_form = rho * v * dx

    # Dirichlet BC: phi = phi_exact on entire boundary
    facet_indices = mesh.locate_entities_boundary(
        domain, dim - 1,
        lambda x_local: np.full(x_local.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, dim - 1, facet_indices)
    bc = fem.dirichletbc(phi_exact, bc_dofs)

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
        petsc_options_prefix="jackson_pc_",
    )

    phi = problem.solve()
    phi.name = "phi"

    if RANK == 0:
        try:
            its = problem.solver.getIterationNumber()
            print(f"KSP iterations: {its}")
        except Exception:
            pass

    # ----- Global relative L2 and H1 errors -----
    diff = phi - phi_exact

    form_num_L2 = fem.form(ufl.inner(diff, diff) * dx)
    form_den_L2 = fem.form(ufl.inner(phi_exact, phi_exact) * dx)

    num_L2 = fem.assemble_scalar(form_num_L2)
    den_L2 = fem.assemble_scalar(form_den_L2)
    num_L2 = COMM.allreduce(num_L2, op=MPI.SUM)
    den_L2 = COMM.allreduce(den_L2, op=MPI.SUM)

    rel_L2 = np.sqrt(num_L2) / np.sqrt(den_L2)

    form_num_H1 = fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * dx)
    form_den_H1 = fem.form(ufl.inner(ufl.grad(phi_exact), ufl.grad(phi_exact)) * dx)

    num_H1 = fem.assemble_scalar(form_num_H1)
    den_H1 = fem.assemble_scalar(form_den_H1)
    num_H1 = COMM.allreduce(num_H1, op=MPI.SUM)
    den_H1 = COMM.allreduce(den_H1, op=MPI.SUM)

    rel_H1 = np.sqrt(num_H1) / np.sqrt(den_H1)

    if RANK == 0:
        print(f"Relative L2 error  = {rel_L2:.3e}")
        print(f"Relative H1 error  = {rel_H1:.3e}")

    # ----- Pointwise relative error field (for visualization) -----
    err_rel = fem.Function(V, name="err_rel")
    num_vals = phi.x.array
    exact_vals = phi_exact.x.array
    phi_floor = 1e-6
    denom_vals = np.maximum(np.abs(exact_vals), phi_floor)
    err_vals = np.abs(num_vals - exact_vals) / denom_vals
    err_rel.x.array[:] = err_vals

    # ----- XDMF output -----
    outdir = Path(args.outdir)
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    xdmf_path = outdir / f"{args.basename}.xdmf"

    # CG1 view for XDMF (mesh geom is degree 1)
    V1 = fem.functionspace(domain, ("Lagrange", 1))

    def to_V1(f: fem.Function, name: str) -> fem.Function:
        f1 = fem.Function(V1, name=name)
        f1.interpolate(f)
        return f1

    phi1       = to_V1(phi, "phi")
    phi_exact1 = to_V1(phi_exact, "phi_exact")
    err_rel1   = to_V1(err_rel, "err_rel")
    rho1       = to_V1(rho, "rho")

    if RANK == 0:
        print(f"Writing XDMF to {xdmf_path}")

    with io.XDMFFile(COMM, xdmf_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi1)
        xdmf.write_function(phi_exact1)
        xdmf.write_function(err_rel1)
        xdmf.write_function(rho1)
        # eps_fun is DG0; write directly to preserve the hard jump
        xdmf.write_function(eps_fun)


if __name__ == "__main__":
    main()
