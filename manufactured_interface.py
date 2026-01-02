#!/usr/bin/env python3
"""
Manufactured-solution benchmark for a planar dielectric interface.

We prescribe an analytic potential phi_ms(x,y,z) that is continuous and has
continuous flux across z=0, with a jump in permittivity eps(z):

  eps(z) = eps1_abs for z > 0
         = eps2_abs for z < 0

We then form

  rho_ms(x) = - div( eps(z) * grad(phi_ms(x)) )

and solve:

  -div( eps(z) grad(phi_h) ) = rho_ms  in Omega
   phi_h = phi_ms               on ∂Omega

so that the exact solution of the PDE is phi_ms by construction.

We compare phi_h to phi_ms (phi_exact) in L2 and H1 and output a relative
error field for visualization.
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12  # vacuum permittivity


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manufactured-solution benchmark for planar dielectric interface."
    )
    p.add_argument("--Lx", type=float, default=1e-7,
                   help="Box length in x [m]")
    p.add_argument("--Ly", type=float, default=1e-7,
                   help="Box length in y [m]")
    p.add_argument("--Lz", type=float, default=1e-7,
                   help="Box length in z [m]")
    p.add_argument("--h", type=float, default=8e-9,
                   help="Target cell size [m]")
    p.add_argument("--eps1r", type=float, default=11.7,
                   help="Relative permittivity for z>0")
    p.add_argument("--eps2r", type=float, default=3.9,
                   help="Relative permittivity for z<0")
    p.add_argument("--deg", type=int, default=1,
                   help="CG polynomial degree for phi")
    p.add_argument("--outdir", type=str, default="results/manufactured_interface",
                   help="Output directory")
    p.add_argument("--basename", type=str, default="ms_interface",
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


# ---------- Permittivity (DG0) ----------

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


# ---------- Main solve + error ----------

def main():
    args = parse_args()

    if RANK == 0:
        print("=== Manufactured planar-interface benchmark ===")
        print(f"Lx={args.Lx:.3e}, Ly={args.Ly:.3e}, Lz={args.Lz:.3e}, h~{args.h:.3e}")
        print(f"eps1r={args.eps1r}, eps2r={args.eps2r}, deg={args.deg}")
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

    # Permittivity as UFL expression (for PDE)
    x = ufl.SpatialCoordinate(domain)
    eps_ufl = ufl.conditional(
        ufl.ge(x[2], 0.0),
        eps1_abs,
        eps2_abs,
    )

    # Manufactured analytic potential as UFL expression
    # s(x,y) = sin(pi x / Lx) * sin(pi y / Ly)
    s = ufl.sin(ufl.pi * x[0] / args.Lx) * ufl.sin(ufl.pi * x[1] / args.Ly)
    ratio = eps1_abs / eps2_abs
    phi_ms_ufl = ufl.conditional(
        ufl.ge(x[2], 0.0),
        s * x[2],
        s * (ratio * x[2]),
    )

    # Manufactured source: rho_ms = -div(eps * grad(phi_ms))
    rho_ms_ufl = -ufl.div(eps_ufl * ufl.grad(phi_ms_ufl))

    # Interpolate analytic phi_ms to V for comparison and BC
    phi_exact = fem.Function(V, name="phi_exact")
    phi_expr = fem.Expression(phi_ms_ufl, V.element.interpolation_points)
    phi_exact.interpolate(phi_expr)

    # Interpolate rho_ms_ufl to V so we can output it too
    rho_ms = fem.Function(V, name="rho_ms")
    rho_expr = fem.Expression(rho_ms_ufl, V.element.interpolation_points)
    rho_ms.interpolate(rho_expr)

    # Quick debug info on rank 0
    if RANK == 0:
        print(f"phi_exact: min={phi_exact.x.array.min():.3e}, max={phi_exact.x.array.max():.3e}")
        print(f"rho_ms:    min={rho_ms.x.array.min():.3e}, max={rho_ms.x.array.max():.3e}")

    # Weak form: -div(eps grad(phi)) = rho_ms_ufl
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_ufl * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_form = rho_ms_ufl * v * dx

    # Dirichlet BC: phi = phi_exact on the entire boundary
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
        petsc_options_prefix="ms_ifc_",
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
    den_L2 = COMM.allreduce(den_L2, op(MPI.SUM) if False else MPI.SUM)  # keep MPI.SUM explicit
    # Work around weird syntax in some editors
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

    # CG1 view for XDMF (mesh geometry degree = 1)
    V1 = fem.functionspace(domain, ("Lagrange", 1))

    def to_V1(f: fem.Function, name: str) -> fem.Function:
        f1 = fem.Function(V1, name=name)
        f1.interpolate(f)
        return f1

    phi1       = to_V1(phi, "phi")
    phi_exact1 = to_V1(phi_exact, "phi_exact")
    err_rel1   = to_V1(err_rel, "err_rel")
    rho1       = to_V1(rho_ms, "rho_ms")

    if RANK == 0:
        print(f"Writing XDMF to {xdmf_path}")

    with io.XDMFFile(COMM, xdmf_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi1)
        xdmf.write_function(phi_exact1)
        xdmf.write_function(err_rel1)
        xdmf.write_function(rho1)
        # eps_fun is DG0; we write it directly to preserve the hard jump
        xdmf.write_function(eps_fun)


if __name__ == "__main__":
    main()
