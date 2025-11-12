#!/usr/bin/env python3
# Solve -div(eps0 grad phi) = rho (Gaussian point charge) in a cube.
# Write phi, E=-grad(phi), D=eps0*E, epsilon_r to XDMF and compare to Coulomb.

from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=2.0, help="Half-size of box; domain=[-L,L]^3")
    p.add_argument("--nx", type=int, default=64, help="Cells per axis")
    p.add_argument("--q", type=float, default=1.0, help="Charge magnitude")
    p.add_argument("--eps0", type=float, default=1.0, help="Permittivity")
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--z0", type=float, default=0.2)
    p.add_argument("--sigma", type=float, default=0.05, help="Gaussian width")
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--out", type=str, default="results/coulomb_vacuum_compare.xdmf")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Mesh & spaces ---
    domain = mesh.create_box(
        comm,
        [np.array([-args.L, -args.L, -args.L]), np.array([args.L, args.L, args.L])],
        [args.nx, args.nx, args.nx],
        cell_type=mesh.CellType.hexahedron,
    )
    V  = fem.functionspace(domain, ("Lagrange", args.deg))
    Wv = fem.functionspace(domain, ("Lagrange", args.deg, (3,)))
    Q0 = fem.functionspace(domain, ("Discontinuous Lagrange", 0))  # for epsilon_r

    x = ufl.SpatialCoordinate(domain)
    r02 = (x[0]-args.x0)**2 + (x[1]-args.y0)**2 + (x[2]-args.z0)**2

    # --- Gaussian source (normalized to total charge q) ---
    rho_expr = args.q / ((np.sqrt(np.pi)*args.sigma)**3) * ufl.exp(-r02/args.sigma**2)

    # --- Variational problem ---
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = args.eps0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    Lform = rho_expr * v * ufl.dx

    # --- Dirichlet boundary: analytic Coulomb potential at outer boundary ---
    def on_boundary(X):
        return (np.isclose(X[0], -args.L) | np.isclose(X[0], args.L) |
                np.isclose(X[1], -args.L) | np.isclose(X[1], args.L) |
                np.isclose(X[2], -args.L) | np.isclose(X[2], args.L))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, on_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    def phi_exact_call(X):
        # X shape: (3, n)
        dx = X[0] - args.x0
        dy = X[1] - args.y0
        dz = X[2] - args.z0
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        r = np.maximum(r, 1e-12)
        return (args.q / (4.0*np.pi*args.eps0)) * (1.0 / r)

    phi_bc_fun = fem.Function(V)
    phi_bc_fun.interpolate(phi_exact_call)
    bcs = [fem.dirichletbc(phi_bc_fun, dofs)]
    # ---------- Analytic: Gaussian-smeared point charge in uniform medium ----------
    # rho(r) = Q/[(sqrt(pi)*sigma)^3] * exp(-r^2/sigma^2)
    # phi(r) = (Q/(4*pi*eps_abs)) * erf(r/sigma) / r,  phi(0) = (Q/(4*pi*eps_abs)) * 2/(sqrt(pi)*sigma)
    def phi_gauss_uniform(points_xyz, q, eps_abs, sigma):
        X = np.asarray(points_xyz)
        r = np.sqrt(np.sum(X**2, axis=1))
        # avoid division by zero with analytic r->0 limit
        phi = np.empty_like(r)
        mask = r > 1e-18
        from math import sqrt, pi
        import math
        # vectorized erf via scipy is absent; use math.erf per element
        # do elementwise to keep dependency-free
        for i, ri in enumerate(r):
            if ri > 1e-18:
                phi[i] = (q/(4.0*np.pi*eps_abs)) * math.erf(ri/sigma) / ri
            else:
                phi[i] = (q/(4.0*np.pi*eps_abs)) * (2.0/(np.sqrt(np.pi)*sigma))
        return phi

    def make_phi_analytic_function(domain, q, eps_abs, sigma):
        V = fem.functionspace(domain, ("Lagrange", 1))
        phiA = fem.Function(V, name="phi_analytic")
        def _phiA(X):
            # X is shape (3, N) per DOLFINx convention
            pts = np.column_stack((X[0], X[1], X[2]))
            return phi_gauss_uniform(pts, q, eps_abs, sigma)
        phiA.interpolate(_phiA)
        return phiA

    # --- Solve for phi ---
    problem = LinearProblem(
        a, Lform, bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre", "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-9, "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix="lp_",
    )
    phi = problem.solve(); phi.name = "phi"

    # --- E and D projections to CG1-vector ---
    w, zt = ufl.TrialFunction(Wv), ufl.TestFunction(Wv)
    mass = ufl.inner(w, zt) * ufl.dx

    E_sym = -ufl.grad(phi)
    E_rhs = ufl.inner(E_sym, zt) * ufl.dx
    E_fun = LinearProblem(mass, E_rhs,
                          petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
                          petsc_options_prefix="projE_").solve()
    E_fun.name = "E"

    D_sym = args.eps0 * E_sym
    D_rhs = ufl.inner(D_sym, zt) * ufl.dx
    D_fun = LinearProblem(mass, D_rhs,
                          petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
                          petsc_options_prefix="projD_").solve()
    D_fun.name = "D"

    # --- epsilon_r (DG0) for completeness (vacuum => 1 everywhere) ---
    epsr = fem.Function(Q0, name="epsilon_r")
    epsr.x.array[:] = 1.0

    # --- Global charge checks ---
    dx, ds = ufl.dx, ufl.ds
    rho_dg0 = fem.Function(Q0, name="rho")
    rho_dg0.interpolate(lambda X: args.q / ((np.sqrt(np.pi)*args.sigma)**3) *
                                  np.exp(-(((X[0]-args.x0)**2 + (X[1]-args.y0)**2 + (X[2]-args.z0)**2) /
                                           (args.sigma**2))))
    Q_vol  = fem.assemble_scalar(fem.form(rho_dg0 * dx))
    n = ufl.FacetNormal(domain)
    Q_flux = fem.assemble_scalar(fem.form(ufl.inner(D_fun, n) * ds))
    if comm.rank == 0:
        print(f"[CHECK] Total charge (volume integral of rho): {Q_vol:.8e}")
        print(f"[CHECK] Total charge (boundary flux of D·n):  {Q_flux:.8e}")
    def _coulomb_E_vec(X, q, r0, eps_abs, rmin=1e-12):
        """
        Vector Coulomb field at points X (shape (3,N)) from charge q at r0 (len 3).
        Returns array (N,3) with components (Ex, Ey, Ez).
        """
        dx = X[0] - r0[0]
        dy = X[1] - r0[1]
        dz = X[2] - r0[2]
        r2 = dx*dx + dy*dy + dz*dz
        r = np.sqrt(np.maximum(r2, rmin*rmin))
        coeff = q / (4.0*np.pi*eps_abs)
        Ex = coeff * dx / (r**3)
        Ey = coeff * dy / (r**3)
        Ez = coeff * dz / (r**3)
        return np.stack((Ex, Ey, Ez), axis=1)`


    # --- Coulomb reference samples along +z through the charge ---
    if comm.rank == 0:
        zs = np.linspace(args.z0 + 0.2, args.L, 6)  # away from the singularity
        for z in zs:
            r = abs(z - args.z0)
            phi_ref = args.q/(4*np.pi*args.eps0*r)
            E_ref   = args.q/(4*np.pi*args.eps0*r*r)
            print(f"[REF z={z:.3f}]  phi={phi_ref:.6f}   |E|={E_ref:.6f}")

    # --- Write XDMF (single file, multiple functions) ---
    with io.XDMFFile(domain.comm, str(out), "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(phi)
        xf.write_function(E_fun)
        xf.write_function(D_fun)
        xf.write_function(epsr)
        xf.write_function(rho_dg0)
    if comm.rank == 0:
        print(f"[WRITE] {out}")

if __name__ == "__main__":
    main()
