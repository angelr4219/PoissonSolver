#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
from pathlib import Path
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=2.0)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--eps0", type=float, default=1.0)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--z0", type=float, default=0.2)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--out", type=str, default="results/coulomb_vacuum_compare.xdmf")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Mesh and spaces
    domain = mesh.create_box(
        comm,
        [np.array([-args.L, -args.L, -args.L]), np.array([args.L, args.L, args.L])],
        [args.nx, args.nx, args.nx],
        cell_type=mesh.CellType.hexahedron,
    )
    V  = fem.functionspace(domain, ("Lagrange", args.deg))
    Wv = fem.functionspace(domain, ("Lagrange", args.deg, (3,)))
    Q0 = fem.functionspace(domain, ("Discontinuous Lagrange", 0))

    x = ufl.SpatialCoordinate(domain)
    r02 = (x[0]-args.x0)**2 + (x[1]-args.y0)**2 + (x[2]-args.z0)**2

    # Gaussian source (total charge q)
    rho_expr = args.q / ((np.sqrt(np.pi)*args.sigma)**3) * ufl.exp(-r02/args.sigma**2)

    # Weak form: -div(eps0 grad phi) = rho
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = args.eps0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    Lform = rho_expr * v * ufl.dx

    # Dirichlet at outer boundary using Coulomb phi
    def on_bnd(X):
        return (np.isclose(X[0], -args.L) | np.isclose(X[0], args.L) |
                np.isclose(X[1], -args.L) | np.isclose(X[1], args.L) |
                np.isclose(X[2], -args.L) | np.isclose(X[2], args.L))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, on_bnd)
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    def phi_exact_call(X):
        dx = X[0] - args.x0
        dy = X[1] - args.y0
        dz = X[2] - args.z0
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        r = np.maximum(r, 1e-12)
        return (args.q / (4.0*np.pi*args.eps0)) * (1.0 / r)

    phi_bc_fun = fem.Function(V)
    phi_bc_fun.interpolate(phi_exact_call)
    bcs = [fem.dirichletbc(phi_bc_fun, dofs)]

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

    # Projections: E = -grad(phi), D = eps0 E
    w, zt = ufl.TrialFunction(Wv), ufl.TestFunction(Wv)
    mass = ufl.inner(w, zt) * ufl.dx

    E_rhs = ufl.inner(-ufl.grad(phi), zt) * ufl.dx
    E_fun = LinearProblem(mass, E_rhs,
                          petsc_options={"ksp_type":"cg","pc_type":"jacobi"},
                          petsc_options_prefix="projE_").solve()
    E_fun.name = "E"

    D_rhs = ufl.inner(args.eps0 * (-ufl.grad(phi)), zt) * ufl.dx
    D_fun = LinearProblem(mass, D_rhs,
                          petsc_options={"ksp_type":"cg","pc_type":"jacobi"},
                          petsc_options_prefix="projD_").solve()
    D_fun.name = "D"

    # epsilon_r (vacuum = 1)
    epsr = fem.Function(Q0, name="epsilon_r"); epsr.x.array[:] = 1.0

    # rho on DG0 for viz + charge checks
    rho_dg0 = fem.Function(Q0, name="rho")
    rho_dg0.interpolate(lambda X: args.q / ((np.sqrt(np.pi)*args.sigma)**3) *
                                  np.exp(-(((X[0]-args.x0)**2 + (X[1]-args.y0)**2 + (X[2]-args.z0)**2) /
                                           (args.sigma**2))))
    dx, ds = ufl.dx, ufl.ds
    Q_vol  = fem.assemble_scalar(fem.form(rho_dg0 * dx))
    n = ufl.FacetNormal(domain)
    Q_flux = fem.assemble_scalar(fem.form(ufl.inner(D_fun, n) * ds))
    if MPI.COMM_WORLD.rank == 0:
        print(f"[CHECK] Total charge (volume integral of rho): {Q_vol:.8e}")
        print(f"[CHECK] Total charge (boundary flux of D·n):  {Q_flux:.8e}")

    # Write all to one XDMF
    with io.XDMFFile(domain.comm, str(args.out), "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(phi)
        xf.write_function(E_fun)
        xf.write_function(D_fun)
        xf.write_function(epsr)
        xf.write_function(rho_dg0)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[WRITE] {args.out}")

if __name__ == "__main__":
    main()
