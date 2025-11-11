#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
import ufl
import matplotlib.pyplot as plt

from poisson.geom2d import rectangle
from poisson.materials import piecewise_eps_DG0_2regions
from poisson.charges import gaussian_rho
from poisson.fem_solve import assemble_solve_poisson
from poisson.io_utils import write_field_xdmf
from poisson.sample import interface_continuity_phi, interface_flux_jump

def main():
    ap = argparse.ArgumentParser(description="2D dielectric interface sanity check")
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--H", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=400)
    ap.add_argument("--ny", type=int, default=400)
    ap.add_argument("--eps1", type=float, default=11.7*8.854e-12) # Si
    ap.add_argument("--eps2", type=float, default=3.9*8.854e-12)  # SiO2
    ap.add_argument("--d", type=float, default=0.4)   # source height (y>0)
    ap.add_argument("--q", type=float, default=1.0e-12)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--delta", type=float, default=1e-3) # offset for derivative
    ap.add_argument("--out", type=str, default="out_2d.xdmf")
    ap.add_argument("--options_file", type=str, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    dom = rectangle(comm, args.L, args.H, args.nx, args.ny, cell_type="triangle")

    # Regions: y>0 => eps1, else eps2
    eps, DG0 = piecewise_eps_DG0_2regions(dom, lambda x: x[1] > 1e-14, args.eps1, args.eps2)

    V = fem.FunctionSpace(dom, ("CG", args.k))

    # Source: 2D Gaussian at (0,d)
    rho = gaussian_rho(dom, x0=np.array([0.0, args.d]), q=args.q, sigma=args.sigma)

    # Homogeneous Dirichlet on outer boundary
    tdim = dom.topology.dim
    def on_bdry(x):
        return np.isclose(x[0], -args.L) | np.isclose(x[0], args.L) | \
               np.isclose(x[1], -args.H) | np.isclose(x[1], args.H)
    facets = mesh.locate_entities_boundary(dom, tdim-1, on_bdry)
    dofs = fem.locate_dofs_topological(V, tdim-1, facets)
    bc = fem.dirichletbc(value=fem.Function(V), dofs=dofs)

    u, ksp = assemble_solve_poisson(dom, V, eps, rho, bcs=[bc], options_file=args.options_file)
    if comm.rank == 0:
        print(ksp.getConvergedReason(), "iterations:", ksp.getIterationNumber())

    write_field_xdmf(args.out, dom, u, eps)

    # Interface continuity φ(y=0+/-)
    xs, phi_top, phi_bot = interface_continuity_phi(u, axis=1, offset=args.delta, span=(-args.L, args.L), n=801)
    max_jump = np.max(np.abs(phi_top - phi_bot))
    if comm.rank == 0:
        print(f"max |phi(0+) - phi(0-)| = {max_jump:.3e}")

    # Normal flux: compare eps1 E1n vs eps2 E2n
    # We know which side is which, so we use constants:
    _, E1n, E2n = interface_flux_jump(u, eps, axis=1, delta=args.delta, span=(-args.L, args.L), n=801)
    flux_jump = np.max(np.abs(args.eps1*E1n - args.eps2*E2n))
    if comm.rank == 0:
        print(f"max |eps1 E1n - eps2 E2n| = {flux_jump:.3e}")

    if args.plot and comm.rank == 0:
        fig1 = plt.figure()
        plt.plot(xs, phi_top, label="phi(y=+δ)")
        plt.plot(xs, phi_bot, label="phi(y=-δ)", linestyle="--")
        plt.title("Interface continuity of φ")
        plt.xlabel("x"); plt.ylabel("φ"); plt.legend(); plt.tight_layout()

        fig2 = plt.figure()
        plt.plot(xs, args.eps1*E1n, label="eps1 E1n")
        plt.plot(xs, args.eps2*E2n, label="eps2 E2n", linestyle="--")
        plt.title("Normal flux continuity")
        plt.xlabel("x"); plt.ylabel("ε En"); plt.legend(); plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
