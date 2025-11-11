#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
import ufl
import matplotlib.pyplot as plt

from poisson.geom3d import box
from poisson.materials import piecewise_eps_DG0_2regions
from poisson.charges import gaussian_rho
from poisson.fem_solve import assemble_solve_poisson
from poisson.io_utils import write_field_xdmf
from poisson.analytics import phi_image_3d
from poisson.sample import evaluate_on_points, line_points

def main():
    ap = argparse.ArgumentParser(description="3D image-charge benchmark")
    ap.add_argument("--L", type=float, default=1.5)
    ap.add_argument("--H", type=float, default=1.5)
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nz", type=int, default=80)
    ap.add_argument("--eps1", type=float, default=11.7*8.854e-12)
    ap.add_argument("--eps2", type=float, default=3.9*8.854e-12)
    ap.add_argument("--d", type=float, default=0.4)
    ap.add_argument("--q", type=float, default=1.0e-9)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--out", type=str, default="out_3d.xdmf")
    ap.add_argument("--options_file", type=str, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    dom = box(comm, args.L, args.H, args.nx, args.ny, args.nz, cell_type="tet")

    # Regions: z>0 => eps1, else eps2
    eps, DG0 = piecewise_eps_DG0_2regions(dom, lambda x: x[2] > 1e-14, args.eps1, args.eps2)

    V = fem.FunctionSpace(dom, ("CG", args.k))

    # 3D Gaussian at (0,0,d)
    rho = gaussian_rho(dom, x0=np.array([0.0, 0.0, args.d]), q=args.q, sigma=args.sigma)

    # Dirichlet φ=0 on outer boundary
    tdim = dom.topology.dim
    def on_bdry(x):
        return np.isclose(x[0], -args.L) | np.isclose(x[0], args.L) | \
               np.isclose(x[1], -args.L) | np.isclose(x[1], args.L) | \
               np.isclose(x[2], -args.H) | np.isclose(x[2], args.H)
    facets = mesh.locate_entities_boundary(dom, tdim-1, on_bdry)
    dofs = fem.locate_dofs_topological(V, tdim-1, facets)
    bc = fem.dirichletbc(value=fem.Function(V), dofs=dofs)

    u, ksp = assemble_solve_poisson(dom, V, eps, rho, bcs=[bc], options_file=args.options_file)
    if comm.rank == 0:
        print(ksp.getConvergedReason(), "iterations:", ksp.getIterationNumber())

    write_field_xdmf(args.out, dom, u, eps)

    # Compare to analytic on several cuts:
    if comm.rank == 0:
        # Line z-axis through (0,0,z), avoid z=0 exactly
        pz = np.stack([np.zeros(401), np.zeros(401), np.linspace(-args.H*0.9, args.H*0.9, 401)], axis=1)
        # Line x-axis at y=0,z=d
        px = np.stack([np.linspace(-args.L*0.9, args.L*0.9, 401), np.zeros(401), np.full(401, args.d)], axis=1)
        # Interface line x-axis at z=0, y=0
        pi = np.stack([np.linspace(-args.L*0.9, args.L*0.9, 401), np.zeros(401), np.zeros(401)], axis=1)

        uhz = evaluate_on_points(u, pz)
        uhx = evaluate_on_points(u, px)
        uhi = evaluate_on_points(u, pi)

        phi_z = phi_image_3d(pz, args.q, args.d, args.eps1, args.eps2)
        phi_x = phi_image_3d(px, args.q, args.d, args.eps1, args.eps2)
        phi_i = phi_image_3d(pi, args.q, args.d, args.eps1, args.eps2)

        # Errors
        def errs(num, ref):
            L2 = np.sqrt(np.mean((num-ref)**2))
            Linf = np.max(np.abs(num-ref))
            return L2, Linf

        L2z, LInfz = errs(uhz, phi_z)
        L2x, LInfx = errs(uhx, phi_x)
        L2i, LInfi = errs(uhi, phi_i)
        print(f"z-axis   : L2={L2z:.3e}  Linf={LInfz:.3e}")
        print(f"x-axis@d : L2={L2x:.3e}  Linf={LInfx:.3e}")
        print(f"interface: L2={L2i:.3e}  Linf={LInfi:.3e}")

        if args.plot:
            import matplotlib.pyplot as plt
            xs = px[:,0]; zs = pz[:,2]; xi = pi[:,0]
            plt.figure(); plt.plot(zs, uhz, label="FEM"); plt.plot(zs, phi_z, "--", label="analytic")
            plt.xlabel("z"); plt.ylabel("phi"); plt.title("Linecut (0,0,z)"); plt.legend(); plt.tight_layout()

            plt.figure(); plt.plot(xs, uhx, label="FEM"); plt.plot(xs, phi_x, "--", label="analytic")
            plt.xlabel("x"); plt.ylabel("phi"); plt.title(f"Linecut (x,0,{args.d})"); plt.legend(); plt.tight_layout()

            plt.figure(); plt.plot(xi, uhi, label="FEM"); plt.plot(xi, phi_i, "--", label="analytic")
            plt.xlabel("x"); plt.ylabel("phi"); plt.title("Interface (x,0,0)"); plt.legend(); plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
