# src/main.py
import argparse, os
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx import mesh as dmesh
from . import geometry as geo
from . import permittivity as pm
from . import poisson as ps

def build_parser():
    p = argparse.ArgumentParser(description="Poisson/Laplace driver")
    p.add_argument("--case", choices=["2d_dirichlet", "2d_mixed", "3d_dirichlet", "3d_mixed", "disk2d"], required=True)
    p.add_argument("--outfile", default="results/run")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=24)
    p.add_argument("--eps_lo", type=float, default=1.0)
    p.add_argument("--eps_hi", type=float, default=3.0)
    return p

def main():
    args = build_parser().parse_args()
    comm = MPI.COMM_WORLD

    # Build domain
    if args.case in ("2d_dirichlet", "2d_mixed"):
        domain = geo.unit_square(comm, args.nx, args.ny)
        V = fem.functionspace(domain, ("Lagrange", 1))
        x = ufl.SpatialCoordinate(domain)
        # DG0 epsilon by x<0.5
        eps = pm.dg0_from_indicator(domain, lambda X: X[0] < 0.5, args.eps_hi, args.eps_lo)
        # manufactured solution & RHS for verification
        pi = np.pi
        u_exact = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
        f = (eps) * (2.0*pi**2) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])

        # boundaries
        tdim, fdim = domain.topology.dim, domain.topology.dim-1
        facets = {
            "x0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 0.0)),
            "x1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 1.0)),
            "y0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 0.0)),
            "y1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 1.0)),
        }
        uD = fem.Function(V); uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
        bcs = [fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, facets[k])) for k in ("x0","x1","y0")]

        if args.case == "2d_dirichlet":
            uh = ps.solve_dirichlet(domain, V, eps, f, bcs, prefix="dir2d_")
        else:
            # Neumann on y1: g = (eps grad u_exact)·n, n = +ŷ
            n = ufl.as_vector((0.0, 1.0))
            g = ufl.dot((eps)*ufl.grad(u_exact), n)
            # make facet tags for ds
            import numpy as np
            indices = np.concatenate([facets["x0"], facets["x1"], facets["y0"], facets["y1"]])
            values  = np.concatenate([np.full_like(facets["x0"], 1),
                                      np.full_like(facets["x1"], 2),
                                      np.full_like(facets["y0"], 3),
                                      np.full_like(facets["y1"], 4)])
            order = np.argsort(indices)
            mt = dmesh.meshtags(domain, fdim, indices[order], values[order])
            uh = ps.solve_mixed(domain, V, eps, f, bcs, neumann_terms=[(g, 4)], facet_tags=mt, prefix="mix2d_")

        L2, H1s = ps.norms(domain, V, uh, u_exact)

    elif args.case in ("3d_dirichlet", "3d_mixed"):
        domain = geo.unit_cube(comm, n=(args.nz, args.nz, args.nz))
        V = fem.functionspace(domain, ("Lagrange", 1))
        x = ufl.SpatialCoordinate(domain)
        eps = pm.dg0_from_indicator(domain, lambda X: X[2] < 0.5, args.eps_lo, args.eps_hi)
        pi = np.pi
        u_exact = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * ufl.sin(pi*x[2])
        f = (eps) * (3.0*pi**2) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * ufl.sin(pi*x[2])
        tdim, fdim = domain.topology.dim, domain.topology.dim-1
        facets = {
            "x0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 0.0)),
            "x1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 1.0)),
            "y0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 0.0)),
            "y1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 1.0)),
            "z0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 0.0)),
            "z1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 1.0)),
        }
        uD = fem.Function(V); uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
        if args.case == "3d_dirichlet":
            # Dirichlet on all faces → pure Dirichlet verification
            bcs = [fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, facets[k]))
                   for k in ("x0","x1","y0","y1","z0","z1")]
            uh = ps.solve_dirichlet(domain, V, eps, f, bcs, prefix="dir3d_")
        else:
            # Dirichlet on all except z1; Neumann on z1
            bcs = [fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, facets[k]))
                   for k in ("x0","x1","y0","y1","z0")]
            n = ufl.as_vector((0.0,0.0,1.0))
            g = ufl.dot((eps)*ufl.grad(u_exact), n)
            import numpy as np
            indices = np.concatenate([facets["x0"], facets["x1"], facets["y0"], facets["y1"], facets["z0"], facets["z1"]])
            values  = np.concatenate([np.full_like(facets["x0"], 1),
                                      np.full_like(facets["x1"], 2),
                                      np.full_like(facets["y0"], 3),
                                      np.full_like(facets["y1"], 4),
                                      np.full_like(facets["z0"], 5),
                                      np.full_like(facets["z1"], 6)])
            order = np.argsort(indices)
            mt = dmesh.meshtags(domain, fdim, indices[order], values[order])
            uh = ps.solve_mixed(domain, V, eps, f, bcs, neumann_terms=[(g, 6)], facet_tags=mt, prefix="mix3d_")
        L2, H1s = ps.norms(domain, V, uh, u_exact)

    elif args.case == "disk2d":
        domain, _, facet_tags = geo.disk_in_box(comm)
        V = fem.functionspace(domain, ("Lagrange", 1))
        # constant eps
        W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
        eps = fem.Function(W); eps.x.array[:] = 1.0
        # Dirichlet: hole=0.3V (tag=2), outer=0V (tag=1)
        u_out = fem.Constant(domain, PETSc.ScalarType(0.0))
        u_in  = fem.Constant(domain, PETSc.ScalarType(0.3))
        fdim = domain.topology.dim - 1
        dofs_out = fem.locate_dofs_topological(V, fdim, facet_tags.find(1))
        dofs_in  = fem.locate_dofs_topological(V, fdim, facet_tags.find(2))
        bcs = [fem.dirichletbc(u_out, dofs_out, V), fem.dirichletbc(u_in, dofs_in, V)]
        # Laplace
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(eps*ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx
        uh = ps._linear_problem(a, L, bcs, "disk_").solve()
        L2 = H1s = np.nan
    else:
        raise ValueError("unknown case")

    # Write outputs
    out = args.outfile
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with XDMFFile(MPI.COMM_WORLD, f"{out}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[main] wrote {out}.xdmf (+ .h5)")
        print(f"[main] L2={L2:.6e}  H1semi={H1s:.6e}")

if __name__ == "__main__":
    main()
