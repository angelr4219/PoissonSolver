# tests/test_mms_3d.py
import sys
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
import numpy as np
import ufl

from src import geometry as geo, permittivity as pm, poisson as ps

def main():
    domain = geo.unit_cube(MPI.COMM_WORLD, n=(18,18,18))
    V = fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)
    eps = pm.dg0_from_indicator(domain, lambda X: X[2] < 0.5, 1.0, 4.0)
    pi = np.pi
    u_exact = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * ufl.sin(pi*x[2])
    f = (eps) * (3.0*pi**2) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) * ufl.sin(pi*x[2])

    fdim = domain.topology.dim - 1
    facets = {
        "x0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 0.0)),
        "x1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 1.0)),
        "y0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 0.0)),
        "y1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 1.0)),
        "z0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 0.0)),
        "z1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 1.0)),
    }
    uD = fem.Function(V); uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    bcs = [fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, facets[k]))
           for k in ("x0","x1","y0","y1","z0")]
    n = ufl.as_vector((0.0,0.0,1.0))
    g = ufl.dot((eps)*ufl.grad(u_exact), n)

    indices = np.concatenate([facets[k] for k in facets])
    tag_map = {"x0":1,"x1":2,"y0":3,"y1":4,"z0":5,"z1":6}
    values  = np.concatenate([np.full_like(facets[k], tag_map[k]) for k in facets])
    order = np.argsort(indices)
    mt = dmesh.meshtags(domain, fdim, indices[order], values[order])

    uh = ps.solve_mixed(domain, V, eps, f, bcs, neumann_terms=[(g, 6)], facet_tags=mt, prefix="t3d_")
    L2, H1s = ps.norms(domain, V, uh, u_exact)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[test3d] L2={L2:.3e} H1semi={H1s:.3e}")
    tol = 8e-2
    ok = (L2 < tol) and (H1s < 6e-1)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
