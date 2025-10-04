# tests/test_mms_2d.py
import sys
from mpi4py import MPI
from dolfinx import fem
import numpy as np
import ufl

from src import geometry as geo, permittivity as pm, poisson as ps

def main():
    domain = geo.unit_square(MPI.COMM_WORLD, 64, 64)
    V = fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)
    eps = pm.dg0_from_indicator(domain, lambda X: X[0] < 0.5, 3.0, 1.0)
    pi = np.pi
    u_exact = ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])
    f = (eps) * (2.0*pi**2) * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1])

    fdim = domain.topology.dim - 1
    from dolfinx import mesh as dmesh
    facets = {
        "x0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 0.0)),
        "x1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 1.0)),
        "y0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 0.0)),
        "y1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 1.0)),
    }
    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    bcs = [fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, facets[k])) for k in ("x0","x1","y0")]

    # Neumann on y1
    n = ufl.as_vector((0.0, 1.0))
    g = ufl.dot((eps)*ufl.grad(u_exact), n)
    import numpy as np
    indices = np.concatenate([facets["x0"], facets["x1"], facets["y0"], facets["y1"]])
    values  = np.concatenate([np.full_like(facets["x0"], 1),
                              np.full_like(facets["x1"], 2),
                              np.full_like(facets["y0"], 3),
                              np.full_like(facets["y1"], 4)])
    order = np.argsort(indices)
    mt = dmesh.meshtags(domain, fdim, indices[order], values[order])

    uh = ps.solve_mixed(domain, V, eps, f, bcs, neumann_terms=[(g, 4)], facet_tags=mt, prefix="t2d_")
    L2, H1s = ps.norms(domain, V, uh, u_exact)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[test2d] L2={L2:.3e} H1semi={H1s:.3e}")
    tol = 5e-2
    ok = (L2 < tol) and (H1s < 5e-1)  # modest mesh
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
