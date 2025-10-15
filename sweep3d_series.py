from mpi4py import MPI
from petsc4py import PETSc
import numpy as np, ufl, os
from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile

# your modules
from main.geometry import geometry as geo
from main.physics  import permittivity as pm
from main.solver   import poisson as ps

comm = MPI.COMM_WORLD
rank = comm.rank

# sweep parameters
z0_list      = np.linspace(0.2, 0.8, 5)     # adjust as you like
phi_top_list = [0.0, 0.1, 0.2, 0.3]
q, sigma     = 1.0, 0.06
x0, y0       = 0.5, 0.5
N            = 28

# geometry and spaces
domain = geo.unit_cube(comm, n=(N, N, N))
V = fem.functionspace(domain, ("Lagrange", 1))
W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
eps = fem.Function(W, name="epsilon"); eps.x.array[:] = 1.0

fdim = domain.topology.dim - 1
facets = {
    "x0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 0.0)),
    "x1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 1.0)),
    "y0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 0.0)),
    "y1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 1.0)),
    "z0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 0.0)),
    "z1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 1.0)),
}

x = ufl.SpatialCoordinate(domain)
pi = np.pi
def gaussian3d(q, x, x0, y0, z0, s):
    r2 = (x[0]-x0)**2 + (x[1]-y0)**2 + (x[2]-z0)**2
    return q * (1.0 / ((2*pi*s**2)**1.5)) * ufl.exp(-r2/(2*s**2))

# output
out = "Results/Oct-4/sweeps/phi_sweep_series"
if rank == 0:
    os.makedirs(os.path.dirname(out), exist_ok=True)

# set up XDMF time series
with XDMFFile(comm, f"{out}.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

    t = 0.0
    for z0 in z0_list:
        for Vtop in phi_top_list:
            f = gaussian3d(q, x, x0, y0, float(z0), sigma)

            # Dirichlet: 0 on all faces except z=1 -> Vtop
            u0 = fem.Constant(domain, PETSc.ScalarType(0.0))
            u1 = fem.Constant(domain, PETSc.ScalarType(Vtop))
            bcs = [fem.dirichletbc(u0, fem.locate_dofs_topological(V, fdim, facets[k]), V)
                   for k in ("x0","x1","y0","y1","z0")]
            bcs.append(fem.dirichletbc(u1, fem.locate_dofs_topological(V, fdim, facets["z1"]), V))

            uh = ps.solve_dirichlet(domain, V, eps, f, bcs, prefix="series3d_")
            uh.name = f"phi_z{z0:.2f}_V{Vtop:.2f}"

            # write this frame at time 't'
            xdmf.write_function(uh, t)
            xdmf.write_function(eps, t)
            if rank == 0:
                print(f"[series] t={t:.0f}: z0={z0:.2f}, Vtop={Vtop:.2f}")
            t += 1.0
