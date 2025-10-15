from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile

# Import your modules (lowercase packages)
from main.geometry import geometry as geo
from main.physics import permittivity as pm
from main.solver import poisson as ps

comm = MPI.COMM_WORLD
rank = comm.rank

# ----------------- user sweep knobs -----------------
# z-positions for the Gaussian source (electron)
z0_list = np.linspace(0.2, 0.8, 5)     # e.g. 0.20, 0.35, 0.50, 0.65, 0.80
# top boundary Dirichlet values (Volts)
phi_top_list = [0.0, 0.1, 0.2, 0.3]
# Gaussian parameters
q      = 1.0           # total “charge” (arbitrary units)
sigma  = 0.06          # Gaussian width
x0, y0 = 0.5, 0.5      # centered in x-y
# mesh density
N = 28                 # cube n=n=n
# output directory prefix (relative so it ends up on host)
out_prefix = "Results/Oct-4/sweeps/phi"

# ----------------- build geometry / spaces -----------------
domain = geo.unit_cube(comm, n=(N, N, N))
V = fem.functionspace(domain, ("Lagrange", 1))
tdim, fdim = domain.topology.dim, domain.topology.dim - 1

# DG0 permittivity: uniform eps=1
W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
eps = fem.Function(W, name="epsilon"); eps.x.array[:] = 1.0

# facet sets
facets = {
    "x0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 0.0)),
    "x1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[0], 1.0)),
    "y0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 0.0)),
    "y1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[1], 1.0)),
    "z0": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 0.0)),
    "z1": dmesh.locate_entities_boundary(domain, fdim, lambda X: np.isclose(X[2], 1.0)),
}

# measures
dx = ufl.Measure("dx", domain=domain)

# spatial coord
x = ufl.SpatialCoordinate(domain)
pi = np.pi

# ----------------- helper to build normalized 3D Gaussian -----------------
def gaussian3d(q, x, x0, y0, z0, sigma):
    r2 = (x[0]-x0)**2 + (x[1]-y0)**2 + (x[2]-z0)**2
    norm = 1.0 / ((2.0*pi*sigma**2)**1.5)
    return q * norm * ufl.exp(-r2/(2.0*sigma**2))

# ----------------- sweep -----------------
for z0 in z0_list:
    for phi_top in phi_top_list:
        # RHS source term
        f = gaussian3d(q, x, x0, y0, float(z0), sigma)

        # Dirichlet: 0 V on all faces except z=1 -> phi_top
        u0 = fem.Constant(domain, PETSc.ScalarType(0.0))
        u1 = fem.Constant(domain, PETSc.ScalarType(phi_top))
        # zero on x0,x1,y0,y1,z0
        bcs = [fem.dirichletbc(u0, fem.locate_dofs_topological(V, fdim, facets[k]), V)
               for k in ("x0","x1","y0","y1","z0")]
        # phi_top on z1
        dofs_z1 = fem.locate_dofs_topological(V, fdim, facets["z1"])
        bcs.append(fem.dirichletbc(u1, dofs_z1, V))

        # Solve -div(eps grad u) = f with Dirichlet BCs
        uh = ps.solve_dirichlet(domain, V, eps, f, bcs, prefix="sweep3d_")

        # Report
        phi_min = float(uh.x.array.min())
        phi_max = float(uh.x.array.max())
        if rank == 0:
            print(f"[sweep] z0={z0:.3f}, phi_top={phi_top:.3f} -> phi range [{phi_min:.6g}, {phi_max:.6g}]")

        # Write results (XDMF/H5) under mounted tree
        out = f"{out_prefix}_z{z0:.2f}_V{phi_top:.2f}"
        if rank == 0:
            import os
            os.makedirs(os.path.dirname(out), exist_ok=True)
        with XDMFFile(comm, f"{out}.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)
            xdmf.write_function(eps)
