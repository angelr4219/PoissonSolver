from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
import numpy as np
import ufl

comm = MPI.COMM_WORLD

domain = mesh.create_box(
    comm,
    [np.array([0,0,0]), np.array([1,1,1])],
    n=(24,24,24),
    cell_type=mesh.CellType.tetrahedron
)
V = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet: z=0 -> 0 V, z=1 -> 0.5 V
tdim = domain.topology.dim
facets_z0 = mesh.locate_entities_boundary(domain, tdim-1, lambda x: np.isclose(x[2], 0.0))
facets_z1 = mesh.locate_entities_boundary(domain, tdim-1, lambda x: np.isclose(x[2], 1.0))
dofs_z0 = fem.locate_dofs_topological(V, tdim-1, facets_z0)
dofs_z1 = fem.locate_dofs_topological(V, tdim-1, facets_z1)
u0 = fem.Constant(domain, PETSc.ScalarType(0.0))
u1 = fem.Constant(domain, PETSc.ScalarType(0.5))
bcs = [fem.dirichletbc(u0, dofs_z0, V), fem.dirichletbc(u1, dofs_z1, V)]

# -∇²u = 0
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
    petsc_options_prefix="qp_"
)
uh = problem.solve()
uh.name = "phi"

phi_min = float(uh.x.array.min()); phi_max = float(uh.x.array.max())
if comm.rank == 0:
    print(f"[quick_box3d] phi range: min={phi_min:.6g}, max={phi_max:.6g}")

with XDMFFile(comm, "results/quick_box3d.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
