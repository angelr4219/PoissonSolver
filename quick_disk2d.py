import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio, XDMFFile
import ufl

# --- Geometry: unit square with centered circular hole (R=0.2) ---
gmsh.initialize()
gmsh.model.add("disk2d")
Lx, Ly, R = 1.0, 1.0, 0.20
rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
circ = gmsh.model.occ.addDisk(Lx/2, Ly/2, 0, R, R)
geom = gmsh.model.occ.cut([(2, rect)], [(2, circ)], removeObject=True, removeTool=False)
gmsh.model.occ.synchronize()

# Tag boundaries: outer=1, hole=2; cells=1
b_all  = gmsh.model.getBoundary(geom[0], oriented=False, recursive=False)
b_hole = gmsh.model.getBoundary([(2, circ)], oriented=False, recursive=False)
outer  = [c for c in b_all if c not in b_hole]
gmsh.model.addPhysicalGroup(1, [e[1] for e in outer], tag=1)
gmsh.model.addPhysicalGroup(1, [e[1] for e in b_hole], tag=2)
gmsh.model.addPhysicalGroup(2, [geom[0][0][1]], tag=1)

gmsh.model.mesh.generate(2)

comm = MPI.COMM_WORLD
domain, cell_tags, facet_tags, *_ = gmshio.model_to_mesh(gmsh.model, comm, 0)
gmsh.finalize()

V = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet BCs: outer=0 V, hole=0.3 V
u_outer = fem.Constant(domain, PETSc.ScalarType(0.0))
u_inner = fem.Constant(domain, PETSc.ScalarType(0.3))
tdim = domain.topology.dim
outer_facets = facet_tags.find(1)
inner_facets = facet_tags.find(2)
dofs_outer = fem.locate_dofs_topological(V, tdim-1, outer_facets)
dofs_inner = fem.locate_dofs_topological(V, tdim-1, inner_facets)
bcs = [fem.dirichletbc(u_outer, dofs_outer, V),
       fem.dirichletbc(u_inner, dofs_inner, V)]

# Solve -∇²u = 0
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
    petsc_options_prefix="qp_"   # <- non-empty prefix required by your build
)
uh = problem.solve()
uh.name = "phi"

phi_min = float(uh.x.array.min()); phi_max = float(uh.x.array.max())
if comm.rank == 0:
    print(f"[quick_disk2d] phi range: min={phi_min:.6g}, max={phi_max:.6g}")

with XDMFFile(comm, "results/quick_disk2d.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
