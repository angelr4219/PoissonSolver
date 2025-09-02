# --- Poisson with rectangular gates + variable permittivity by subdomain ---
# Adds a top "oxide" strip (e.g., y in [0.8, 1.0]) and the rest "semiconductor".
# Keep your previous imports; only the CAD/build section and eps definition change.

from mpi4py import MPI
import gmsh, numpy as np, ufl
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem

comm = MPI.COMM_WORLD
rank = comm.rank

gmsh.initialize()
gmsh.model.add("rect-with-holes-materials")

# --- Outer domain
Lx, Ly = 1.0, 1.0
outer = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)

# --- Holes (gates)
hA = gmsh.model.occ.addRectangle(0.15, 0.75, 0.0, 0.20, 0.20)
hB = gmsh.model.occ.addRectangle(0.55, 0.75, 0.0, 0.30, 0.20)

# Subtract holes to create inner boundaries
gmsh.model.occ.cut([(2, outer)], [(2, hA), (2, hB)], removeObject=True, removeTool=True)

# --- Add a top "oxide" strip (this is NOT a hole; we keep material there)
oxide_strip = gmsh.model.occ.addRectangle(0.0, 0.80, 0.0, 1.0, 0.20)

# Fragment the current surface with the oxide rectangle to split materials
gmsh.model.occ.fragment(gmsh.model.occ.getEntities(2), [(2, oxide_strip)])
gmsh.model.occ.synchronize()

# Collect final 2D surfaces (should be 2: bottom+top), and classify by centroid y
surfs = gmsh.model.getEntities(dim=2)
assert len(surfs) >= 2

def centroid_of_surface(tag):
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, tag)
    return 0.5*(xmin+xmax), 0.5*(ymin+ymax)

top_surf_tags, bot_surf_tags = [], []
for _, s in surfs:
    cx, cy = centroid_of_surface(s)
    # The small hole faces do not exist (holes were cut), so remaining are the two material patches.
    if cy > 0.7:
        top_surf_tags.append(s)   # oxide region
    else:
        bot_surf_tags.append(s)   # semiconductor region

# --- Tag boundaries: outer vs hole boundaries (gateA/gateB)
# Get ALL boundary curves; separate hole perimeters (high y) from outer
curves = gmsh.model.getBoundary([(2, t) for t in top_surf_tags + bot_surf_tags],
                                oriented=False, recursive=True)

def bbox_of_curve(curve):
    return gmsh.model.getBoundingBox(curve[0], curve[1])

outer_curves, gateA_curves, gateB_curves = [], [], []
for c in curves:
    if c[0] != 1:
        continue
    xmin, ymin, _, xmax, ymax, _ = bbox_of_curve(c)
    cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
    # Heuristic: gate perimeters are high in y and lie under either hole A (x<0.5) or B (x>0.5)
    if cy > 0.7 and 0.15 - 1e-6 <= cx <= 0.35 + 1e-6:
        gateA_curves.append(c)
    elif cy > 0.7 and 0.55 - 1e-6 <= cx <= 0.85 + 1e-6:
        gateB_curves.append(c)
    else:
        outer_curves.append(c)

# Deduplicate curve tags
outer_tags = list({tag for (_, tag) in outer_curves})
gateA_tags = list({tag for (_, tag) in gateA_curves})
gateB_tags = list({tag for (_, tag) in gateB_curves})

# --- Physical groups
pg_outer = gmsh.model.addPhysicalGroup(1, outer_tags, name="outer")
pg_gateA = gmsh.model.addPhysicalGroup(1, gateA_tags, name="gateA")
pg_gateB = gmsh.model.addPhysicalGroup(1, gateB_tags, name="gateB")

# Material (cell) groups
pg_semic = gmsh.model.addPhysicalGroup(2, bot_surf_tags, name="semiconductor")
pg_oxide = gmsh.model.addPhysicalGroup(2, top_surf_tags, name="oxide")

# Mesh and convert
gmsh.model.mesh.generate(2)
domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
gmsh.finalize()

# ----------------------------
# FE spaces
# ----------------------------
V = fem.functionspace(domain, ("Lagrange", 1))
W = fem.functionspace(domain, ("DG", 0))  # cell-wise constants for epsilon

# --- Permittivities
eps0 = 8.8541878128e-12
eps_r_oxide = 3.9
eps_r_semic = 11.7  # e.g., Si; change as needed

eps_vals = np.full(len(cell_tags.values), eps0 * eps_r_semic)
# Overwrite oxide cells
oxide_cells = np.where(cell_tags.values == pg_oxide)[0]
eps_vals[oxide_cells] = eps0 * eps_r_oxide

# Put into a DG0 function (cell-wise ε)
eps = fem.Function(W)
eps.x.array[:] = eps_vals

# --- Source (volume charge); start with Laplace
rho = fem.Constant(domain, 0.0)

# --- Dirichlet on gate hole boundaries
import numpy as np
ft = facet_tags

def dirichlet_on(tag, value):
    dofs = fem.locate_dofs_topological(V, ft.dim, np.where(ft.values == tag)[0])
    return fem.dirichletbc(fem.Constant(domain, value), dofs, V)

VgateA, VgateB = 0.20, 0.00
bcs = [dirichlet_on(pg_gateA, VgateA), dirichlet_on(pg_gateB, VgateB)]

# --- Variational problem: -div( ε ∇φ ) = ρ
phi = ufl.TrialFunction(V)
v   = ufl.TestFunction(V)

a = ufl.inner(eps * ufl.grad(phi), ufl.grad(v)) * ufl.dx
L = rho * v * ufl.dx

uh = LinearProblem(a, L, bcs=bcs).solve()

# Quick check
if rank == 0:
    print(f"phi: min={uh.x.array.min():.4g} V, max={uh.x.array.max():.4g} V")

# Save (mesh + field + cell_tags) for ParaView
from dolfinx.io import XDMFFile
with XDMFFile(comm, "phi_with_gates_var_eps.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    # Also write the DG0 epsilon as a scalar field for visual verification
    xdmf.write_function(eps, "epsilon")
