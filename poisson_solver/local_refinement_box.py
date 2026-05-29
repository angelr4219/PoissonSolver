"""
local_refinement_box.py

Geometry/mesh verification: 3D box with a locally refined inner region.
No physics — pure mesh quality test.

Box:         500 x 500 x 150 nm   (x: -250→250, y: -250→250, z: 0→150)
Fine region:  80 x  80 x  30 nm   (x:  -40→ 40, y:  -40→ 40, z: 45→ 75)

h_coarse = 20 nm   h_fine = 4 nm

Outputs (current working directory):
    locally_refined_box.xdmf
    locally_refined_box.h5

ParaView workflow:
    Open .xdmf → Slice at z=60 nm → Surface With Edges → color by material_id
    material_id = 1  fine region
    material_id = 2  coarse region
"""

import numpy as np
from mpi4py import MPI
import gmsh as gmsh_api
from dolfinx.io import gmsh as gmshio, XDMFFile
from dolfinx.fem import functionspace, Function

# ---------------------------------------------------------------------------
# Parameters (nm)
# ---------------------------------------------------------------------------
H_COARSE = 20.0
H_FINE   =  4.0

BOX_X0, BOX_Y0, BOX_Z0 = -250.0, -250.0,  0.0
BOX_LX, BOX_LY, BOX_LZ =  500.0,  500.0, 150.0

FINE_X0, FINE_Y0, FINE_Z0 = -40.0, -40.0, 45.0
FINE_X1, FINE_Y1, FINE_Z1 =  40.0,  40.0, 75.0

# ---------------------------------------------------------------------------
# Build Gmsh model
# ---------------------------------------------------------------------------
gmsh_api.initialize()
gmsh_api.option.setNumber("General.Terminal", 1)
gmsh_api.model.add("local_refinement_box")

box_tag = gmsh_api.model.occ.addBox(BOX_X0, BOX_Y0, BOX_Z0,
                                     BOX_LX, BOX_LY, BOX_LZ)
gmsh_api.model.occ.synchronize()

# Physical groups required by gmshio
gmsh_api.model.addPhysicalGroup(3, [box_tag], tag=1, name="domain")
bnd = gmsh_api.model.getBoundary([(3, box_tag)], oriented=False, combined=False)
gmsh_api.model.addPhysicalGroup(2, [abs(b[1]) for b in bnd], tag=1, name="boundary")

# Box mesh-size field: h_fine inside, h_coarse outside, sharp transition
fid = gmsh_api.model.mesh.field.add("Box")
gmsh_api.model.mesh.field.setNumber(fid, "VIn",       H_FINE)
gmsh_api.model.mesh.field.setNumber(fid, "VOut",      H_COARSE)
gmsh_api.model.mesh.field.setNumber(fid, "XMin",      FINE_X0)
gmsh_api.model.mesh.field.setNumber(fid, "XMax",      FINE_X1)
gmsh_api.model.mesh.field.setNumber(fid, "YMin",      FINE_Y0)
gmsh_api.model.mesh.field.setNumber(fid, "YMax",      FINE_Y1)
gmsh_api.model.mesh.field.setNumber(fid, "ZMin",      FINE_Z0)
gmsh_api.model.mesh.field.setNumber(fid, "ZMax",      FINE_Z1)
gmsh_api.model.mesh.field.setNumber(fid, "Thickness", 0.0)
gmsh_api.model.mesh.field.setAsBackgroundMesh(fid)

gmsh_api.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay 3D
gmsh_api.model.mesh.generate(3)

# ---------------------------------------------------------------------------
# Convert to DOLFINx
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mesh_data = gmshio.model_to_mesh(gmsh_api.model, comm, rank=0, gdim=3)
mesh = mesh_data.mesh
gmsh_api.finalize()

# ---------------------------------------------------------------------------
# Diagnostic DG0 fields
# ---------------------------------------------------------------------------
tdim  = mesh.topology.dim
ncells = mesh.topology.index_map(tdim).size_local

# Cell midpoints
mesh.topology.create_connectivity(tdim, 0)
c2v    = mesh.topology.connectivity(tdim, 0)
coords = mesh.geometry.x

try:
    from dolfinx.mesh import compute_midpoints
    midpoints = compute_midpoints(mesh, tdim, np.arange(ncells, dtype=np.int32))
except (ImportError, AttributeError):
    midpoints = np.array([coords[c2v.links(c)].mean(axis=0) for c in range(ncells)])

in_fine = (
    (midpoints[:, 0] >= FINE_X0) & (midpoints[:, 0] <= FINE_X1) &
    (midpoints[:, 1] >= FINE_Y0) & (midpoints[:, 1] <= FINE_Y1) &
    (midpoints[:, 2] >= FINE_Z0) & (midpoints[:, 2] <= FINE_Z1)
)

V0 = functionspace(mesh, ("DG", 0))

material_id = Function(V0, name="material_id")
material_id.x.array[:] = np.where(in_fine, 1.0, 2.0)

target_h = Function(V0, name="target_h_nm")
target_h.x.array[:] = np.where(in_fine, H_FINE, H_COARSE)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
with XDMFFile(comm, "locally_refined_box.xdmf", "w") as xf:
    xf.write_mesh(mesh)
    xf.write_function(material_id)
    xf.write_function(target_h)

n_fine   = int(in_fine.sum())
n_coarse = ncells - n_fine
print(f"\nDone.")
print(f"  Total cells  : {ncells}")
print(f"  Fine   cells (material_id=1, h={H_FINE} nm)  : {n_fine}")
print(f"  Coarse cells (material_id=2, h={H_COARSE} nm): {n_coarse}")
print(f"  Output : locally_refined_box.xdmf / .h5")
