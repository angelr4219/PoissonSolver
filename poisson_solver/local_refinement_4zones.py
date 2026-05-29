"""
local_refinement_4zones.py

4 concentric nested refinement zones, each progressively finer toward the center.
Same outer box as before. No physics — mesh geometry test.

Box: 500 x 500 x 150 nm  (x: -250→250, y: -250→250, z: 0→150)

Zones (center = x=0, y=0, z=60 nm):
    Zone 1  background         500x500x150 nm   h = 20 nm   material_id = 1
    Zone 2  outer ring         200x200x 80 nm   h =  8 nm   material_id = 2
    Zone 3  middle ring        100x100x 40 nm   h =  4 nm   material_id = 3
    Zone 4  innermost           40x 40x 20 nm   h =  1 nm   material_id = 4

Gmsh strategy: one Box field per zone + a MathEval constant background,
combined with a Min field so the finest target always wins at each point.

Output: locally_refined_4zones.xdmf / .h5
"""

import numpy as np
from mpi4py import MPI
import gmsh as gmsh_api
from dolfinx.io import gmsh as gmshio, XDMFFile
from dolfinx.fem import functionspace, Function

# ---------------------------------------------------------------------------
# Zone definitions  (all nm, center = 0,0,60)
# ---------------------------------------------------------------------------
CX, CY, CZ = 0.0, 0.0, 60.0

ZONES = [
    # (label,  half_x, half_y, half_z,  h_nm,  material_id)
    ("zone4_inner",   20.0,  20.0, 10.0,  1.0,  4),
    ("zone3_mid",     50.0,  50.0, 20.0,  4.0,  3),
    ("zone2_outer",  100.0, 100.0, 40.0,  8.0,  2),
    # zone1 = everything else, handled by background constant field
]
H_BACKGROUND = 20.0
MATERIAL_BACKGROUND = 1

BOX_X0, BOX_Y0, BOX_Z0 = -250.0, -250.0,  0.0
BOX_LX, BOX_LY, BOX_LZ =  500.0,  500.0, 150.0

# ---------------------------------------------------------------------------
# Build Gmsh model
# ---------------------------------------------------------------------------
gmsh_api.initialize()
gmsh_api.option.setNumber("General.Terminal", 1)
gmsh_api.model.add("4zones")

box_tag = gmsh_api.model.occ.addBox(BOX_X0, BOX_Y0, BOX_Z0,
                                     BOX_LX, BOX_LY, BOX_LZ)
gmsh_api.model.occ.synchronize()

gmsh_api.model.addPhysicalGroup(3, [box_tag], tag=1, name="domain")
bnd = gmsh_api.model.getBoundary([(3, box_tag)], oriented=False, combined=False)
gmsh_api.model.addPhysicalGroup(2, [abs(b[1]) for b in bnd], tag=1, name="boundary")

# Background constant size field
fid_bg = gmsh_api.model.mesh.field.add("MathEval")
gmsh_api.model.mesh.field.setString(fid_bg, "F", str(H_BACKGROUND))
field_ids = [fid_bg]

# One Box field per zone (VOut large so Min picks background outside)
for label, hx, hy, hz, h_nm, _ in ZONES:
    fid = gmsh_api.model.mesh.field.add("Box")
    gmsh_api.model.mesh.field.setNumber(fid, "VIn",       h_nm)
    gmsh_api.model.mesh.field.setNumber(fid, "VOut",      1e6)
    gmsh_api.model.mesh.field.setNumber(fid, "XMin",      CX - hx)
    gmsh_api.model.mesh.field.setNumber(fid, "XMax",      CX + hx)
    gmsh_api.model.mesh.field.setNumber(fid, "YMin",      CY - hy)
    gmsh_api.model.mesh.field.setNumber(fid, "YMax",      CY + hy)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMin",      CZ - hz)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMax",      CZ + hz)
    gmsh_api.model.mesh.field.setNumber(fid, "Thickness", 0.0)
    field_ids.append(fid)

# Min field: finest target wins everywhere
fid_min = gmsh_api.model.mesh.field.add("Min")
gmsh_api.model.mesh.field.setNumbers(fid_min, "FieldsList", field_ids)
gmsh_api.model.mesh.field.setAsBackgroundMesh(fid_min)

gmsh_api.option.setNumber("Mesh.Algorithm3D", 1)
gmsh_api.model.mesh.generate(3)

# ---------------------------------------------------------------------------
# Convert to DOLFINx
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mesh_data = gmshio.model_to_mesh(gmsh_api.model, comm, rank=0, gdim=3)
mesh = mesh_data.mesh
gmsh_api.finalize()

# ---------------------------------------------------------------------------
# DG0 diagnostic fields
# ---------------------------------------------------------------------------
tdim   = mesh.topology.dim
ncells = mesh.topology.index_map(tdim).size_local

mesh.topology.create_connectivity(tdim, 0)
c2v    = mesh.topology.connectivity(tdim, 0)
coords = mesh.geometry.x

try:
    from dolfinx.mesh import compute_midpoints
    midpoints = compute_midpoints(mesh, tdim, np.arange(ncells, dtype=np.int32))
except (ImportError, AttributeError):
    midpoints = np.array([coords[c2v.links(c)].mean(axis=0) for c in range(ncells)])

V0 = functionspace(mesh, ("DG", 0))
material_id = Function(V0, name="material_id")
target_h    = Function(V0, name="target_h_nm")

# Assign from finest (zone4) outward so inner labels overwrite outer ones
material_id.x.array[:] = float(MATERIAL_BACKGROUND)
target_h.x.array[:]    = H_BACKGROUND

for label, hx, hy, hz, h_nm, mat_id in reversed(ZONES):
    mask = (
        (midpoints[:, 0] >= CX - hx) & (midpoints[:, 0] <= CX + hx) &
        (midpoints[:, 1] >= CY - hy) & (midpoints[:, 1] <= CY + hy) &
        (midpoints[:, 2] >= CZ - hz) & (midpoints[:, 2] <= CZ + hz)
    )
    material_id.x.array[mask] = float(mat_id)
    target_h.x.array[mask]    = h_nm

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
with XDMFFile(comm, "locally_refined_4zones.xdmf", "w") as xf:
    xf.write_mesh(mesh)
    xf.write_function(material_id)
    xf.write_function(target_h)

# Summary
print(f"\n{'='*60}")
print(f"4-zone concentric refinement mesh")
print(f"{'='*60}")
print(f"{'Zone':<12} {'bounds (half-nm)':<28} {'h_target':>8}  {'cells':>8}")
print(f"{'-'*60}")
for label, hx, hy, hz, h_nm, mat_id in ZONES:
    mask = (material_id.x.array == float(mat_id))
    n = int(mask.sum())
    print(f"  id={mat_id}  {label:<18}  {hx:.0f}x{hy:.0f}x{hz:.0f} nm   {h_nm:>5.1f} nm  {n:>8,}")
n_bg = int((material_id.x.array == float(MATERIAL_BACKGROUND)).sum())
print(f"  id=1  background (rest)                 {H_BACKGROUND:>5.1f} nm  {n_bg:>8,}")
print(f"  {'TOTAL':>44}  {ncells:>8,}")
print(f"\nOutput: locally_refined_4zones.xdmf / .h5")
print(f"\nParaView: Slice at z=60 nm → Surface With Edges → color by material_id")
print(f"  material_id 4 = finest inner zone (h~1 nm)")
print(f"  material_id 1 = coarsest background (h~20 nm)")
