#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio  # pip install meshio

xdmf_path = "jackson_interface.xdmf"

# --- Read mesh + data ---
mesh = meshio.read(xdmf_path)

print("Points shape:", mesh.points.shape)
print("Point data keys:", list(mesh.point_data.keys()))
print("Cell types:", [c.type for c in mesh.cells])

# --- Choose which field to plot ---
if "phi" in mesh.point_data:
    field_name = "phi"
else:
    # Fallback: first available scalar field
    field_name = list(mesh.point_data.keys())[0]

field = mesh.point_data[field_name].ravel()

# --- Extract (x, z) coordinates ---
coords = mesh.points
if coords.shape[1] == 2:
    x = coords[:, 0]
    z = coords[:, 1]
else:
    # assume 3D coords (x, y, z) and we want x–z plane
    x = coords[:, 0]
    z = coords[:, 2]

# --- Get 2D cells (triangles / quads) ---
triangles = None
for cell_block in mesh.cells:
    if cell_block.type == "triangle":
        triangles = cell_block.data
        break
    if cell_block.type == "quad":
        quads = cell_block.data
        triangles = np.vstack([
            quads[:, [0, 1, 2]],
            quads[:, [0, 2, 3]],
        ])
        break

if triangles is None:
    raise RuntimeError("No triangle/quad cells found to build a 2D contour.")

triang = mtri.Triangulation(x, z, triangles)

# --- Make contour plot ---
plt.figure(figsize=(7, 4))
tcf = plt.tricontourf(triang, field, levels=80)
plt.colorbar(tcf, label=field_name)

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title(f"Jackson interface: {field_name}(x, z)")
plt.tight_layout()
plt.savefig("jackson_interface_contour.png", dpi=300)
plt.show()
