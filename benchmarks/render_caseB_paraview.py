#!/usr/bin/env python3
"""Render the Case B FEM-Dirichlet phi field to PNGs via pyvista (VTK)
off-screen rendering -- a ParaView-equivalent view without an interactive
ParaView session.

VTK's built-in Xdmf3Reader segfaults on this dolfinx-written XDMF/HDF5 pair
in this environment, so the mesh + field are read directly out of the HDF5
file with h5py and assembled into a pyvista UnstructuredGrid by hand.

Usage (inside a container with pyvista + h5py installed):
    python3 benchmarks/render_caseB_paraview.py
"""
import h5py
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

H5 = "results/point_charge/caseB_fem_dirichlet.h5"
OUTDIR = "results/point_charge"

with h5py.File(H5, "r") as f:
    points = f["Mesh/mesh/geometry"][:]
    topo = f["Mesh/mesh/topology"][:]
    phi = f["Function/phi/0"][:].reshape(-1)

print("points:", points.shape, "topo:", topo.shape, "phi:", phi.shape)

n_cells = topo.shape[0]
cells = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64), topo.astype(np.int64)]).flatten()
celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
grid = pv.UnstructuredGrid(cells, celltypes, points)
grid["phi"] = phi

print(grid)

# 1. Cutaway (clip) view of the whole box, colored by phi
clipped = grid.clip(normal="z", origin=grid.center)
p = pv.Plotter(off_screen=True, window_size=(1000, 800))
p.add_mesh(clipped, scalars="phi", cmap="coolwarm", show_edges=False)
p.add_axes()
p.camera_position = "iso"
p.screenshot(f"{OUTDIR}/caseB_clip_iso.png")
p.close()

# 2. Mid-plane slice through the charge (z = centre)
sl = grid.slice(normal="z", origin=grid.center)
p = pv.Plotter(off_screen=True, window_size=(1000, 800))
p.add_mesh(sl, scalars="phi", cmap="coolwarm", show_edges=False)
p.view_xy()
p.add_axes()
p.screenshot(f"{OUTDIR}/caseB_slice_z0.png")
p.close()

# 3. Diagonal slice (x = y plane) -- the direction the Case B probe points lie along
sl2 = grid.slice(normal=[1, -1, 0], origin=grid.center)
p = pv.Plotter(off_screen=True, window_size=(1000, 800))
p.add_mesh(sl2, scalars="phi", cmap="coolwarm", show_edges=False)
p.add_axes()
p.view_vector([1, 1, 0])
p.reset_camera()
p.screenshot(f"{OUTDIR}/caseB_slice_diag.png")
p.close()

print("Wrote:")
print(f"  {OUTDIR}/caseB_clip_iso.png")
print(f"  {OUTDIR}/caseB_slice_z0.png")
print(f"  {OUTDIR}/caseB_slice_diag.png")
