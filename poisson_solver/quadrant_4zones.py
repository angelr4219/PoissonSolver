"""
quadrant_4zones.py

3D box split into 4 equal quadrants in XY, each spanning full Z.
Each quadrant is a real OCC volume with a unique material tag and
a dramatically different mesh refinement.

Box: 500 x 500 x 150 nm  (x: -250→250, y: -250→250, z: 0→150)

Quadrants (each 250x250 nm, full Z):
    Q1  top-left     x<0, y>0   h = 20 nm   material_id = 1  (coarsest)
    Q2  top-right    x>0, y>0   h =  8 nm   material_id = 2
    Q3  bottom-left  x<0, y<0   h =  4 nm   material_id = 3
    Q4  bottom-right x>0, y<0   h =  1 nm   material_id = 4  (finest)

OCC fragment is used so all 4 volumes share faces → conforming mesh.

Then runs 4 uniform benchmark meshes (matching each quadrant's h)
and logs mesh-gen time, DOLFINx conversion, XDMF write, total time,
cell count, and peak memory.

Outputs (current working directory):
    quadrant_4zones.xdmf / .h5
    benchmark_h20nm.xdmf / .h5
    benchmark_h8nm.xdmf / .h5
    benchmark_h4nm.xdmf / .h5
    benchmark_h1nm.xdmf / .h5   (large — may take several minutes)
"""

import time
import tracemalloc
import resource
import numpy as np
from mpi4py import MPI
import gmsh as gmsh_api
from dolfinx.io import gmsh as gmshio, XDMFFile
from dolfinx.fem import functionspace, Function

comm = MPI.COMM_WORLD

BOX_Z0, BOX_LZ = 0.0, 150.0

# (name, x0, y0, dx, dy, h_nm, material_id)
QUADRANTS = [
    ("Q1_top_left",     -250.0,    0.0,  250.0, 250.0, 20.0, 1),
    ("Q2_top_right",       0.0,    0.0,  250.0, 250.0, 10.0, 2),
    ("Q3_bottom_left",  -250.0, -250.0,  250.0, 250.0,  5.0, 3),
    ("Q4_bottom_right",    0.0, -250.0,  250.0, 250.0,  2.0, 4),
]

# ---------------------------------------------------------------------------
# Helper: peak RSS memory in MB (Linux: ru_maxrss is kB)
# ---------------------------------------------------------------------------
def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# PART 1: 4-quadrant mesh
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("PART 1: 4-quadrant mesh")
print("="*65)

t0_total = time.perf_counter()
tracemalloc.start()

# --- Gmsh build ---
t0 = time.perf_counter()

gmsh_api.initialize()
gmsh_api.option.setNumber("General.Terminal", 0)
gmsh_api.model.add("quadrant_4zones")

box_tags = []
for name, x0, y0, dx, dy, h, mat_id in QUADRANTS:
    t = gmsh_api.model.occ.addBox(x0, y0, BOX_Z0, dx, dy, BOX_LZ)
    box_tags.append(t)

# Fragment: makes all 4 volumes share internal faces (conforming mesh)
input_dimtags = [(3, t) for t in box_tags]
gmsh_api.model.occ.fragment(input_dimtags, [])
gmsh_api.model.occ.synchronize()

# Identify each resulting volume by its centroid and assign physical groups
all_vols = gmsh_api.model.getEntities(3)
for dim, vtag in all_vols:
    bb = gmsh_api.model.getBoundingBox(dim, vtag)
    cx = (bb[0] + bb[3]) / 2.0
    cy = (bb[1] + bb[4]) / 2.0
    for name, x0, y0, dx, dy, h, mat_id in QUADRANTS:
        if (x0 < cx < x0 + dx) and (y0 < cy < y0 + dy):
            gmsh_api.model.addPhysicalGroup(3, [vtag], tag=mat_id, name=name)
            break

# Tag all surfaces (outer walls + shared internal interfaces)
all_surfs = [t for _, t in gmsh_api.model.getEntities(2)]
gmsh_api.model.addPhysicalGroup(2, all_surfs, tag=1, name="all_surfaces")

# Box field per quadrant (VOut=1e6 so Min field picks correct h everywhere)
field_ids = []
for name, x0, y0, dx, dy, h, mat_id in QUADRANTS:
    fid = gmsh_api.model.mesh.field.add("Box")
    gmsh_api.model.mesh.field.setNumber(fid, "VIn",       h)
    gmsh_api.model.mesh.field.setNumber(fid, "VOut",      1e6)
    gmsh_api.model.mesh.field.setNumber(fid, "XMin",      x0)
    gmsh_api.model.mesh.field.setNumber(fid, "XMax",      x0 + dx)
    gmsh_api.model.mesh.field.setNumber(fid, "YMin",      y0)
    gmsh_api.model.mesh.field.setNumber(fid, "YMax",      y0 + dy)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMin",      BOX_Z0)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMax",      BOX_Z0 + BOX_LZ)
    gmsh_api.model.mesh.field.setNumber(fid, "Thickness", 0.0)
    field_ids.append(fid)

fid_min = gmsh_api.model.mesh.field.add("Min")
gmsh_api.model.mesh.field.setNumbers(fid_min, "FieldsList", field_ids)
gmsh_api.model.mesh.field.setAsBackgroundMesh(fid_min)

gmsh_api.option.setNumber("Mesh.Algorithm3D", 1)
gmsh_api.model.mesh.generate(3)

t_gmsh = time.perf_counter() - t0
print(f"  Gmsh mesh generation:  {t_gmsh:.2f}s")

# --- DOLFINx conversion ---
t0 = time.perf_counter()
mesh_data  = gmshio.model_to_mesh(gmsh_api.model, comm, rank=0, gdim=3)
mesh       = mesh_data.mesh
cell_tags  = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
gmsh_api.finalize()
t_dolfinx = time.perf_counter() - t0
print(f"  DOLFINx conversion:    {t_dolfinx:.2f}s")

ncells = mesh.topology.index_map(mesh.topology.dim).size_local
nnodes = mesh.geometry.x.shape[0]

# DG0 material_id for reliable ParaView coloring
tdim  = mesh.topology.dim
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

for name, x0, y0, dx, dy, h, mat_id in QUADRANTS:
    mask = (
        (midpoints[:, 0] >= x0) & (midpoints[:, 0] <= x0 + dx) &
        (midpoints[:, 1] >= y0) & (midpoints[:, 1] <= y0 + dy)
    )
    material_id.x.array[mask] = float(mat_id)
    target_h.x.array[mask]    = h

# --- XDMF export ---
t0 = time.perf_counter()
with XDMFFile(comm, "quadrant_4zones.xdmf", "w") as xf:
    xf.write_mesh(mesh)
    xf.write_function(material_id)
    xf.write_function(target_h)
t_xdmf = time.perf_counter() - t0
print(f"  XDMF write:            {t_xdmf:.2f}s")

t_total_part1 = time.perf_counter() - t0_total
_, mem_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
rss_mb = peak_rss_mb()

print(f"\n  Total cells: {ncells:,}   nodes: {nnodes:,}")
for name, x0, y0, dx, dy, h, mat_id in QUADRANTS:
    n = int((material_id.x.array == float(mat_id)).sum())
    print(f"    id={mat_id} {name:<20} h={h:4.0f}nm  cells={n:>8,}")
print(f"\n  Total time: {t_total_part1:.2f}s   RSS peak: {rss_mb:.0f} MB")
print(f"  Output: quadrant_4zones.xdmf / .h5")


# ---------------------------------------------------------------------------
# PART 2: 4 uniform benchmark meshes
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("PART 2: uniform benchmark meshes")
print("="*65)

BENCHMARK_CASES = [
    ("benchmark_h20nm", 20.0),
    ("benchmark_h10nm", 10.0),
    ("benchmark_h5nm",   5.0),
    ("benchmark_h2nm",   2.0),
]

# Rough cell count estimate: V / (h³/(6√2))
BOX_VOL = 500.0 * 500.0 * 150.0
def estimate_cells(h):
    return int(BOX_VOL / (h**3 / (6 * np.sqrt(2))))

MAX_CELLS = 5_000_000  # skip if estimated > 5M to avoid OOM

bench_results = []

for label, h in BENCHMARK_CASES:
    est = estimate_cells(h)
    if est > MAX_CELLS:
        print(f"\n  [{label}]  h={h:.0f}nm  SKIPPED — estimated ~{est/1e6:.1f}M cells (>{MAX_CELLS/1e6:.0f}M limit)")
        bench_results.append({
            "label": label, "h": h, "skipped": True, "est_cells": est,
        })
        continue

    print(f"\n  [{label}]  h={h:.0f}nm  (estimated ~{est/1e3:.0f}k cells)  building...", flush=True)
    tracemalloc.start()
    t0_total_b = time.perf_counter()

    # Gmsh
    t0 = time.perf_counter()
    gmsh_api.initialize()
    gmsh_api.option.setNumber("General.Terminal", 0)
    gmsh_api.model.add(label)

    bt = gmsh_api.model.occ.addBox(-250.0, -250.0, 0.0, 500.0, 500.0, 150.0)
    gmsh_api.model.occ.synchronize()
    gmsh_api.model.addPhysicalGroup(3, [bt], tag=1, name="domain")
    bnd = gmsh_api.model.getBoundary([(3, bt)], oriented=False, combined=False)
    gmsh_api.model.addPhysicalGroup(2, [abs(b[1]) for b in bnd], tag=1, name="boundary")
    gmsh_api.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh_api.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh_api.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh_api.model.mesh.generate(3)
    t_gmsh_b = time.perf_counter() - t0

    # DOLFINx
    t0 = time.perf_counter()
    md   = gmshio.model_to_mesh(gmsh_api.model, comm, rank=0, gdim=3)
    msh  = md.mesh
    gmsh_api.finalize()
    t_dolfinx_b = time.perf_counter() - t0

    nc_b = msh.topology.index_map(msh.topology.dim).size_local
    nn_b = msh.geometry.x.shape[0]

    # XDMF
    t0 = time.perf_counter()
    with XDMFFile(comm, f"{label}.xdmf", "w") as xf:
        xf.write_mesh(msh)
    t_xdmf_b = time.perf_counter() - t0

    t_total_b = time.perf_counter() - t0_total_b
    _, mem_peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_b = peak_rss_mb()

    print(f"    cells={nc_b:>8,}  nodes={nn_b:>8,}")
    print(f"    gmsh={t_gmsh_b:.2f}s  dolfinx={t_dolfinx_b:.2f}s  xdmf={t_xdmf_b:.2f}s  total={t_total_b:.2f}s")
    print(f"    RSS peak: {rss_b:.0f} MB")

    bench_results.append({
        "label": label, "h": h, "skipped": False,
        "ncells": nc_b, "nnodes": nn_b,
        "t_gmsh": t_gmsh_b, "t_dolfinx": t_dolfinx_b,
        "t_xdmf": t_xdmf_b, "t_total": t_total_b,
        "rss_mb": rss_b,
    })


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("BENCHMARK SUMMARY")
print("="*65)
hdr = f"{'label':<20} {'h':>5}  {'cells':>10}  {'t_gmsh':>8}  {'t_total':>8}  {'RSS':>8}"
print(hdr)
print("-" * len(hdr))

for r in bench_results:
    if r["skipped"]:
        print(f"  {r['label']:<18} {r['h']:>4.0f}nm  {'SKIPPED':>10}  "
              f"  est ~{r['est_cells']/1e6:.1f}M cells")
    else:
        print(f"  {r['label']:<18} {r['h']:>4.0f}nm  {r['ncells']:>10,}  "
              f"{r['t_gmsh']:>7.2f}s  {r['t_total']:>7.2f}s  {r['rss_mb']:>6.0f}MB")

# Scaling analysis
valid = [r for r in bench_results if not r["skipped"]]
if len(valid) > 1:
    print(f"\nCell-count scaling (observed vs expected ~8x per 2x refinement):")
    for i in range(1, len(valid)):
        r0, r1 = valid[i-1], valid[i]
        ratio_h = r0["h"] / r1["h"]
        ratio_c = r1["ncells"] / r0["ncells"]
        print(f"  h {r0['h']:.0f}→{r1['h']:.0f} nm:  "
              f"{ratio_c:.1f}x more cells  (expected {ratio_h**3:.1f}x)")
