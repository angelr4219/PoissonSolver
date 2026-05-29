"""
local_refinement_sweep.py

Mesh density sweep over the fine region of the 3D local-refinement test box.
Keeps h_coarse fixed and varies h_fine to verify that the Gmsh Box field
actually delivers the requested element sizes.

Sweep cases (h_fine in nm):   20, 10, 5, 2, 1
Coarse region (fixed):         h_coarse = 20 nm  (~0.05 pts/nm)

"Points per nm" = 1 / h_fine
    h=20 nm  →  0.05 pts/nm  (same as coarse, baseline)
    h=10 nm  →  0.10 pts/nm
    h= 5 nm  →  0.20 pts/nm
    h= 2 nm  →  0.50 pts/nm
    h= 1 nm  →  1.00 pts/nm

One XDMF/H5 pair is written per case so you can inspect each in ParaView.
A summary table is printed at the end.

Box:         500 x 500 x 150 nm  (x: -250→250, y: -250→250, z: 0→150)
Fine region:  80 x  80 x  30 nm  (x:  -40→ 40, y:  -40→ 40, z: 45→ 75)
"""

import numpy as np
from mpi4py import MPI
import gmsh as gmsh_api
from dolfinx.io import gmsh as gmshio, XDMFFile
from dolfinx.fem import functionspace, Function

# ---------------------------------------------------------------------------
# Fixed geometry (nm)
# ---------------------------------------------------------------------------
H_COARSE = 20.0

BOX_X0, BOX_Y0, BOX_Z0 = -250.0, -250.0,  0.0
BOX_LX, BOX_LY, BOX_LZ =  500.0,  500.0, 150.0

FINE_X0, FINE_Y0, FINE_Z0 = -40.0, -40.0, 45.0
FINE_X1, FINE_Y1, FINE_Z1 =  40.0,  40.0, 75.0
FINE_VOL = (FINE_X1-FINE_X0) * (FINE_Y1-FINE_Y0) * (FINE_Z1-FINE_Z0)  # nm³

H_FINE_SWEEP = [20.0, 10.0, 5.0, 2.0, 1.0]  # nm

comm = MPI.COMM_WORLD


def build_mesh(h_fine: float, h_coarse: float) -> tuple:
    gmsh_api.initialize()
    gmsh_api.option.setNumber("General.Terminal", 0)  # quiet
    gmsh_api.model.add("sweep_box")

    box_tag = gmsh_api.model.occ.addBox(BOX_X0, BOX_Y0, BOX_Z0,
                                         BOX_LX, BOX_LY, BOX_LZ)
    gmsh_api.model.occ.synchronize()

    gmsh_api.model.addPhysicalGroup(3, [box_tag], tag=1, name="domain")
    bnd = gmsh_api.model.getBoundary([(3, box_tag)], oriented=False, combined=False)
    gmsh_api.model.addPhysicalGroup(2, [abs(b[1]) for b in bnd], tag=1, name="boundary")

    fid = gmsh_api.model.mesh.field.add("Box")
    gmsh_api.model.mesh.field.setNumber(fid, "VIn",       h_fine)
    gmsh_api.model.mesh.field.setNumber(fid, "VOut",      h_coarse)
    gmsh_api.model.mesh.field.setNumber(fid, "XMin",      FINE_X0)
    gmsh_api.model.mesh.field.setNumber(fid, "XMax",      FINE_X1)
    gmsh_api.model.mesh.field.setNumber(fid, "YMin",      FINE_Y0)
    gmsh_api.model.mesh.field.setNumber(fid, "YMax",      FINE_Y1)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMin",      FINE_Z0)
    gmsh_api.model.mesh.field.setNumber(fid, "ZMax",      FINE_Z1)
    gmsh_api.model.mesh.field.setNumber(fid, "Thickness", 0.0)
    gmsh_api.model.mesh.field.setAsBackgroundMesh(fid)

    gmsh_api.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh_api.model.mesh.generate(3)

    mesh_data = gmshio.model_to_mesh(gmsh_api.model, comm, rank=0, gdim=3)
    mesh = mesh_data.mesh
    gmsh_api.finalize()
    return mesh


def label_and_export(mesh, h_fine: float, h_coarse: float, label: str) -> dict:
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

    in_fine = (
        (midpoints[:, 0] >= FINE_X0) & (midpoints[:, 0] <= FINE_X1) &
        (midpoints[:, 1] >= FINE_Y0) & (midpoints[:, 1] <= FINE_Y1) &
        (midpoints[:, 2] >= FINE_Z0) & (midpoints[:, 2] <= FINE_Z1)
    )

    V0 = functionspace(mesh, ("DG", 0))

    material_id = Function(V0, name="material_id")
    material_id.x.array[:] = np.where(in_fine, 1.0, 2.0)

    target_h = Function(V0, name="target_h_nm")
    target_h.x.array[:] = np.where(in_fine, h_fine, h_coarse)

    fname = f"locally_refined_box_{label}.xdmf"
    with XDMFFile(comm, fname, "w") as xf:
        xf.write_mesh(mesh)
        xf.write_function(material_id)
        xf.write_function(target_h)

    n_fine   = int(in_fine.sum())
    n_coarse = ncells - n_fine

    # Estimate actual mean edge length inside fine region from cell volume
    # V_tet ≈ h³/(6√2), so h_actual ≈ (6√2 * V/N)^(1/3)
    # Simpler: use FINE_VOL / n_fine as proxy
    vol_per_tet_fine = FINE_VOL / max(n_fine, 1)
    h_actual = (6.0 * np.sqrt(2) * vol_per_tet_fine) ** (1.0 / 3.0)

    return {
        "h_fine_target": h_fine,
        "pts_per_nm":    1.0 / h_fine,
        "n_total":       ncells,
        "n_fine":        n_fine,
        "n_coarse":      n_coarse,
        "h_actual_nm":   h_actual,
        "file":          fname,
    }


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Mesh density sweep   h_coarse = {H_COARSE} nm (fixed)")
print(f"Fine region: {FINE_X0}→{FINE_X1} x {FINE_Y0}→{FINE_Y1} x {FINE_Z0}→{FINE_Z1} nm")
print(f"{'='*70}")

results = []
for h_fine in H_FINE_SWEEP:
    label = f"hfine{int(h_fine)}nm" if h_fine >= 1 else f"hfine{h_fine:.1f}nm"
    print(f"\n[Case] h_fine = {h_fine:5.1f} nm  ({1/h_fine:.2f} pts/nm)  building mesh...", flush=True)
    mesh = build_mesh(h_fine, H_COARSE)
    stats = label_and_export(mesh, h_fine, H_COARSE, label)
    results.append(stats)
    print(f"       cells total={stats['n_total']:>7,}  fine={stats['n_fine']:>7,}  "
          f"coarse={stats['n_coarse']:>7,}  h_actual≈{stats['h_actual_nm']:.2f} nm  → {stats['file']}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"SUMMARY TABLE")
print(f"{'='*70}")
header = f"{'h_fine':>8}  {'pts/nm':>7}  {'n_total':>9}  {'n_fine':>9}  {'n_coarse':>9}  {'h_actual':>9}  {'file'}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['h_fine_target']:>7.1f}nm  "
          f"{r['pts_per_nm']:>7.2f}  "
          f"{r['n_total']:>9,}  "
          f"{r['n_fine']:>9,}  "
          f"{r['n_coarse']:>9,}  "
          f"{r['h_actual_nm']:>8.2f}nm  "
          f"{r['file']}")

print(f"\nExpected fine-cell scaling: n_fine ∝ (1/h_fine)³")
for i in range(1, len(results)):
    r0, r1 = results[i-1], results[i]
    h_ratio    = r0["h_fine_target"] / r1["h_fine_target"]
    cell_ratio = r1["n_fine"] / max(r0["n_fine"], 1)
    expected   = h_ratio ** 3
    print(f"  h {r0['h_fine_target']:.0f}→{r1['h_fine_target']:.0f} nm: "
          f"cell ratio = {cell_ratio:.1f}x  (expected ~{expected:.1f}x)")
