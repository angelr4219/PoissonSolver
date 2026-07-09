# June 26 MaSQE vs DOLFINx comparison

## Folder layout

| Folder | Contents |
|--------|----------|
| `vtk/` | MaSQE reference VTK potential(s) |
| `fem/` | DOLFINx XDMF + H5 outputs |
| `notebooks/` | Comparison notebook(s) |
| `outputs/` | Per-attempt comparison outputs |
| `notes/` | Manifests and run notes |

## Primary comparison

Reference: `vtk/basePotential3d.vtk`
- Grid: 51 × 51 × 111 (rectilinear)
- x: [−250, +250] nm, y: [−250, +250] nm, z: [0, 255] nm
- Scalar: `basePotential`, range [−12, −1] V

## Unit convention

VTK coordinates are in **nm**.
DOLFINx mesh coordinates are in **metres** (multiply by 1e9 to get nm).

## ⚠ Do not compare basePotential3d.vtk against the staged FEM benchmarks

`fem/benchmark_results/` and `fem/disk_Q1e_p3_dense/` are staged for solver
verification only.  They solve **different physics** from the MaSQE reference:

| Folder | Problem | phi range | Domain |
|--------|---------|-----------|--------|
| `benchmark_results/` | Point-charge in vacuum | 0 – 0.16 V | ±100 nm cube |
| `disk_Q1e_p3_dense/` | Charged disk | ~mV | ±300 nm × ±200 nm |
| **MaSQE reference** | **Device bias, layered stack** | **−12 to −1 V** | **±250 nm × 0–255 nm** |

A comparison plot between these will be mathematically valid but physically
meaningless.  Pretty plots from mismatched physics are worse than no plots
because they look convincing.

## Coordinate conventions (z-axis)

VTK z=0 nm is the **gate surface** (φ = −1 V).  VTK z=255 nm is the **bulk contact** (φ = −12 V).

DOLFINx `layered_disk_gate_box` is built with z ∈ [0, 255 nm] where z=0 is the
`bottom_voltage` (bulk) and z=H is the `disk_voltage` (gate). The z-axes are **flipped**.

The comparison notebook applies `z_dolfinx = 255 − z_vtk` automatically before
querying the FEM solution. No manual transform is required when running the notebook.

## Layer ordering for matching solve

The physical stack from the VTK coordinate frame (gate at z=0, bulk at z=255 nm):

| VTK z [nm] | Material   | ε_r   | Role     |
|------------|------------|-------|----------|
| 0 – 50     | Si70Ge30   | 13.05 | spacer   |
| 50 – 55    | sSi        | 11.70 | QW       |
| 55 – 255   | Si70Ge30   | 13.05 | buffer   |

In DOLFINx (z_dolfinx = 255 − z_vtk), the **same physical stack** maps to:

| DOLFINx z [nm] | Material | ε_r   |
|----------------|----------|-------|
| 0 – 200        | Si70Ge30 | 13.05 |
| 200 – 205      | sSi      | 11.70 |
| 205 – 255      | Si70Ge30 | 13.05 |

The `--layer` flags for the DOLFINx solve must use these DOLFINx-frame z-values.
(The default layers in `layered_disk_gate_box.py` place the QW at z=50–55 nm, which
is the wrong position.  Since both SiGe layers share ε=13.05, the error is < 1%
for a diagnostic h=20 nm run, but correct layers are required for production.)

## Matching device-scale FEM case (target)

Confirmed coordinate system (from reading basePotential3d.vtk):
- VTK z=0: φ = −1.0000 V (gate Dirichlet) ✓
- VTK z=255 nm: φ = −12.0000 V (bulk Dirichlet) ✓

No existing `Results/` case is usable — all disk-gate cases have only a 10 nm disk at
−1 V on the top face; the rest of the top is Neumann.  Run this command inside Docker:

```bash
docker run --rm -v "$(pwd)":/work -w /work \
  ghcr.io/jorgensd/dolfinx_mpc:latest \
  python3 main.py layered_disk_gate_box \
    --outdir Results/caseA_base_neg1_bottom_neg12 \
    --basename phi \
    --Lx 500e-9 --Ly 500e-9 --H 255e-9 \
    --h 20e-9 \
    --disk_R 1000e-9 \
    --bottom_voltage -12.0 \
    --disk_voltage   -1.0 \
    --layer 0,200e-9,13.05 \
    --layer 200e-9,205e-9,11.7 \
    --layer 205e-9,255e-9,13.05 \
    --cell_type hex \
    --hard_exit
```

`--disk_R 1000e-9` (much larger than the 250 nm half-domain) makes the entire top
surface fall inside the "disk" → uniform Dirichlet −1 V on the full top face.

The `--layer` arguments use DOLFINx z-coordinates so the sSi QW (5 nm) sits at the
correct position relative to the gate.

After the run, stage and configure:

```bash
python3 scripts/stage_june26.py \
  --fem Results/caseA_base_neg1_bottom_neg12/phi.xdmf \
        Results/caseA_base_neg1_bottom_neg12/phi.h5

# Then in june26_vtk_xdmf_compare.py, add to CASES:
# {
#   "label": "CaseA_h20nm",
#   "xdmf":  "june 26/fem/Results/caseA_base_neg1_bottom_neg12/phi.xdmf",
#   "h5":    "june 26/fem/Results/caseA_base_neg1_bottom_neg12/phi.h5",
#   "field": "phi",
# }
```

Planned refinement ladder (add one at a time; compare before moving on):

| Case | h [nm] | Notes |
|------|--------|-------|
| A    | 20     | Diagnostic — establishes shape, not accuracy |
| B    | 10     | Coarse reference |
| C    |  5     | First serious comparison |
| D    |  3     | Better |
| E    |  2     | Aggressive (uniform); use local refinement below this |
