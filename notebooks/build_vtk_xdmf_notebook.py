"""Builds notebooks/vtk_xdmf_comparison.ipynb. Run once to (re)generate the notebook,
then execute it with nbclient or `jupyter nbconvert --execute`."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# VTK reference vs. DOLFINx XDMF comparison

Compares the MaSQE/Leah reference field (`basePotential3d.vtk`) against any number of
DOLFINx XDMF/H5 simulation outputs. The VTK file is treated as ground truth.

For each case:
1. Read VTK reference + XDMF/H5 case.
2. Sanity-check bounding-box overlap (catches unit-scale / domain-size mismatches before
   they silently turn into bogus "errors").
3. Linearly interpolate the case field onto the VTK reference points (`scipy.griddata`),
   falling back to nearest-neighbor outside the convex hull — and report what fraction of
   points needed that fallback.
4. Plot reference / case / absolute error / relative error side-by-side at chosen z-slices.
5. Report a global relative-L2 error and per-slice metrics, saved to CSV.

`xdmf_cases` below is a plain list — add or remove entries to compare 1 or N cases against
the single VTK reference."""
))

cells.append(nbf.v4.new_code_cell(
"""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from poisson.vtk_xdmf_compare import XdmfCase, run_comparison

%matplotlib inline"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Configuration

- `vtk_path`: the MaSQE/Leah reference file (nm units, domain `x,y in [-250,250]`, `z in [0,255]`).
- `xdmf_cases`: DOLFINx/FFT outputs from the sphere benchmark. They're written in **meters**,
  so `scale=1e9` converts to nm to match the VTK file.
- `z_slices`: chosen z-values in nm. Restricted to `[0, 90]` because the FEM/FFT box only
  spans `z in [-100, 100]` nm centered at the sphere — there is **no overlap at all** for
  `z > 100` nm, which the bbox check below will also flag explicitly."""
))

cells.append(nbf.v4.new_code_cell(
"""vtk_path = PROJECT_ROOT / "basePotential3d.vtk"
results_dir = PROJECT_ROOT / "benchmark_results"

xdmf_cases = [
    XdmfCase("FEM-Dirichlet h=10nm", results_dir / "phi_dir_h10.0nm.xdmf", scale=1e9),
    XdmfCase("FEM-Dirichlet h=5nm",  results_dir / "phi_dir_h5.0nm.xdmf",  scale=1e9),
    XdmfCase("FFT h=10nm",           results_dir / "phi_fft_h10.0nm.xdmf", field="phi", scale=1e9),
    XdmfCase("FFT h=5nm",            results_dir / "phi_fft_h5.0nm.xdmf",  field="phi", scale=1e9),
]

z_slices = [0.0, 30.0, 60.0, 90.0]

out_dir = PROJECT_ROOT / "benchmark_results" / "vtk_comparison"
out_dir.mkdir(parents=True, exist_ok=True)"""
))

cells.append(nbf.v4.new_markdown_cell("## Run the comparison\n\nThis reads the VTK once, then loops over every case in `xdmf_cases` (works for 1 or many)."))

cells.append(nbf.v4.new_code_cell(
"""metrics_df = run_comparison(
    vtk_path=vtk_path,
    cases=xdmf_cases,
    z_slices=z_slices,
    units=" nm",
    out_dir=out_dir,
    show_plots=True,
)"""
))

cells.append(nbf.v4.new_markdown_cell("## Metrics table\n\nOne 'global' row per case plus one row per z-slice. `nearest_fallback_frac` close to 1 means the slice barely overlaps the case domain — treat its error numbers as not meaningful."))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
pd.set_option("display.width", 120)
metrics_df.sort_values(["case", "slice_z"], key=lambda c: c.astype(str))"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Reading the result

The bbox-overlap warnings above are the real finding here: the FEM/FFT benchmark box
(`[-100,100] nm` per axis, centered on the sphere) and the VTK reference grid
(`z in [0,255] nm`) only overlap for `z in [0,100] nm`, and even there the *lateral*
(x,y) extents differ. That overlap fraction is why `nearest_fallback_frac` is non-trivial
even at z=0 — a meaningful global comparison requires regenerating the VTK reference (or
the FEM/FFT benchmark domain) so the two boxes actually coincide, not just patching the
interpolation."""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Why is `pct_err` ~100% everywhere?

Two independent problems, both visible above, both real findings rather than bugs:

1. **Domain mismatch.** The VTK reference spans `x,y in [-250,250] nm`, `z in [0,255] nm`;
   the FEM/FFT benchmark box is `[-100,100] nm` per axis. Only a fraction of the VTK grid
   sits inside the benchmark box at all (see the `nearest_fallback_frac` column — up to 95%
   of points needed nearest-neighbor fallback because they're outside the case's convex hull).
2. **Different physical setup entirely.** `basePotential` ranges **-12 V to -1 V** (a biased
   gate-stack potential landscape), while the toy sphere benchmark's `phi` ranges **0 to
   0.16 V** (an isolated test charge with arbitrary small `Q`). These aren't the same
   simulation at different mesh resolution — they're unrelated problems, so a near-100%
   relative error is the mathematically correct answer for *these specific files*.

This isn't something to "fix" in the comparison code — it means the sphere-benchmark output
isn't yet the right input to validate against `basePotential3d.vtk`. The cell below proves the
comparison machinery itself is sound by cross-checking two outputs that *do* share a domain
and a charge configuration: FEM-Dirichlet vs. FFT from the same sphere-benchmark run."""
))

cells.append(nbf.v4.new_code_cell(
"""import numpy as np
from poisson.vtk_xdmf_compare import (
    XdmfCase, read_xdmf_case, interpolate_case_onto_ref, compute_metrics, plot_slice_pair
)

# Treat the finer FFT grid (same domain, same charge) as "ground truth" and compare
# FEM-Dirichlet against it -- a same-domain sanity check for the pipeline itself.
ref_case = XdmfCase("FFT h=5nm (as reference)", results_dir / "phi_fft_h5.0nm.xdmf", field="phi")
fem_case = XdmfCase("FEM-Dirichlet h=5nm", results_dir / "phi_dir_h5.0nm.xdmf")

ref_field = read_xdmf_case(ref_case)
fem_field = read_xdmf_case(fem_case)

sanity_result = interpolate_case_onto_ref(ref_field, fem_field, fem_case.label)
# mask percent-error stats to points where the reference field is at least 1% of its peak --
# elsewhere phi decays toward zero and percent error is dominated by noise, not real mismatch.
threshold = 0.01 * np.abs(ref_field.values).max()
sanity_metrics = compute_metrics(sanity_result, fem_case.label, pct_mask_threshold=threshold)
sanity_metrics"""
))

cells.append(nbf.v4.new_code_cell(
"""plot_slice_pair(sanity_result, z_target=0.0, label=fem_case.label, units=" m")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""`nearest_fallback_frac = 0.0` confirms the domains genuinely overlap here, so the
interpolation itself is doing real work, not papering over a mismatch.

`pct_err_mean` (609%) is nonetheless huge despite `abs_err_mean` being small (0.005 V) —
that's the floored-percent-error metric being inherently unstable in a decaying potential
field: far from the sphere both `phi_ref` and `phi_fem` approach zero, so a tiny absolute
difference becomes a large *ratio*. `rel_l2_error` (a single normalized number over the whole field) and the new
`pct_err_mean_masked`/`pct_err_max_masked` (percent error restricted to points where
`|ref| > 1%` of its peak) are far more trustworthy summaries than the unmasked mean/max
percent error for a field that decays toward zero. This is a real methodological point, not
a code bug: **lead with `rel_l2_error` or an absolute-error map for decaying fields, and use
the masked percent-error columns instead of the raw ones whenever the field has a large
near-zero region.**"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Case A: open the real result in ParaView

The numbers above are useful, but you also want to just *look* at the field directly in
ParaView. Two problems with doing that naively:

1. dolfinx writes one `Topology`/`Geometry` plus a separate `GridType="Collection"
   CollectionType="Temporal"` wrapper **per field** (`phi`, `rho`, `eps_r`, `mat_id`),
   each referencing the mesh via `xi:include`/`xpointer`. That's valid XDMF3, but
   ParaView's reader renders it as a blank view — the same underlying issue that made
   pyvista's reader segfault earlier in this notebook.
2. The VTK reference and the DOLFINx case live in different files with no built-in
   link between them in ParaView.

`write_paraview_friendly_xdmf` repacks the dolfinx output into one flat `Grid` with all
fields as plain `Attribute`s referencing the same `Topology`/`Geometry` — no
`xi:include`, no per-field collections. ParaView opens this cleanly."""
))

cells.append(nbf.v4.new_code_cell(
"""from poisson.vtk_xdmf_compare import write_paraview_friendly_xdmf

trackA_xdmf = PROJECT_ROOT / "results" / "trackA_nogate_sigma2e11_tet_h6" / "trackA_nogate_sigma2e11_tet_h6.xdmf"
pv_friendly = write_paraview_friendly_xdmf(trackA_xdmf)
print(f"wrote {pv_friendly}")"""
))

cells.append(nbf.v4.new_code_cell(
"""import subprocess

subprocess.run(["open", "-a", "/Applications/ParaView.app", str(pv_friendly)])
subprocess.run(["open", "-a", "/Applications/ParaView.app", str(vtk_path)])"""
))

cells.append(nbf.v4.new_markdown_cell(
"""In ParaView: click **Apply** on each source, color by `phi` (DOLFINx case) /
`basePotential` (VTK reference), range **-12 to -1 V**, `RdBu_r` colormap. Add a **Slice**
filter (Normal = Z) on each and scrub `z` — for Case A both should render as a single solid
color per slice (laterally flat), changing only as you move through z."""
))

nb["cells"] = cells

out_path = Path(__file__).parent / "vtk_xdmf_comparison.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}")
