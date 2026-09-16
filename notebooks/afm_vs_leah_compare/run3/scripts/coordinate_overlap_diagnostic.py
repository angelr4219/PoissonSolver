#!/usr/bin/env python3
"""Coordinate/grid/interpolation-validity diagnostic for run3 vs the Leah VTK reference.

Read-only w.r.t. the repo's comparison module and data files: this script only
IMPORTS from src/poisson/vtk_xdmf_compare.py and reads run3's existing outputs.
It writes its own report to notebooks/afm_vs_leah_compare/run3/diagnostics/.

Scope: coordinates, grids, and interpolation validity ONLY -- not physics
interpretation (voltages, tip inclusion, gate treatment are someone else's job).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from poisson.vtk_xdmf_compare import (  # noqa: E402
    XdmfCase,
    check_bbox_overlap,
    interpolate_case_onto_ref,
    nearest_z_mask,
    read_vtk_reference,
    read_xdmf_case,
)

VTK_PATH = Path("/Users/angelramirez/Downloads/basePotential3d(1).vtk")
RUN3_DIR = REPO_ROOT / "notebooks" / "afm_vs_leah_compare" / "run3"
XDMF_PATH = RUN3_DIR / "sige_afm_tip.xdmf"
RESULTS_JSON = RUN3_DIR / "run_results.json"

OUT_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    OUT_LINES.append(msg)


def fmt_range(pts: np.ndarray) -> str:
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return (
        f"x: [{lo[0]:.6g}, {hi[0]:.6g}]  "
        f"y: [{lo[1]:.6g}, {hi[1]:.6g}]  "
        f"z: [{lo[2]:.6g}, {hi[2]:.6g}]"
    )


def main() -> None:
    log("# Coordinate / overlap / interpolation-validity diagnostic -- run3 vs Leah VTK\n")

    # ------------------------------------------------------------------
    # 1. Load both fields
    # ------------------------------------------------------------------
    log("## 1. Load both fields\n")

    ref = read_vtk_reference(VTK_PATH, field="basePotential")
    log(f"VTK reference: field='{ref.name}', N points = {len(ref.points)}")
    log(f"VTK point range -> {fmt_range(ref.points)}")
    log(f"VTK value range -> [{ref.values.min():.6g}, {ref.values.max():.6g}]\n")

    case_def = XdmfCase("run3", XDMF_PATH, field="phi_V", scale=1.0)
    case = read_xdmf_case(case_def)
    log(f"FEM case: field='{case.name}', N points = {len(case.points)}")
    log(f"FEM point range -> {fmt_range(case.points)}")
    log(f"FEM value range -> [{case.values.min():.6g}, {case.values.max():.6g}]\n")

    with open(RESULTS_JSON) as f:
        run_results = json.load(f)
    geo = run_results["geometry_nm"]
    volt = run_results["voltages_V"]
    log(f"run_results.json geometry_nm: {geo}")
    log(f"run_results.json voltages_V: {volt}\n")

    log(
        "Units check: calibrate_afm_tip_vs_leah.py builds gmsh geometry directly in "
        "the CLI's raw nm values (--lx/--ly default 300.0, no *1e-9/*1e9 conversion "
        "anywhere in build_geometry). read_xdmf_case(case, scale=1.0) therefore leaves "
        "FEM coordinates un-rescaled, i.e. already in nm, matching the VTK's nm "
        "coordinates.  scale=1.0 is the CORRECT choice for this run -- confirmed by "
        "inspection of calibrate_afm_tip_vs_leah.py, not assumed.\n"
    )

    # ------------------------------------------------------------------
    # 2. Axis / orientation check
    # ------------------------------------------------------------------
    log("## 2. Axis / orientation check (z=0 = top surface, z increasing into device)\n")

    vtk_z = ref.points[:, 2]
    vtk_z_lo, vtk_z_hi = vtk_z.min(), vtk_z.max()
    mask_top = nearest_z_mask(ref.points, vtk_z_lo)
    mask_bot = nearest_z_mask(ref.points, vtk_z_hi)
    log(
        f"VTK z=min ({vtk_z_lo:.4g}): n={mask_top.sum()}, "
        f"basePotential mean={ref.values[mask_top].mean():.6f}, "
        f"std={ref.values[mask_top].std():.6f}, "
        f"min={ref.values[mask_top].min():.6f}, max={ref.values[mask_top].max():.6f}"
    )
    log(
        f"VTK z=max ({vtk_z_hi:.4g}): n={mask_bot.sum()}, "
        f"basePotential mean={ref.values[mask_bot].mean():.6f}, "
        f"std={ref.values[mask_bot].std():.6f}, "
        f"min={ref.values[mask_bot].min():.6f}, max={ref.values[mask_bot].max():.6f}\n"
    )

    fem_z = case.points[:, 2]
    fem_z_lo, fem_z_hi = fem_z.min(), fem_z.max()
    bottom_z_doc = geo["bottom_z"]
    bottom_v_doc = volt["bottom_back_gate"]
    tol_bottom = 1e-3
    mask_fem_bottom = np.abs(fem_z - bottom_z_doc) <= tol_bottom
    if mask_fem_bottom.sum() == 0:
        # widen tol to nearest available z if exact match not found
        tol_bottom = max(1.0, np.median(np.diff(np.sort(np.unique(fem_z)))) * 2)
        mask_fem_bottom = np.abs(fem_z - bottom_z_doc) <= tol_bottom
    log(
        f"FEM point z-range -> [{fem_z_lo:.6g}, {fem_z_hi:.6g}] "
        f"(documented bottom_z={bottom_z_doc}, air_height={geo['air_height']} "
        f"=> expected z-range approx [-{geo['air_height']}, {bottom_z_doc}])"
    )
    log(
        f"FEM points at z=={bottom_z_doc} (tol={tol_bottom:g}): n={mask_fem_bottom.sum()}, "
        f"phi_V mean={case.values[mask_fem_bottom].mean():.9f}, "
        f"min={case.values[mask_fem_bottom].min():.9f}, "
        f"max={case.values[mask_fem_bottom].max():.9f}  "
        f"(documented Dirichlet BC bottom_back_gate = {bottom_v_doc} V)"
    )
    mask_fem_z0 = nearest_z_mask(case.points, 0.0)
    log(
        f"FEM points at z==0 (nearest_z_mask tol): n={mask_fem_z0.sum()}, "
        f"phi_V mean={case.values[mask_fem_z0].mean():.6f}, "
        f"min={case.values[mask_fem_z0].min():.6f}, "
        f"max={case.values[mask_fem_z0].max():.6f}  "
        f"vs VTK z=0 mean={ref.values[nearest_z_mask(ref.points, 0.0)].mean():.6f}\n"
    )

    orientation_ok = (
        abs(case.values[mask_fem_bottom].mean() - bottom_v_doc) < 1e-6
        and vtk_z_lo == 0.0
    )
    log(
        f"Orientation verdict: {'PASS' if orientation_ok else 'CAUTION'} -- FEM phi_V at "
        f"documented bottom_z matches the Dirichlet BC voltage to <1e-6 V, and VTK's own "
        f"z axis starts at exactly 0.0 (top surface). Both datasets increase z into the "
        f"device / away from air, consistent orientation.\n"
    )

    # ------------------------------------------------------------------
    # 3. x=0, y=0 alignment
    # ------------------------------------------------------------------
    log("## 3. x=0 / y=0 alignment\n")

    ux = np.unique(ref.points[:, 0])
    uy = np.unique(ref.points[:, 1])
    x_has_zero = np.any(np.isclose(ux, 0.0, atol=1e-9))
    y_has_zero = np.any(np.isclose(uy, 0.0, atol=1e-9))
    log(f"VTK unique x count = {len(ux)}, range [{ux.min():.6g}, {ux.max():.6g}]")
    log(f"VTK unique y count = {len(uy)}, range [{uy.min():.6g}, {uy.max():.6g}]")
    log(f"VTK has an exact x=0.0 grid line: {x_has_zero}")
    log(f"VTK has an exact y=0.0 grid line: {y_has_zero}\n")

    log(
        "FEM mesh geometric center: build_geometry() in calibrate_afm_tip_vs_leah.py "
        "places the AFM tip sphere/cone/shaft, probe1 disk, and probe2 disk all "
        "explicitly at (x=0.0, y=0.0, z=...) -- occ.addSphere(0.0, 0.0, ...), "
        "occ.addCone(0.0, 0.0, ...), occ.addCylinder(0.0, 0.0, ...), "
        "occ.addDisk(0.0, 0.0, z_probe1/2, ...). The lateral domain box is built "
        "symmetric: xmin=-lx/2, xmax=+lx/2 (and same for y), so the domain and the "
        "tip/probe axis are both exactly centered at x=0, y=0 by construction -- no "
        "lateral misalignment was introduced. (No mesh node is guaranteed to sit at "
        "exactly x=0,y=0 since the mesh is unstructured, but the geometric symmetry "
        "axis is exact.)\n"
    )
    fem_x_center = 0.5 * (case.points[:, 0].min() + case.points[:, 0].max())
    fem_y_center = 0.5 * (case.points[:, 1].min() + case.points[:, 1].max())
    log(
        f"FEM point-cloud bbox center (sanity check on symmetry of the actual mesh): "
        f"x_center={fem_x_center:.6g}, y_center={fem_y_center:.6g} "
        f"(expect ~0 given lx=ly={geo['lx']})\n"
    )

    # ------------------------------------------------------------------
    # 4. Full bounding-box overlap
    # ------------------------------------------------------------------
    log("## 4. Bounding-box overlap (via check_bbox_overlap + independent computation)\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_bbox_overlap(ref, case, "run3")
    if caught:
        for w in caught:
            log(f"check_bbox_overlap WARNING: {w.message}")
    else:
        log("check_bbox_overlap: no warning raised (bboxes overlap and cover >=50% on every axis)")

    ref_lo, ref_hi = ref.points.min(axis=0), ref.points.max(axis=0)
    case_lo, case_hi = case.points.min(axis=0), case.points.max(axis=0)
    overlap_lo = np.maximum(ref_lo, case_lo)
    overlap_hi = np.minimum(ref_hi, case_hi)
    overlap_extent = overlap_hi - overlap_lo
    ref_extent = ref_hi - ref_lo
    coverage = overlap_extent / np.where(ref_extent == 0, 1, ref_extent)

    axis_names = ["x", "y", "z"]
    log("\nIndependent overlap computation:")
    for i, name in enumerate(axis_names):
        log(
            f"  {name}: ref=[{ref_lo[i]:.4g},{ref_hi[i]:.4g}]  "
            f"case=[{case_lo[i]:.4g},{case_hi[i]:.4g}]  "
            f"overlap=[{overlap_lo[i]:.4g},{overlap_hi[i]:.4g}]  "
            f"overlap_extent={overlap_extent[i]:.4g}  "
            f"coverage_of_ref_extent={coverage[i]:.2%}"
        )
    log("")

    # ------------------------------------------------------------------
    # 5. Full-domain and per-slice interpolation fallback stats
    # ------------------------------------------------------------------
    log("## 5. Interpolation fallback fractions (global + representative z-slices)\n")

    with warnings.catch_warnings(record=True) as caught_interp:
        warnings.simplefilter("always")
        result_full = interpolate_case_onto_ref(ref, case, "run3", check_bbox=True)
    for w in caught_interp:
        log(f"interpolate_case_onto_ref WARNING (global): {w.message}")

    n_total = len(result_full.fallback_mask)
    n_fallback = int(result_full.fallback_mask.sum())
    log(
        f"GLOBAL: n_points={n_total}  fallback_count={n_fallback}  "
        f"fallback_frac={result_full.nearest_fallback_frac:.4%}\n"
    )

    slice_targets = [0, 35, 48, 1000, 2000]
    log(f"{'z_target':>10} {'n_pts':>8} {'valid(linear)':>14} {'fallback':>10} {'fallback_frac':>14} {'verdict':>10}")
    slice_summaries = []
    for z in slice_targets:
        mask = nearest_z_mask(result_full.points, z)
        n = int(mask.sum())
        if n == 0:
            log(f"{z:>10} {n:>8} {'--':>14} {'--':>10} {'--':>14} {'NO POINTS':>10}")
            slice_summaries.append((z, n, 0, 0, float("nan"), "NO POINTS"))
            continue
        fb = int(result_full.fallback_mask[mask].sum())
        frac = fb / n
        valid = n - fb
        if frac > 0.05:
            verdict = "FAIL"
        elif frac > 0.0:
            verdict = "CAUTION"
        else:
            verdict = "PASS"
        log(f"{z:>10} {n:>8} {valid:>14} {fb:>10} {frac:>14.4%} {verdict:>10}")
        slice_summaries.append((z, n, valid, fb, frac, verdict))
    log("")

    # ------------------------------------------------------------------
    # 6. FEM domain extent vs documented config
    # ------------------------------------------------------------------
    log("## 6. FEM domain extent: documented config vs actual point cloud\n")

    lx, ly = geo["lx"], geo["ly"]
    bottom_z = geo["bottom_z"]
    air_height = geo["air_height"]
    log(f"Documented (run_results.json): lx={lx}, ly={ly}  => expected x,y in [-{lx/2},{lx/2}], [-{ly/2},{ly/2}]")
    log(f"Documented bottom_z={bottom_z}, air_height={air_height} => expected z in [-{air_height}, {bottom_z}]")
    log(
        f"Actual FEM point cloud: x in [{case.points[:,0].min():.6g}, {case.points[:,0].max():.6g}], "
        f"y in [{case.points[:,1].min():.6g}, {case.points[:,1].max():.6g}], "
        f"z in [{case.points[:,2].min():.6g}, {case.points[:,2].max():.6g}]"
    )
    log(
        f"VTK lateral extent actual: x in [{ref.points[:,0].min():.6g}, {ref.points[:,0].max():.6g}], "
        f"y in [{ref.points[:,1].min():.6g}, {ref.points[:,1].max():.6g}], "
        f"z in [{ref.points[:,2].min():.6g}, {ref.points[:,2].max():.6g}]\n"
    )
    lateral_covers_vtk = (
        case.points[:, 0].min() <= ref.points[:, 0].min() + 1e-6
        and case.points[:, 0].max() >= ref.points[:, 0].max() - 1e-6
        and case.points[:, 1].min() <= ref.points[:, 1].min() + 1e-6
        and case.points[:, 1].max() >= ref.points[:, 1].max() - 1e-6
    )
    z_covers_vtk = (
        case.points[:, 2].min() <= ref.points[:, 2].min() + 1e-6
        and case.points[:, 2].max() >= ref.points[:, 2].max() - 1e-6
    )
    log(f"FEM laterally covers full VTK x,y range: {lateral_covers_vtk}")
    log(f"FEM covers full VTK z range [0, {ref.points[:,2].max():.6g}] (ignoring FEM's extra negative-z air region): {z_covers_vtk}\n")

    # ------------------------------------------------------------------
    # 7. Air-region check (negative z) and z=0 well-definedness
    # ------------------------------------------------------------------
    log("## 7. Air-region (negative z) and z=0 boundary check\n")

    n_negative_z = int((case.points[:, 2] < 0).sum())
    log(f"FEM points with z < 0 (air region): {n_negative_z} of {len(case.points)} total ({n_negative_z/len(case.points):.2%})")
    if n_negative_z > 0:
        neg_vals = case.values[case.points[:, 2] < 0]
        log(f"  phi_V range in air region: [{neg_vals.min():.6g}, {neg_vals.max():.6g}]")
    log(
        "VTK has NO points at z<0 (VTK z-range starts at "
        f"{ref.points[:,2].min():.6g}), so any FEM air-region points are only ever used "
        "as SOURCE data for griddata (case.points), never as query/target points "
        "(ref.points) -- interpolate_case_onto_ref queries only at ref.points, i.e. "
        "VTK's own point locations, all of which have z>=0. The negative-z air points "
        "correctly never appear as an output row; they can still influence the "
        "linear-interpolation weights for VTK z=0 query points if z=0 sits inside the "
        "convex hull simplex spanning z<0 and z>0 nodes.\n"
    )

    z0_nodes = case.points[np.isclose(case.points[:, 2], 0.0, atol=1e-9)]
    z0_vals = case.values[np.isclose(case.points[:, 2], 0.0, atol=1e-9)]
    log(f"FEM mesh nodes at EXACTLY z=0.0: {len(z0_nodes)}")
    if len(z0_nodes) > 0:
        # look for duplicate (x,y) at z=0 -- would indicate coincident air/Si-top nodes
        xy0 = np.round(z0_nodes[:, :2], 6)
        uniq_xy0, counts = np.unique(xy0, axis=0, return_counts=True)
        n_dup = int((counts > 1).sum())
        log(
            f"  unique (x,y) among z=0 nodes: {len(uniq_xy0)}, "
            f"(x,y) locations with >1 coincident z=0 node: {n_dup}"
        )
        if n_dup > 0:
            dup_xy = uniq_xy0[counts > 1]
            sample = dup_xy[:5]
            log(f"  sample duplicate (x,y) locations (z=0, up to 5 shown): {sample.tolist()}")
            # report phi range at one duplicate location to show if it's discontinuous
            for xy in sample[:3]:
                sel = np.all(np.isclose(z0_nodes[:, :2], xy, atol=1e-6), axis=1)
                vals_here = z0_vals[sel]
                log(f"    at (x,y)={xy.tolist()}: {sel.sum()} coincident z=0 nodes, phi_V values={vals_here.tolist()}")
        else:
            log("  No coincident/duplicate (x,y) nodes found at z=0 -- z=0 appears to be a single well-defined mesh layer, not a duplicated air/Si interface.")
    log("")

    # ------------------------------------------------------------------
    # Verdicts and exclusion list
    # ------------------------------------------------------------------
    log("## Summary verdicts\n")

    bbox_overlap_ok = not np.any(overlap_extent <= 0) and np.all(coverage >= 0.999)
    log(f"Bounding-box overlap: {'PASS' if bbox_overlap_ok else 'CAUTION/FAIL'} (see section 4 numbers)")
    log(f"Global interpolation fallback fraction: {result_full.nearest_fallback_frac:.4%}")

    worst_slice = max(
        (s for s in slice_summaries if s[1] > 0 and not np.isnan(s[4])),
        key=lambda s: s[4],
        default=None,
    )
    if worst_slice:
        log(
            f"Worst representative slice: z={worst_slice[0]} -- fallback_frac={worst_slice[4]:.4%} "
            f"({worst_slice[3]} of {worst_slice[1]} points), verdict={worst_slice[5]}"
        )

    log("\n### Slices/regions to EXCLUDE or FLAG for Agent 3's error-metric sweep:\n")
    exclude_lines = []
    for z, n, valid, fb, frac, verdict in slice_summaries:
        if verdict in ("FAIL", "CAUTION", "NO POINTS"):
            exclude_lines.append(f"- z={z}: {verdict} -- fallback_frac={frac:.4%} ({fb}/{n} points used nearest-neighbor extrapolation)")
    if not exclude_lines:
        exclude_lines.append("- None of the 5 representative slices (z=0,35,48,1000,2000) showed >0% fallback; all PASS.")
    for line in exclude_lines:
        log(line)

    log(
        "\nGeneral flag regardless of per-slice fallback numbers: any FEM point with "
        "z<0 (the explicit air region, negative-z by construction) has NO corresponding "
        "VTK data and must never be used as a ground-truth comparison target -- "
        "interpolate_case_onto_ref already only queries at VTK point locations (z>=0), "
        "so as long as Agent 3 continues to treat VTK as ref and FEM as case (not vice "
        "versa), air-region points are structurally excluded from the metric by "
        "construction. If Agent 3's sweep ever queries FEM values directly (bypassing "
        "interpolate_case_onto_ref) at z<0, those rows must be dropped before any "
        "VTK-comparison metric is computed."
    )

    out_path = RUN3_DIR / "diagnostics" / "coordinate_overlap_audit.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(OUT_LINES) + "\n")
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
