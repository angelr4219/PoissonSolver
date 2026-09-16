#!/usr/bin/env python3
"""Corrected depth-sweep comparison metrics: run3 (FEM, field `phi_V`) vs Leah's
VTK reference (`/Users/angelramirez/Downloads/basePotential3d(1).vtk`, field
`basePotential`).

This is a NEW, separate module. It imports (does not reimplement) the core I/O
and interpolation primitives from `src/poisson/vtk_xdmf_compare.py`:
`read_vtk_reference`, `read_xdmf_case`, `interpolate_case_onto_ref`,
`nearest_z_mask`, `XdmfCase`. It does not modify that file.

Bug this module deliberately avoids
-----------------------------------
The task spec that produced this module called out a specific bug pattern to
avoid: naming a SIGNED difference `abs_err` without ever taking `np.abs()`.
We independently re-checked `compute_metrics` in `vtk_xdmf_compare.py` (as of
this writing): it computes `diff = case - ref` and then correctly
`abs_err = np.abs(diff)` -- that specific line is NOT buggy. However, this
module still keeps signed and absolute error in clearly, separately named
fields everywhere (`signed_err_*` vs `abs_err_*`) and never lets one silently
stand in for the other, per the task's explicit requirement, independent of
whatever the historical bug status of the sibling module is.

Why E_L2 / E_range are the PRIMARY metrics, not masked pointwise % error
-------------------------------------------------------------------------
`vtk_xdmf_compare.compute_metrics`'s own docstring notes that percent/relative
pointwise error is unreliable wherever the reference field is itself near
zero -- a small absolute error blows up into a huge percent error there, even
though it is physically negligible next to the ~4.4 V scale of this problem.
This module follows the same philosophy: `E_L2` (global relative L2 norm) and
`E_range` (RMSE normalized by the VTK's own GLOBAL peak-to-peak range, i.e.
computed once over the whole reference field, not per slice) are the metrics
that should be used to judge the <1% accuracy target. The masked pointwise
relative error (`masked_relerr_*` columns) is reported for completeness and
for finding *localized* problem spots, but is explicitly SECONDARY and must
never be used alone, or instead of E_L2/E_range, to judge whether the overall
device-region comparison is "good."

Masked pointwise relative error threshold
------------------------------------------
We use |phi_VTK| >= 0.044 V (1% of the VTK's own peak |phi| ~ 4.4 V) as the
threshold below which pointwise percent error is considered unreliable/noisy.
This matches the value already established elsewhere in this repo's
diagnostics for this exact VTK/run3 pair (run3's own solved phi range is
[-4.4, +1.0] V and the VTK's bottom Dirichlet plane is exactly -4.4 V, so 1%
of ~4.4 V = 0.044 V is the natural, already-precedented choice here). We did
not find data in this audit suggesting a better alternative, so we use the
established convention as-is rather than inventing a new one.

Depth grid
----------
z = 1..300 nm in 1 nm steps, then z = 300..2000 nm in 5 nm steps (z=300 is not
duplicated). z=0 is treated as a separate, specially-flagged diagnostic point
(`slice_num=0`, `is_special_z0=True`) -- it sits exactly on the VTK's top
surface and is not part of the main numbered sweep.

Outputs
-------
- notebooks/afm_vs_leah_compare/run3/metrics/depth_metrics_corrected.csv
- notebooks/afm_vs_leah_compare/run3/metrics/summary_metrics_corrected.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from poisson.vtk_xdmf_compare import (  # noqa: E402
    XdmfCase,
    interpolate_case_onto_ref,
    nearest_coord_mask,
    nearest_z_mask,
    read_vtk_reference,
    read_xdmf_case,
)

VTK_PATH = Path("/Users/angelramirez/Downloads/basePotential3d(1).vtk")
RUN3_DIR = REPO_ROOT / "notebooks" / "afm_vs_leah_compare" / "run3"
XDMF_PATH = RUN3_DIR / "sige_afm_tip.xdmf"
OUT_DIR = RUN3_DIR / "metrics"

MASK_THRESHOLD_V = 0.044  # 1% of VTK peak |phi| (~4.4 V) -- repo convention
DEVICE_REGION_Z = (0.0, 53.0)  # nm, per run3's material stack (see run_results.json)
TRIVIAL_NEAR_BOUNDARY_Z = 1500.0  # nm; used only to caveat the "overall best slice"

EPS = 1e-12


# ---------------------------------------------------------------------------
# Depth grid
# ---------------------------------------------------------------------------
def build_z_grid() -> list[float]:
    fine = list(range(1, 301))  # 1..300 inclusive, 1 nm steps
    coarse = list(range(305, 2001, 5))  # 305..2000, 5 nm steps (300 not repeated)
    return [float(z) for z in fine + coarse]


def nearest_unique_z(all_z: np.ndarray, z_target: float) -> tuple[float, float]:
    """Find the VTK's actual nearest z-plane to z_target and a tolerance tight
    enough to select ONLY that plane (half the local gap to its neighbors).

    We deliberately do not rely on nearest_z_mask's own default (auto) tol,
    because that default is computed from the median spacing across the ENTIRE
    z grid. This VTK has wildly non-uniform z spacing (0.5 nm in [25,58] nm,
    ~20 nm elsewhere), so a single global tolerance would either merge many
    fine-region planes into one query or fail to find real matches. Instead we
    compute a local, per-target tolerance here and pass it explicitly into
    nearest_z_mask (reusing that function for the actual masking).
    """
    uz = np.unique(all_z)
    idx = int(np.argmin(np.abs(uz - z_target)))
    actual = float(uz[idx])
    left_gap = actual - uz[idx - 1] if idx > 0 else np.inf
    right_gap = uz[idx + 1] - actual if idx < len(uz) - 1 else np.inf
    gap = min(left_gap, right_gap)
    tol = gap / 2.0 if np.isfinite(gap) else 1e-6
    return actual, max(tol, 1e-9)


# ---------------------------------------------------------------------------
# Shape-comparison helpers
# ---------------------------------------------------------------------------
def _fwhm_along_line(coord: np.ndarray, centered_vals: np.ndarray) -> float:
    """Full-width-at-half-max of a single dominant extremum along a 1D profile.

    Returns NaN if there is no clear single interior extremum (extremum at an
    edge, no half-max crossing found on one side, or the profile is
    essentially flat).
    """
    order = np.argsort(coord)
    c = coord[order]
    v = centered_vals[order]
    if len(v) < 3:
        return float("nan")
    idx = int(np.argmax(np.abs(v)))
    peak_val = v[idx]
    if idx == 0 or idx == len(v) - 1:
        return float("nan")
    if abs(peak_val) < 1e-6:
        return float("nan")
    half = peak_val / 2.0

    left = float("nan")
    for i in range(idx, 0, -1):
        if (v[i] - half) * (v[i - 1] - half) <= 0 and v[i] != v[i - 1]:
            t = (half - v[i]) / (v[i - 1] - v[i])
            left = c[i] + t * (c[i - 1] - c[i])
            break
    right = float("nan")
    for i in range(idx, len(v) - 1):
        if (v[i] - half) * (v[i + 1] - half) <= 0 and v[i] != v[i + 1]:
            t = (half - v[i]) / (v[i + 1] - v[i])
            right = c[i] + t * (c[i + 1] - c[i])
            break
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return abs(right - left)


def _extremum_xy(x: np.ndarray, y: np.ndarray, vals: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(np.abs(vals)))
    return float(x[idx]), float(y[idx])


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = pearsonr(a, b)
    return float(r)


# ---------------------------------------------------------------------------
# Per-slice metrics
# ---------------------------------------------------------------------------
def compute_slice_row(
    slice_num: int,
    is_special_z0: bool,
    z_requested: float,
    points: np.ndarray,
    vtk_v: np.ndarray,
    fem_v: np.ndarray,
    fallback_mask: np.ndarray,
    actual_z: float,
    global_vtk_min: float,
    global_vtk_max: float,
) -> dict:
    n_points = len(vtk_v)
    n_fallback = int(fallback_mask.sum())
    n_overlap = n_points - n_fallback
    valid_overlap_frac = n_overlap / n_points if n_points else float("nan")
    fallback_frac = n_fallback / n_points if n_points else float("nan")

    signed_err = fem_v - vtk_v  # e = phi_FEM - phi_VTK
    abs_err = np.abs(signed_err)

    vtk_norm = np.linalg.norm(vtk_v)
    E_L2 = float(np.linalg.norm(signed_err) / vtk_norm) if vtk_norm > EPS else float("nan")

    rmse = float(np.sqrt(np.mean(signed_err**2)))
    global_range = global_vtk_max - global_vtk_min
    E_range = rmse / global_range if global_range > EPS else float("nan")

    # Masked pointwise relative error (SECONDARY metric -- see module docstring)
    sig_mask = np.abs(vtk_v) >= MASK_THRESHOLD_V
    n_masked_incl = int(sig_mask.sum())
    n_masked_excl = int((~sig_mask).sum())
    if n_masked_incl:
        masked_relerr = abs_err[sig_mask] / np.abs(vtk_v[sig_mask])
        masked_relerr_min = float(np.min(masked_relerr))
        masked_relerr_mean = float(np.mean(masked_relerr))
        masked_relerr_max = float(np.max(masked_relerr))
    else:
        masked_relerr_min = masked_relerr_mean = masked_relerr_max = float("nan")

    # Shape-only comparison (removes constant offset / voltage bias)
    vtk_shape = vtk_v - vtk_v.mean()
    fem_shape = fem_v - fem_v.mean()
    shape_diff = fem_shape - vtk_shape
    shape_rmse = float(np.sqrt(np.mean(shape_diff**2)))
    vtk_shape_norm = np.linalg.norm(vtk_shape)
    shape_rel_l2 = (
        float(np.linalg.norm(shape_diff) / vtk_shape_norm) if vtk_shape_norm > EPS else float("nan")
    )
    pearson_r = _safe_pearsonr(fem_v, vtk_v)

    x, y = points[:, 0], points[:, 1]

    # FWHM along the y=0 line through the origin (if that line is present in slice)
    line_mask = nearest_coord_mask(points, axis="y", target=0.0)
    if line_mask.sum() >= 3:
        fwhm_vtk = _fwhm_along_line(x[line_mask], vtk_shape[line_mask])
        fwhm_fem = _fwhm_along_line(x[line_mask], fem_shape[line_mask])
    else:
        fwhm_vtk = fwhm_fem = float("nan")
    fwhm_diff = (
        fwhm_fem - fwhm_vtk if not (np.isnan(fwhm_vtk) or np.isnan(fwhm_fem)) else float("nan")
    )

    peak_x_vtk, peak_y_vtk = _extremum_xy(x, y, vtk_shape)
    peak_x_fem, peak_y_fem = _extremum_xy(x, y, fem_shape)
    peak_dx = peak_x_fem - peak_x_vtk
    peak_dy = peak_y_fem - peak_y_vtk
    peak_dist = float(np.hypot(peak_dx, peak_dy))

    pp_amp_vtk = float(vtk_shape.max() - vtk_shape.min())
    pp_amp_fem = float(fem_shape.max() - fem_shape.min())
    pp_amp_diff = pp_amp_fem - pp_amp_vtk

    return {
        "slice_num": slice_num,
        "is_special_z0": is_special_z0,
        "z_requested_nm": z_requested,
        "z_actual_nm": actual_z,
        "n_vtk_points": n_points,
        "n_overlap": n_overlap,
        "valid_overlap_frac": valid_overlap_frac,
        "nearest_fallback_frac": fallback_frac,
        "n_excluded": n_fallback,
        "vtk_min": float(vtk_v.min()),
        "vtk_mean": float(vtk_v.mean()),
        "vtk_max": float(vtk_v.max()),
        "vtk_range": float(vtk_v.max() - vtk_v.min()),
        "vtk_std": float(vtk_v.std()),
        "fem_min": float(fem_v.min()),
        "fem_mean": float(fem_v.mean()),
        "fem_max": float(fem_v.max()),
        "fem_range": float(fem_v.max() - fem_v.min()),
        "fem_std": float(fem_v.std()),
        "signed_err_min": float(signed_err.min()),
        "signed_err_mean": float(signed_err.mean()),  # bias
        "signed_err_max": float(signed_err.max()),
        "abs_err_min": float(abs_err.min()),
        "abs_err_mean": float(abs_err.mean()),
        "abs_err_max": float(abs_err.max()),
        "rmse": rmse,
        "E_L2": E_L2,
        "E_range": E_range,
        "masked_relerr_min": masked_relerr_min,
        "masked_relerr_mean": masked_relerr_mean,
        "masked_relerr_max": masked_relerr_max,
        "n_masked_included": n_masked_incl,
        "n_masked_excluded": n_masked_excl,
        "shape_rmse": shape_rmse,
        "shape_rel_l2": shape_rel_l2,
        "pearson_r": pearson_r,
        "fwhm_vtk_nm": fwhm_vtk,
        "fwhm_fem_nm": fwhm_fem,
        "fwhm_diff_nm": fwhm_diff,
        "peak_x_vtk": peak_x_vtk,
        "peak_y_vtk": peak_y_vtk,
        "peak_x_fem": peak_x_fem,
        "peak_y_fem": peak_y_fem,
        "peak_loc_dx": peak_dx,
        "peak_loc_dy": peak_dy,
        "peak_loc_dist": peak_dist,
        "pp_amp_vtk": pp_amp_vtk,
        "pp_amp_fem": pp_amp_fem,
        "pp_amp_diff": pp_amp_diff,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(vtk_path: Path = VTK_PATH, xdmf_path: Path = XDMF_PATH, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = read_vtk_reference(vtk_path, field="basePotential")
    case = read_xdmf_case(XdmfCase(label="run3", xdmf_path=xdmf_path, field="phi_V"))
    result = interpolate_case_onto_ref(ref, case, "run3")

    all_z = result.points[:, 2]
    global_vtk_min = float(ref.values.min())
    global_vtk_max = float(ref.values.max())  # computed ONCE over the whole reference field

    z_targets = build_z_grid()
    requests = [(0, True, 0.0)] + [(i + 1, False, z) for i, z in enumerate(z_targets)]

    rows = []
    for slice_num, is_special, z_req in requests:
        actual_z, tol = nearest_unique_z(all_z, z_req)
        mask = nearest_z_mask(result.points, actual_z, tol=tol)
        if mask.sum() == 0:
            warnings.warn(f"slice_num={slice_num} z_requested={z_req}: no points found, skipping")
            continue
        row = compute_slice_row(
            slice_num=slice_num,
            is_special_z0=is_special,
            z_requested=z_req,
            points=result.points[mask],
            vtk_v=result.ref_values[mask],
            fem_v=result.case_values[mask],
            fallback_mask=result.fallback_mask[mask],
            actual_z=actual_z,
            global_vtk_min=global_vtk_min,
            global_vtk_max=global_vtk_max,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "depth_metrics_corrected.csv"
    df.to_csv(csv_path, index=False)

    # ---- Global (whole-domain) metrics, computed once over ALL VTK points ----
    signed_err_full = result.case_values - result.ref_values
    rmse_full = float(np.sqrt(np.mean(signed_err_full**2)))
    vtk_norm_full = np.linalg.norm(result.ref_values)
    E_L2_full = float(np.linalg.norm(signed_err_full) / vtk_norm_full)
    E_range_full = rmse_full / (global_vtk_max - global_vtk_min)
    bias_full = float(signed_err_full.mean())

    device_mask = (all_z >= DEVICE_REGION_Z[0]) & (all_z <= DEVICE_REGION_Z[1])
    bias_device = (
        float(signed_err_full[device_mask].mean()) if device_mask.any() else float("nan")
    )

    # ---- Device-region (z in [0,53] nm) best/worst by E_range ----
    dev_df = df[(df["z_actual_nm"] >= DEVICE_REGION_Z[0]) & (df["z_actual_nm"] <= DEVICE_REGION_Z[1])]
    dev_df_valid = dev_df.dropna(subset=["E_range"])
    best_dev = dev_df_valid.loc[dev_df_valid["E_range"].idxmin()].to_dict() if len(dev_df_valid) else {}
    worst_dev = dev_df_valid.loc[dev_df_valid["E_range"].idxmax()].to_dict() if len(dev_df_valid) else {}

    # ---- Overall (full sweep, all z) best/worst by E_range ----
    df_valid = df.dropna(subset=["E_range"])
    best_overall = df_valid.loc[df_valid["E_range"].idxmin()].to_dict() if len(df_valid) else {}
    worst_overall = df_valid.loc[df_valid["E_range"].idxmax()].to_dict() if len(df_valid) else {}

    best_overall_is_trivial = bool(
        best_overall and best_overall.get("z_actual_nm", 0) >= TRIVIAL_NEAR_BOUNDARY_Z
    )

    summary = {
        "vtk_path": str(vtk_path),
        "xdmf_path": str(xdmf_path),
        "vtk_field": "basePotential",
        "fem_field": "phi_V",
        "n_vtk_points_total": int(len(ref.values)),
        "n_slices_computed": int(len(df)),
        "global_vtk_min_V": global_vtk_min,
        "global_vtk_max_V": global_vtk_max,
        "global_full_domain": {
            "E_L2": E_L2_full,
            "E_range": E_range_full,
            "rmse_V": rmse_full,
            "bias_mean_signed_err_V": bias_full,
            "nearest_fallback_frac": result.nearest_fallback_frac,
        },
        "device_region_z_nm": list(DEVICE_REGION_Z),
        "bias_mean_signed_err_device_region_V": bias_device,
        "device_region_best_slice_by_E_range": best_dev,
        "device_region_worst_slice_by_E_range": worst_dev,
        "overall_best_slice_by_E_range": best_overall,
        "overall_worst_slice_by_E_range": worst_overall,
        "note": (
            "Best/worst slices above are selected by E_range (RMSE normalized by the "
            "VTK's own GLOBAL peak-to-peak range), NOT by the masked pointwise relative "
            "error, which this module's docstring documents as secondary/unreliable near "
            "phi_VTK=0. IMPORTANT CAVEAT: the 'overall best slice' can trivially be a deep "
            "slice near the shared -4.4V bottom Dirichlet boundary (z close to 2053 nm), "
            "where both FEM and VTK are pinned to the same fixed voltage by construction "
            "and therefore agree almost perfectly -- this is NOT evidence of a physically "
            "meaningful match and must not be reported as 'the best result' without this "
            "caveat. The device-region (z=0-53 nm) best/worst slices above are the "
            "physically meaningful ones for judging the AFM-tip/SiGe stack comparison."
            + (
                " NOTE: the computed overall-best slice IS near that trivial boundary region."
                if best_overall_is_trivial
                else ""
            )
        ),
    }

    json_path = out_dir / "summary_metrics_corrected.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {csv_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Wrote {json_path}")
    print(f"Global E_L2={E_L2_full:.6e}  E_range={E_range_full:.6e}")
    if best_dev:
        print(
            f"Device-region best slice: z_actual={best_dev['z_actual_nm']:.3g}nm "
            f"E_range={best_dev['E_range']:.6e}"
        )
    if worst_dev:
        print(
            f"Device-region worst slice: z_actual={worst_dev['z_actual_nm']:.3g}nm "
            f"E_range={worst_dev['E_range']:.6e}"
        )

    return df, summary, result


if __name__ == "__main__":
    run()
