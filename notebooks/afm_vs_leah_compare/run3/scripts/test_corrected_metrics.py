#!/usr/bin/env python3
"""Synthetic validation suite for corrected_comparison_metrics.py.

This module builds hand-designed synthetic FieldData-equivalent test cases
(plain numpy point/value arrays), computes analytically-expected results
BEFORE calling any of the module-under-test's functions, then compares.

It imports the public functions of corrected_comparison_metrics.py
(`compute_slice_row`, `_fwhm_along_line`, `_extremum_xy`, `_safe_pearsonr`)
and the shared primitives from src/poisson/vtk_xdmf_compare.py
(`FieldData`, `interpolate_case_onto_ref`) directly -- it does NOT modify
either file.

Run as a script: python3 test_corrected_metrics.py
Every test prints EXPECTED vs MEASURED and a PASS/FAIL verdict; a final
summary line reports the overall tally. Exits nonzero if any test fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from poisson.vtk_xdmf_compare import (  # noqa: E402
    FieldData,
    XdmfCase,
    interpolate_case_onto_ref,
    nearest_z_mask,
    read_vtk_reference,
    read_xdmf_case,
)

import corrected_comparison_metrics as ccm  # noqa: E402

RESULTS = []  # list of (name, passed: bool, detail: str)


def record(name: str, passed: bool, detail: str):
    RESULTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}\n    {detail}\n")


def close(a, b, atol=0, rtol=1e-9):
    if (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return np.isnan(a) and np.isnan(b)
    return np.isclose(a, b, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Shared synthetic grid helpers
# ---------------------------------------------------------------------------
def make_grid(nx=21, ny=21, lo=-5.0, hi=5.0, z=0.0):
    xs = np.linspace(lo, hi, nx)
    ys = np.linspace(lo, hi, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.full_like(X, z)
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return points, X.ravel(), Y.ravel()


def gaussian(x, y, x0=0.0, y0=0.0, amp=1.0, sigma=1.5):
    return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


# ===========================================================================
# TEST 1: identical fields (with real spatial variation)
# ===========================================================================
def test1_identical_fields():
    points, x, y = make_grid()
    vtk_v = np.sin(x) * np.cos(y)  # real spatial variation, mean != const
    fem_v = vtk_v.copy()  # exact copy

    gmin, gmax = float(vtk_v.min()), float(vtk_v.max())
    row = ccm.compute_slice_row(
        slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=vtk_v, fem_v=fem_v, fallback_mask=np.zeros(len(vtk_v), dtype=bool),
        actual_z=0.0, global_vtk_min=gmin, global_vtk_max=gmax,
    )
    ok = (
        close(row["signed_err_mean"], 0.0, atol=1e-12)
        and close(row["rmse"], 0.0, atol=1e-12)
        and close(row["E_L2"], 0.0, atol=1e-12)
        and close(row["E_range"], 0.0, atol=1e-12)
        and close(row["masked_relerr_mean"], 0.0, atol=1e-12)
        and close(row["shape_rmse"], 0.0, atol=1e-12)
        and close(row["pearson_r"], 1.0, atol=1e-9)
    )
    record(
        "1. Identical fields (sin(x)cos(y), real spatial variation)",
        ok,
        f"expected: signed_err_mean=0, rmse=0, E_L2=0, E_range=0, "
        f"masked_relerr_mean=0, shape_rmse=0, pearson_r=1.0 | "
        f"measured: signed_err_mean={row['signed_err_mean']:.3e}, rmse={row['rmse']:.3e}, "
        f"E_L2={row['E_L2']:.3e}, E_range={row['E_range']:.3e}, "
        f"masked_relerr_mean={row['masked_relerr_mean']:.3e}, shape_rmse={row['shape_rmse']:.3e}, "
        f"pearson_r={row['pearson_r']:.6f}",
    )

    # 1b: degenerate case -- perfectly CONSTANT field -> correlation must be NaN
    # (documented, expected degenerate behavior per task spec item 1, not a bug)
    const_v = np.full(len(points), 3.14159)
    row_const = ccm.compute_slice_row(
        slice_num=2, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=const_v, fem_v=const_v.copy(), fallback_mask=np.zeros(len(points), dtype=bool),
        actual_z=0.0, global_vtk_min=float(const_v.min()) - 1, global_vtk_max=float(const_v.max()) + 1,
    )
    ok_const = np.isnan(row_const["pearson_r"]) and close(row_const["shape_rmse"], 0.0, atol=1e-12)
    record(
        "1b. Degenerate constant-field correlation (expected NaN, not a bug)",
        ok_const,
        f"expected: pearson_r=NaN (zero variance, _safe_pearsonr guards this), shape_rmse=0 | "
        f"measured: pearson_r={row_const['pearson_r']}, shape_rmse={row_const['shape_rmse']:.3e}",
    )


# ===========================================================================
# TEST 2: known constant voltage offset
# ===========================================================================
def test2_constant_offset():
    points, x, y = make_grid()
    C = 0.5
    vtk_v = 3.0 + np.sin(x) * np.cos(y)  # spatially varying, nonzero mean
    fem_v = vtk_v + C  # pure constant offset

    gmin, gmax = float(vtk_v.min()), float(vtk_v.max())
    global_range = gmax - gmin
    expected_rmse = abs(C)  # diff is EXACTLY C everywhere -> sqrt(mean(C^2)) = |C|
    expected_E_range = expected_rmse / global_range

    row = ccm.compute_slice_row(
        slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=vtk_v, fem_v=fem_v, fallback_mask=np.zeros(len(vtk_v), dtype=bool),
        actual_z=0.0, global_vtk_min=gmin, global_vtk_max=gmax,
    )
    ok = (
        close(row["signed_err_mean"], C, atol=1e-12)
        and close(row["rmse"], expected_rmse, atol=1e-12)
        and close(row["E_range"], expected_E_range, rtol=1e-9)
        and close(row["shape_rmse"], 0.0, atol=1e-12)
        and close(row["pearson_r"], 1.0, atol=1e-9)
    )
    record(
        "2. Constant voltage offset (C=0.5V, spatially-varying ref)",
        ok,
        f"expected: signed_err_mean=C={C}, rmse=|C|={expected_rmse}, "
        f"E_range=rmse/global_range={expected_E_range:.6f} (global_range={global_range:.6f}), "
        f"shape_rmse=0 (offset must NOT leak into shape metric), pearson_r=1.0 | "
        f"measured: signed_err_mean={row['signed_err_mean']:.6f}, rmse={row['rmse']:.6f}, "
        f"E_range={row['E_range']:.6f}, shape_rmse={row['shape_rmse']:.3e}, "
        f"pearson_r={row['pearson_r']:.9f}",
    )


# ===========================================================================
# TEST 3: known multiplicative scaling
# ===========================================================================
def test3_multiplicative_scaling():
    points, x, y = make_grid()
    vtk_v = 3.0 + np.sin(x) * np.cos(y)  # nonzero mean, real spatial variation
    vtk_norm = np.linalg.norm(vtk_v)
    vtk_rms = np.sqrt(np.mean(vtk_v**2))

    for k, expected_pearson in [(2.0, 1.0), (-1.0, -1.0)]:
        fem_v = k * vtk_v
        diff = fem_v - vtk_v  # = (k-1)*vtk_v
        expected_E_L2 = abs(k - 1)  # ||(k-1)v|| / ||v|| = |k-1| exactly
        expected_rmse = abs(k - 1) * vtk_rms

        gmin, gmax = float(vtk_v.min()), float(vtk_v.max())
        row = ccm.compute_slice_row(
            slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
            vtk_v=vtk_v, fem_v=fem_v, fallback_mask=np.zeros(len(vtk_v), dtype=bool),
            actual_z=0.0, global_vtk_min=gmin, global_vtk_max=gmax,
        )
        ok = (
            close(row["E_L2"], expected_E_L2, rtol=1e-9)
            and close(row["rmse"], expected_rmse, rtol=1e-9)
            and close(row["pearson_r"], expected_pearson, atol=1e-9)
        )
        record(
            f"3. Multiplicative scaling k={k}",
            ok,
            f"expected: E_L2=|k-1|={expected_E_L2}, rmse=|k-1|*rms(ref)={expected_rmse:.6f}, "
            f"pearson_r={expected_pearson} | "
            f"measured: E_L2={row['E_L2']:.9f}, rmse={row['rmse']:.6f}, pearson_r={row['pearson_r']:.9f}",
        )


# ===========================================================================
# TEST 4: shifted field (known lateral shift)
# ===========================================================================
def test4_shifted_field():
    points, x, y = make_grid(nx=41, ny=41, lo=-5.0, hi=5.0)  # 0.25 spacing, includes exact grid pts
    sigma = 1.0
    dx, dy = 1.5, 1.0  # chosen to land exactly on the 0.25-step grid
    vtk_v = gaussian(x, y, x0=0.0, y0=0.0, amp=2.0, sigma=sigma)
    fem_v = gaussian(x, y, x0=dx, y0=dy, amp=2.0, sigma=sigma)  # same shape, shifted origin

    # Independent expectation: peak of (mean-subtracted) vtk field is at (0,0);
    # peak of fem field is at (dx,dy). Since amp>0 and mean-subtraction only
    # subtracts a spatially-uniform constant, argmax location is unaffected.
    expected_peak_dx, expected_peak_dy = dx, dy
    expected_peak_dist = float(np.hypot(dx, dy))

    gmin, gmax = float(vtk_v.min()), float(vtk_v.max())
    row = ccm.compute_slice_row(
        slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=vtk_v, fem_v=fem_v, fallback_mask=np.zeros(len(vtk_v), dtype=bool),
        actual_z=0.0, global_vtk_min=gmin, global_vtk_max=gmax,
    )

    # Independent shape-RMSE re-derivation (not reusing ccm's internals)
    vtk_shape_ind = vtk_v - vtk_v.mean()
    fem_shape_ind = fem_v - fem_v.mean()
    expected_shape_rmse = float(np.sqrt(np.mean((fem_shape_ind - vtk_shape_ind) ** 2)))

    ok = (
        close(row["peak_loc_dx"], expected_peak_dx, atol=1e-9)
        and close(row["peak_loc_dy"], expected_peak_dy, atol=1e-9)
        and close(row["peak_loc_dist"], expected_peak_dist, atol=1e-9)
        and close(row["shape_rmse"], expected_shape_rmse, rtol=1e-9)
        and row["shape_rmse"] > 0.05  # real, non-trivial shape degradation
        and row["pearson_r"] < 0.99  # correlation should degrade, not stay ~1
    )
    record(
        "4. Shifted field (dx=1.5, dy=1.0 lateral shift)",
        ok,
        f"expected: peak_loc_dx={expected_peak_dx}, peak_loc_dy={expected_peak_dy}, "
        f"peak_loc_dist={expected_peak_dist:.6f}, shape_rmse(independent)={expected_shape_rmse:.6f} "
        f"(nontrivial >0), pearson_r < 0.99 (degraded) | "
        f"measured: peak_loc_dx={row['peak_loc_dx']:.6f}, peak_loc_dy={row['peak_loc_dy']:.6f}, "
        f"peak_loc_dist={row['peak_loc_dist']:.6f}, shape_rmse={row['shape_rmse']:.6f}, "
        f"pearson_r={row['pearson_r']:.6f}",
    )


# ===========================================================================
# TEST 5: reflected field (symmetric case + asymmetric case)
# ===========================================================================
def test5_reflected_field():
    points, x, y = make_grid(nx=41, ny=41, lo=-5.0, hi=5.0)
    sigma = 1.0

    # 5a: radially symmetric bump at origin -- reflection x -> -x is a NO-OP
    vtk_v = gaussian(x, y, x0=0.0, y0=0.0, amp=2.0, sigma=sigma)
    fem_v = gaussian(-x, y, x0=0.0, y0=0.0, amp=2.0, sigma=sigma)  # case(x,y)=ref(-x,y)
    # since ref(-x,y) == ref(x,y) for a symmetric bump centered at 0, fem_v should equal vtk_v exactly
    max_diff = float(np.max(np.abs(fem_v - vtk_v)))

    gmin, gmax = float(vtk_v.min()), float(vtk_v.max())
    row_sym = ccm.compute_slice_row(
        slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=vtk_v, fem_v=fem_v, fallback_mask=np.zeros(len(vtk_v), dtype=bool),
        actual_z=0.0, global_vtk_min=gmin, global_vtk_max=gmax,
    )
    ok_sym = (
        close(max_diff, 0.0, atol=1e-12)
        and close(row_sym["shape_rmse"], 0.0, atol=1e-10)
        and close(row_sym["pearson_r"], 1.0, atol=1e-9)
    )
    record(
        "5a. Reflection of a symmetric field (radial bump at origin) -> should be a no-op",
        ok_sym,
        f"expected: fields identical (max|diff|=0 by construction since ref(-x,y)=ref(x,y)), "
        f"shape_rmse=0, pearson_r=1.0 | "
        f"measured: max|diff|={max_diff:.3e}, shape_rmse={row_sym['shape_rmse']:.3e}, "
        f"pearson_r={row_sym['pearson_r']:.9f}",
    )

    # 5b: bump offset at x0=+1 (asymmetric under x -> -x). case(x,y)=ref(-x,y)
    # => reflected bump peak is at x=-1 instead of x=+1: a shift of -2 in x.
    x0 = 1.0
    vtk_v2 = gaussian(x, y, x0=x0, y0=0.0, amp=2.0, sigma=sigma)
    fem_v2 = gaussian(-x, y, x0=x0, y0=0.0, amp=2.0, sigma=sigma)  # = gaussian(x, y, x0=-x0, ...)
    expected_peak_dx = -2 * x0  # from +1 to -1
    expected_peak_dist = abs(expected_peak_dx)

    gmin2, gmax2 = float(vtk_v2.min()), float(vtk_v2.max())
    row_asym = ccm.compute_slice_row(
        slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=vtk_v2, fem_v=fem_v2, fallback_mask=np.zeros(len(vtk_v2), dtype=bool),
        actual_z=0.0, global_vtk_min=gmin2, global_vtk_max=gmax2,
    )
    ok_asym = (
        close(row_asym["peak_loc_dx"], expected_peak_dx, atol=1e-9)
        and close(row_asym["peak_loc_dist"], expected_peak_dist, atol=1e-9)
        and row_asym["shape_rmse"] > 0.05  # real mismatch detected
        and row_asym["pearson_r"] < 0.9
    )
    record(
        "5b. Reflection of an asymmetric field (bump offset at x0=+1) -> real mismatch detected",
        ok_asym,
        f"expected: peak_loc_dx=-2*x0={expected_peak_dx}, peak_loc_dist={expected_peak_dist}, "
        f"shape_rmse>0 (nontrivial), pearson_r<0.9 (mismatch) | "
        f"measured: peak_loc_dx={row_asym['peak_loc_dx']:.6f}, peak_loc_dist={row_asym['peak_loc_dist']:.6f}, "
        f"shape_rmse={row_asym['shape_rmse']:.6f}, pearson_r={row_asym['pearson_r']:.6f}",
    )


# ===========================================================================
# TEST 6: partial domain overlap (known fallback fraction)
# ===========================================================================
def test6_partial_overlap():
    # ref: 10x10 grid in x,y across 3 distinct z layers (needs real 3D extent
    # for scipy's Delaunay triangulation inside griddata to be non-degenerate).
    nx = ny = 10
    xs = np.linspace(-5.0, 5.0, nx)
    ys = np.linspace(-5.0, 5.0, ny)
    zs = np.array([0.0, 1.0, 2.0])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    ref_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    ref_values = np.sin(ref_points[:, 0]) + np.cos(ref_points[:, 1])
    ref = FieldData(name="ref", points=ref_points, values=ref_values)

    # Independent ground truth: how many ref points have x > 1.0 (the 4 largest
    # of the 10 unique x-grid-values: 1.666..., 2.777..., 3.888..., 5.0)
    expected_fallback_mask_ind = ref_points[:, 0] > 1.0
    expected_n_fallback = int(expected_fallback_mask_ind.sum())
    expected_frac = expected_n_fallback / len(ref_points)

    # case: same functional field, but only covering x <= 1.0 (dense enough
    # grid so the covered sub-hull is exactly x in [-5, 1])
    case_nx = 30
    case_xs = np.linspace(-5.0, 1.0, case_nx)
    CX, CY, CZ = np.meshgrid(case_xs, ys, zs, indexing="ij")
    case_points = np.stack([CX.ravel(), CY.ravel(), CZ.ravel()], axis=1)
    case_values = np.sin(case_points[:, 0]) + np.cos(case_points[:, 1])
    case = FieldData(name="case", points=case_points, values=case_values)

    result = interpolate_case_onto_ref(ref, case, "partial_overlap_test", check_bbox=False)

    ok = (
        result.fallback_mask.sum() == expected_n_fallback
        and close(result.nearest_fallback_frac, expected_frac, atol=1e-12)
        and np.array_equal(result.fallback_mask, expected_fallback_mask_ind)
    )
    record(
        "6. Partial domain overlap (case covers only x<=1.0, ref extends to x=5.0)",
        ok,
        f"expected: n_fallback={expected_n_fallback} (points with x>1.0, hand-counted from "
        f"the 10 unique x-grid values), fallback_frac={expected_frac:.4f} | "
        f"measured: n_fallback={int(result.fallback_mask.sum())}, "
        f"fallback_frac={result.nearest_fallback_frac:.4f}, mask arrays identical="
        f"{np.array_equal(result.fallback_mask, expected_fallback_mask_ind)}",
    )
    return result


# ===========================================================================
# TEST 7: reference crossing zero (unmasked blow-up vs masked exclusion)
# ===========================================================================
def test7_zero_crossing():
    points, x, y = make_grid(nx=41, ny=41, lo=-5.0, hi=5.0)
    vtk_v = x.copy()  # linear ramp: negative for x<0, positive for x>0, =0 at x=0
    eps = 0.01
    fem_v = vtk_v + eps  # small constant absolute error everywhere

    threshold = ccm.MASK_THRESHOLD_V  # 0.044 V, module's own convention
    # Independent unmasked pct-error blow-up check (module itself does NOT
    # expose an unmasked pct-error column -- only masked_relerr_*). The main
    # grid's nearest point to the crossing is 0.25 away (eps/0.25=4%, not a
    # dramatic blow-up), so demonstrate the true asymptotic behavior with a
    # separate sequence of points approaching the crossing arbitrarily closely
    # (this is the actual mathematical claim being tested: pct_err = eps/|ref|
    # -> infinity as |ref| -> 0, which is why the masking scheme exists at all).
    x_demo = np.array([1.0, 1e-1, 1e-2, 1e-4, 1e-6])
    pct_err_demo = 100.0 * eps / np.abs(x_demo)  # eps constant, |ref|=|x_demo|
    huge_pct_err = float(pct_err_demo[-1])  # at x=1e-6: 100*0.01/1e-6 = 1e8 %
    is_monotonically_blowing_up = bool(np.all(np.diff(pct_err_demo) > 0))

    # Independent hand-count of points below mask threshold
    expected_excluded_mask = np.abs(vtk_v) < threshold
    expected_n_excluded = int(expected_excluded_mask.sum())
    expected_n_included = len(vtk_v) - expected_n_excluded

    gmin, gmax = float(vtk_v.min()), float(vtk_v.max())
    row = ccm.compute_slice_row(
        slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=vtk_v, fem_v=fem_v, fallback_mask=np.zeros(len(vtk_v), dtype=bool),
        actual_z=0.0, global_vtk_min=gmin, global_vtk_max=gmax,
    )

    # masked relative error should be bounded and sane: since fem=vtk+eps
    # everywhere, for included points relerr = eps/|vtk| <= eps/threshold
    expected_masked_max_bound = eps / threshold

    ok = (
        row["n_masked_excluded"] == expected_n_excluded
        and row["n_masked_included"] == expected_n_included
        and row["masked_relerr_max"] <= expected_masked_max_bound + 1e-9
        and np.isfinite(row["masked_relerr_max"])
        and huge_pct_err >= 1e6  # confirms unmasked pct err is genuinely unbounded near |ref|=0
        and is_monotonically_blowing_up
    )
    record(
        "7. Reference crossing zero (linear ramp, |vtk| threshold=0.044V)",
        ok,
        f"expected: n_masked_excluded={expected_n_excluded} (hand-count of |vtk|<{threshold}), "
        f"n_masked_included={expected_n_included}, masked_relerr_max <= eps/threshold="
        f"{expected_masked_max_bound:.4f} (bounded); independent unmasked pct_err=100*eps/|ref| "
        f"at |ref|=1,0.1,0.01,1e-4,1e-6 -> {np.array2string(pct_err_demo, precision=3)}% "
        f"(monotonically diverging, confirming the known blow-up failure mode the module's "
        f"masking scheme is designed to route around; ccm's compute_slice_row itself does "
        f"NOT expose an unmasked pct-error column, only masked_relerr_*) | "
        f"measured: n_masked_excluded={row['n_masked_excluded']}, "
        f"n_masked_included={row['n_masked_included']}, masked_relerr_max={row['masked_relerr_max']:.6f}",
    )


# ===========================================================================
# TEST 8: known lateral-width (FWHM) difference
# ===========================================================================
def test8_fwhm_width_difference():
    points, x, y = make_grid(nx=81, ny=5, lo=-8.0, hi=8.0)  # fine x-resolution along y=0
    sigma_vtk = 1.0
    sigma_fem = 2.0
    amp = 3.0
    vtk_v = gaussian(x, y, x0=0.0, y0=0.0, amp=amp, sigma=sigma_vtk)
    fem_v = gaussian(x, y, x0=0.0, y0=0.0, amp=amp, sigma=sigma_fem)

    # Analytic FWHM of a Gaussian: 2*sqrt(2*ln2)*sigma ~= 2.35482*sigma
    k_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    expected_fwhm_vtk = k_fwhm * sigma_vtk
    expected_fwhm_fem = k_fwhm * sigma_fem
    expected_fwhm_diff = expected_fwhm_fem - expected_fwhm_vtk

    gmin, gmax = float(vtk_v.min()), float(vtk_v.max())
    row = ccm.compute_slice_row(
        slice_num=1, is_special_z0=False, z_requested=0.0, points=points,
        vtk_v=vtk_v, fem_v=fem_v, fallback_mask=np.zeros(len(vtk_v), dtype=bool),
        actual_z=0.0, global_vtk_min=gmin, global_vtk_max=gmax,
    )

    # Feature exists (fwhm_vtk_nm/fwhm_fem_nm/fwhm_diff_nm are real columns
    # computed via _fwhm_along_line). Check sign and rough magnitude
    # (linear-interpolation-based half-max crossing on a discrete grid, so
    # allow ~5% tolerance relative to the fine grid spacing of 0.2).
    grid_spacing = 16.0 / 80
    tol = max(0.05 * expected_fwhm_vtk, 3 * grid_spacing)
    ok = (
        not np.isnan(row["fwhm_vtk_nm"])
        and not np.isnan(row["fwhm_fem_nm"])
        and np.sign(row["fwhm_diff_nm"]) == np.sign(expected_fwhm_diff)
        and close(row["fwhm_vtk_nm"], expected_fwhm_vtk, atol=tol)
        and close(row["fwhm_fem_nm"], expected_fwhm_fem, atol=tol)
    )
    record(
        "8. Known lateral-width (FWHM) difference (sigma_vtk=1.0, sigma_fem=2.0)",
        ok,
        f"expected: fwhm_vtk=2.3548*sigma_vtk={expected_fwhm_vtk:.4f}, "
        f"fwhm_fem=2.3548*sigma_fem={expected_fwhm_fem:.4f}, "
        f"fwhm_diff={expected_fwhm_diff:.4f} (positive, fem wider), tol={tol:.4f} | "
        f"measured: fwhm_vtk_nm={row['fwhm_vtk_nm']:.4f}, fwhm_fem_nm={row['fwhm_fem_nm']:.4f}, "
        f"fwhm_diff_nm={row['fwhm_diff_nm']:.4f}. "
        f"Feature IS implemented (_fwhm_along_line + fwhm_* columns in compute_slice_row).",
    )


# ===========================================================================
# INDEPENDENT CSV SPOT-CHECKS (z=10nm and z=100nm requested; different from
# the prior agent's z=0, z=35, z~1993nm checks)
# ===========================================================================
def spot_check_csv():
    csv_path = REPO_ROOT / "notebooks/afm_vs_leah_compare/run3/metrics/depth_metrics_corrected.csv"
    df = pd.read_csv(csv_path)

    vtk_path = Path("/Users/angelramirez/Downloads/basePotential3d(1).vtk")
    xdmf_path = REPO_ROOT / "notebooks/afm_vs_leah_compare/run3/sige_afm_tip.xdmf"

    ref = read_vtk_reference(vtk_path, field="basePotential")
    case = read_xdmf_case(XdmfCase(label="run3", xdmf_path=xdmf_path, field="phi_V"))
    result = interpolate_case_onto_ref(ref, case, "run3", check_bbox=False)

    global_vtk_min = float(ref.values.min())
    global_vtk_max = float(ref.values.max())
    threshold = ccm.MASK_THRESHOLD_V

    for z_req in (10.0, 100.0):
        csv_row = df[df["z_requested_nm"] == z_req].iloc[0]
        actual_z = float(csv_row["z_actual_nm"])

        # Independent re-slice: use ONLY nearest_z_mask with a tolerance we
        # derive ourselves (tight enough to hit only the single nearest plane),
        # not reusing ccm.nearest_unique_z.
        all_z = result.points[:, 2]
        uz = np.unique(all_z)
        idx = int(np.argmin(np.abs(uz - actual_z)))
        left_gap = uz[idx] - uz[idx - 1] if idx > 0 else np.inf
        right_gap = uz[idx + 1] - uz[idx] if idx < len(uz) - 1 else np.inf
        tol = max(min(left_gap, right_gap) / 2.0, 1e-9)
        mask = nearest_z_mask(result.points, uz[idx], tol=tol)

        pts = result.points[mask]
        vtk_v = result.ref_values[mask]
        fem_v = result.case_values[mask]
        fb = result.fallback_mask[mask]

        n_points = len(vtk_v)
        n_fallback = int(fb.sum())
        n_overlap = n_points - n_fallback
        signed_err = fem_v - vtk_v
        abs_err = np.abs(signed_err)
        rmse = float(np.sqrt(np.mean(signed_err**2)))
        vtk_norm = np.linalg.norm(vtk_v)
        E_L2 = float(np.linalg.norm(signed_err) / vtk_norm) if vtk_norm > 1e-12 else float("nan")
        global_range = global_vtk_max - global_vtk_min
        E_range = rmse / global_range

        sig_mask = np.abs(vtk_v) >= threshold
        n_masked_incl = int(sig_mask.sum())
        n_masked_excl = int((~sig_mask).sum())
        masked_relerr = abs_err[sig_mask] / np.abs(vtk_v[sig_mask])
        masked_relerr_mean = float(np.mean(masked_relerr)) if n_masked_incl else float("nan")
        masked_relerr_max = float(np.max(masked_relerr)) if n_masked_incl else float("nan")

        vtk_shape = vtk_v - vtk_v.mean()
        fem_shape = fem_v - fem_v.mean()
        shape_diff = fem_shape - vtk_shape
        shape_rmse = float(np.sqrt(np.mean(shape_diff**2)))
        std_v, std_f = np.std(fem_v), np.std(vtk_v)
        if std_v < 1e-12 or std_f < 1e-12:
            pearson_r = float("nan")
        else:
            pearson_r = float(np.corrcoef(fem_v, vtk_v)[0, 1])

        checks = {
            "n_vtk_points": (n_points, int(csv_row["n_vtk_points"])),
            "n_overlap": (n_overlap, int(csv_row["n_overlap"])),
            "nearest_fallback_frac": (n_fallback / n_points, float(csv_row["nearest_fallback_frac"])),
            "vtk_min": (float(vtk_v.min()), float(csv_row["vtk_min"])),
            "vtk_max": (float(vtk_v.max()), float(csv_row["vtk_max"])),
            "fem_min": (float(fem_v.min()), float(csv_row["fem_min"])),
            "fem_max": (float(fem_v.max()), float(csv_row["fem_max"])),
            "signed_err_mean": (float(signed_err.mean()), float(csv_row["signed_err_mean"])),
            "abs_err_max": (float(abs_err.max()), float(csv_row["abs_err_max"])),
            "rmse": (rmse, float(csv_row["rmse"])),
            "E_L2": (E_L2, float(csv_row["E_L2"])),
            "E_range": (E_range, float(csv_row["E_range"])),
            "n_masked_included": (n_masked_incl, int(csv_row["n_masked_included"])),
            "n_masked_excluded": (n_masked_excl, int(csv_row["n_masked_excluded"])),
            "masked_relerr_mean": (masked_relerr_mean, float(csv_row["masked_relerr_mean"])),
            "masked_relerr_max": (masked_relerr_max, float(csv_row["masked_relerr_max"])),
            "shape_rmse": (shape_rmse, float(csv_row["shape_rmse"])),
            "pearson_r": (pearson_r, float(csv_row["pearson_r"])),
        }

        all_ok = True
        detail_lines = []
        for name, (mine, theirs) in checks.items():
            match = close(mine, theirs, atol=1e-6, rtol=1e-5)
            all_ok &= match
            detail_lines.append(f"{name}: mine={mine:.6g} csv={theirs:.6g} match={match}")

        record(
            f"CSV spot-check z_requested={z_req}nm (z_actual={actual_z:.4f}nm)",
            all_ok,
            "Independently recomputed from read_vtk_reference/read_xdmf_case/"
            "interpolate_case_onto_ref/nearest_z_mask + raw numpy only (no ccm helpers reused).\n    "
            + "\n    ".join(detail_lines),
        )


# ===========================================================================
def main():
    test1_identical_fields()
    test2_constant_offset()
    test3_multiplicative_scaling()
    test4_shifted_field()
    test5_reflected_field()
    test6_partial_overlap()
    test7_zero_crossing()
    test8_fwhm_width_difference()
    spot_check_csv()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_total = len(RESULTS)
    print("=" * 70)
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    for name, ok, _ in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
