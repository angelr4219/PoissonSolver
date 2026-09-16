#!/usr/bin/env python3
"""Affine tip-voltage least-squares fit: run4 (phi_V, tip=0.0V, the "zero-tip
basis" solution) vs run3 (phi_V, tip=1.0V) vs Leah's VTK reference.

Laplace's equation is linear in the tip Dirichlet value for fixed geometry,
materials, mesh, and bottom voltage, so for any tip voltage V:

    phi(V) = phi_0 + V * (phi_1 - phi_0)

where phi_0 = run4's field (tip=0.0V) and phi_1 = run3's field (tip=1.0V),
both fixed fields on the FEM mesh. The least-squares-optimal V against the
VTK reference, restricted to a point set S, is:

    V* = [(phi_1 - phi_0) . (phi_VTK - phi_0)]_S / [(phi_1 - phi_0) . (phi_1 - phi_0)]_S

This module REUSES (does not reimplement) `read_vtk_reference`,
`read_xdmf_case`, `interpolate_case_onto_ref`, `XdmfCase`, `nearest_z_mask`
from `src/poisson/vtk_xdmf_compare.py`, and follows the same E_L2/E_range/
shape-RMSE/Pearson-r definitions used in
`notebooks/afm_vs_leah_compare/run3/scripts/corrected_comparison_metrics.py`
(read, not imported, since that module's driver is hardwired to run3-only
paths; the metric formulas themselves are reproduced here exactly, with the
same GLOBAL vtk range convention: E_range's denominator is the VTK's global
peak-to-peak range computed once over the WHOLE reference field, never a
per-window range).

Does not modify any existing file. Writes only new artifacts under
notebooks/afm_vs_leah_compare/run3/{metrics,diagnostics} if explicitly asked;
by default just prints, and the driving report script captures stdout / the
returned dict.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
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
RUN4_DIR = REPO_ROOT / "notebooks" / "afm_vs_leah_compare" / "run4"
RUN3_XDMF = RUN3_DIR / "sige_afm_tip.xdmf"
RUN4_XDMF = RUN4_DIR / "sige_afm_tip.xdmf"

EPS = 1e-12
DEVICE_REGION_Z = (0.0, 53.0)  # nm, per run3's material stack


def nearest_unique_z(all_z: np.ndarray, z_target: float) -> tuple[float, float]:
    """Same logic as corrected_comparison_metrics.py's nearest_unique_z: find the
    VTK's actual nearest z-plane and a LOCAL tolerance tight enough to select only
    that plane. `nearest_z_mask`'s own default auto-tol uses the median spacing
    across the ENTIRE (highly non-uniform) z grid, which is too coarse here and
    would silently pull in neighboring, physically distinct z-planes."""
    uz = np.unique(all_z)
    idx = int(np.argmin(np.abs(uz - z_target)))
    actual = float(uz[idx])
    left_gap = actual - uz[idx - 1] if idx > 0 else np.inf
    right_gap = uz[idx + 1] - actual if idx < len(uz) - 1 else np.inf
    gap = min(left_gap, right_gap)
    tol = gap / 2.0 if np.isfinite(gap) else 1e-6
    return actual, max(tol, 1e-9)


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = pearsonr(a, b)
    return float(r)


def fit_v_star(phi0, phi1, phi_vtk, mask):
    d = phi1[mask] - phi0[mask]
    num = np.dot(d, phi_vtk[mask] - phi0[mask])
    den = np.dot(d, d)
    return float(num / den) if abs(den) > EPS else float("nan")


def block_metrics(phi_case, phi_vtk, mask, global_range):
    """E_L2 / E_range / bias / shape-RMSE / Pearson-r over one point subset,
    reproducing corrected_comparison_metrics.py's per-slice formulas exactly,
    but aggregated over an arbitrary boolean mask instead of a single z-plane."""
    v = phi_vtk[mask]
    c = phi_case[mask]
    signed_err = c - v
    rmse = float(np.sqrt(np.mean(signed_err**2)))
    vtk_norm = np.linalg.norm(v)
    E_L2 = float(np.linalg.norm(signed_err) / vtk_norm) if vtk_norm > EPS else float("nan")
    E_range = rmse / global_range if global_range > EPS else float("nan")
    bias = float(signed_err.mean())

    vtk_shape = v - v.mean()
    case_shape = c - c.mean()
    shape_diff = case_shape - vtk_shape
    shape_rmse = float(np.sqrt(np.mean(shape_diff**2)))
    pearson_r = _safe_pearsonr(c, v)

    return {
        "n_points": int(mask.sum()),
        "E_L2": E_L2,
        "E_range": E_range,
        "rmse": rmse,
        "bias_mean_signed_err_V": bias,
        "shape_rmse": shape_rmse,
        "pearson_r": pearson_r,
        "vtk_mean": float(v.mean()),
        "case_mean": float(c.mean()),
    }


def run():
    ref = read_vtk_reference(VTK_PATH, field="basePotential")
    print(f"VTK reference: {len(ref.points)} points, field='{ref.name}', "
          f"range=[{ref.values.min():.6f}, {ref.values.max():.6f}] V")

    case0 = read_xdmf_case(XdmfCase(label="run4_V0", xdmf_path=RUN4_XDMF, field="phi_V"))
    case1 = read_xdmf_case(XdmfCase(label="run3_V1", xdmf_path=RUN3_XDMF, field="phi_V"))

    result0 = interpolate_case_onto_ref(ref, case0, "run4_V0")
    result1 = interpolate_case_onto_ref(ref, case1, "run3_V1")

    # Both interpolated onto the SAME ref (VTK) points -> point sets are identical.
    assert np.allclose(result0.points, result1.points)
    assert np.allclose(result0.ref_values, result1.ref_values)

    points = result0.points
    phi_vtk = result0.ref_values
    phi0 = result0.case_values  # tip = 0.0 V
    phi1 = result1.case_values  # tip = 1.0 V

    z = points[:, 2]

    global_vtk_min = float(ref.values.min())
    global_vtk_max = float(ref.values.max())
    global_range = global_vtk_max - global_vtk_min

    # -----------------------------------------------------------------
    # Coordinate-overlap sanity check restricted to the device region
    # -----------------------------------------------------------------
    device_mask = (z >= DEVICE_REGION_Z[0]) & (z <= DEVICE_REGION_Z[1])
    fb0 = result0.fallback_mask[device_mask]
    fb1 = result1.fallback_mask[device_mask]
    print(f"\nDevice-region point count: {int(device_mask.sum())}")
    print(f"  run4 nearest-neighbor fallback frac (device region): {fb0.mean():.4%}")
    print(f"  run3 nearest-neighbor fallback frac (device region): {fb1.mean():.4%}")

    # -----------------------------------------------------------------
    # Primary fit: whole device region z in [0,53] nm
    # -----------------------------------------------------------------
    v_star_device = fit_v_star(phi0, phi1, phi_vtk, device_mask)
    print(f"\nV* (device region z=0-53nm): {v_star_device:.6f} V")

    # -----------------------------------------------------------------
    # Sensitivity across depth sub-windows
    # -----------------------------------------------------------------
    windows = {}

    m = (z >= 0.0) & (z <= 20.0)
    windows["z_0_20"] = (m, fit_v_star(phi0, phi1, phi_vtk, m))

    m = (z >= 20.0) & (z <= 53.0)
    windows["z_20_53"] = (m, fit_v_star(phi0, phi1, phi_vtk, m))

    actual_35, tol_35 = nearest_unique_z(z, 35.0)
    m = nearest_z_mask(points, actual_35, tol=tol_35)
    windows["z_35_slice"] = (m, fit_v_star(phi0, phi1, phi_vtk, m))

    actual_48, tol_48 = nearest_unique_z(z, 48.0)
    m = nearest_z_mask(points, actual_48, tol=tol_48)
    windows["z_48_slice"] = (m, fit_v_star(phi0, phi1, phi_vtk, m))

    actual_0, tol_0 = nearest_unique_z(z, 0.0)
    m = nearest_z_mask(points, actual_0, tol=tol_0)
    windows["z_0_slice"] = (m, fit_v_star(phi0, phi1, phi_vtk, m))

    actual_9p67, tol_9p67 = nearest_unique_z(z, 9.666667)
    m = nearest_z_mask(points, actual_9p67, tol=tol_9p67)
    windows["z_9p67_slice"] = (m, fit_v_star(phi0, phi1, phi_vtk, m))

    print("\nSensitivity of V* across depth windows:")
    for name, (m, vstar) in windows.items():
        print(f"  {name:16s} n={int(m.sum()):5d}  V* = {vstar: .6f} V")

    # -----------------------------------------------------------------
    # Before/after metrics using the device-region V* applied everywhere
    # it's evaluated (device region aggregate + z=35 slice specifically)
    # -----------------------------------------------------------------
    phi_fit_device = phi0 + v_star_device * (phi1 - phi0)

    before_device = block_metrics(phi1, phi_vtk, device_mask, global_range)
    after_device = block_metrics(phi_fit_device, phi_vtk, device_mask, global_range)

    mask_z35 = windows["z_35_slice"][0]
    before_z35 = block_metrics(phi1, phi_vtk, mask_z35, global_range)
    # Use the z=35-specific V* (best-possible fit AT that depth) as well as the
    # device-region V*, to separately test "can ANY voltage fix z=35" vs
    # "does the device-region-optimal voltage fix z=35".
    v_star_z35 = windows["z_35_slice"][1]
    phi_fit_z35_local = phi0 + v_star_z35 * (phi1 - phi0)
    after_z35_localfit = block_metrics(phi_fit_z35_local, phi_vtk, mask_z35, global_range)
    after_z35_devicefit = block_metrics(phi_fit_device, phi_vtk, mask_z35, global_range)

    print("\n--- Device-region (z=0-53nm) aggregate metrics ---")
    print(f"  BEFORE (V=1.0, run3):        E_L2={before_device['E_L2']:.4f}  "
          f"E_range={before_device['E_range']:.4f}  bias={before_device['bias_mean_signed_err_V']:.4f}  "
          f"shape_rmse={before_device['shape_rmse']:.4f}  pearson_r={before_device['pearson_r']:.4f}")
    print(f"  AFTER  (V*={v_star_device:.4f}):    E_L2={after_device['E_L2']:.4f}  "
          f"E_range={after_device['E_range']:.4f}  bias={after_device['bias_mean_signed_err_V']:.4f}  "
          f"shape_rmse={after_device['shape_rmse']:.4f}  pearson_r={after_device['pearson_r']:.4f}")

    print("\n--- z=35nm slice specifically ---")
    print(f"  BEFORE (V=1.0, run3):                 E_range={before_z35['E_range']:.4f}  "
          f"pearson_r={before_z35['pearson_r']:.4f}")
    print(f"  AFTER, device-region V*={v_star_device:.4f}:  E_range={after_z35_devicefit['E_range']:.4f}  "
          f"pearson_r={after_z35_devicefit['pearson_r']:.4f}")
    print(f"  AFTER, z35-local-optimal V*={v_star_z35:.4f}: E_range={after_z35_localfit['E_range']:.4f}  "
          f"pearson_r={after_z35_localfit['pearson_r']:.4f}")

    summary = {
        "vtk_path": str(VTK_PATH),
        "run3_xdmf": str(RUN3_XDMF),
        "run4_xdmf": str(RUN4_XDMF),
        "global_vtk_min_V": global_vtk_min,
        "global_vtk_max_V": global_vtk_max,
        "global_range_V": global_range,
        "device_region_z_nm": list(DEVICE_REGION_Z),
        "device_region_n_points": int(device_mask.sum()),
        "device_region_fallback_frac_run4": float(fb0.mean()),
        "device_region_fallback_frac_run3": float(fb1.mean()),
        "v_star_device_region": v_star_device,
        "v_star_windows": {name: vstar for name, (_, vstar) in windows.items()},
        "before_device_region": before_device,
        "after_device_region": after_device,
        "before_z35": before_z35,
        "after_z35_devicefit": after_z35_devicefit,
        "after_z35_localfit": after_z35_localfit,
        "v_star_z35_local": v_star_z35,
    }

    out_path = RUN3_DIR / "diagnostics" / "voltage_basis_fit_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    return summary


if __name__ == "__main__":
    run()
