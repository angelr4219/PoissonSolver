#!/usr/bin/env python3
"""Depth-profile and cross-sectional visualizations for run3 vs Leah's VTK reference.

Read-only w.r.t. the repo's comparison module: this script only IMPORTS from
src/poisson/vtk_xdmf_compare.py (never modifies it) and reuses its primitives
(read_vtk_reference, read_xdmf_case, interpolate_case_onto_ref, nearest_z_mask,
nearest_coord_mask, plot_cross_section) rather than reimplementing them.

This module computes everything itself from those primitives -- it does not
depend on any other agent's concurrently-produced files. If
notebooks/afm_vs_leah_compare/run3/metrics/depth_metrics_corrected.csv happens
to already exist, it is read purely as an optional convenience overlay; its
absence changes nothing.

Scope note (see diagnostics/physical_case_audit.md): it is UNRESOLVED whether
basePotential3d(1).vtk includes an AFM-tip contribution at all. Every title/
caption below is a factual description of the data shown (field name, source
file, depth, quantity plotted) -- none of them assert that either field's
geometry or tip voltage is "correct."

Interface depths (material_id transitions, from run3's own mesh, confirmed in
physical_case_audit.md): z = 2, 30, 40, 43, 53 nm
  Si 0-2 / SiGe 2-30 / Si 30-40 / SiGe 40-43 / Si 43-53 / SiGe buffer 53-2053(bottom, -4.4V)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from poisson.vtk_xdmf_compare import (  # noqa: E402
    InterpResult,
    XdmfCase,
    interpolate_case_onto_ref,
    nearest_coord_mask,
    plot_cross_section,
    read_vtk_reference,
    read_xdmf_case,
)

VTK_PATH = Path("/Users/angelramirez/Downloads/basePotential3d(1).vtk")
RUN3_DIR = REPO_ROOT / "notebooks" / "afm_vs_leah_compare" / "run3"
XDMF_PATH = RUN3_DIR / "sige_afm_tip.xdmf"
PLOTS_DIR = RUN3_DIR / "plots"
DEPTH_DIR = PLOTS_DIR / "depth_profiles"
XY_DIR = PLOTS_DIR / "representative_xy"
XZ_DIR = PLOTS_DIR / "representative_xz"
YZ_DIR = PLOTS_DIR / "representative_yz"

INTERFACE_Z_NM = [2.0, 30.0, 40.0, 43.0, 53.0]
REL_ERR_FLOOR_V = 0.044  # |VTK| >= this to count in masked relative-error stats
CASE_LABEL = "run3 FEM (phi_V)"
UNITS = "nm"


def _ensure_dirs() -> None:
    for d in (DEPTH_DIR, XY_DIR, XZ_DIR, YZ_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_result() -> InterpResult:
    """One-time load + one-time interpolation, reused for every depth and slice."""
    ref = read_vtk_reference(VTK_PATH, field="basePotential")
    case = read_xdmf_case(XdmfCase("run3", XDMF_PATH, field="phi_V", scale=1.0))
    result = interpolate_case_onto_ref(ref, case, "run3")
    print(f"Loaded ref: {len(ref.points)} pts, case: {len(case.points)} pts")
    return result


def build_depth_sweep_z() -> np.ndarray:
    z1 = np.arange(1.0, 300.0 + 1e-9, 1.0)
    z2 = np.arange(300.0, 2000.0 + 1e-9, 5.0)
    z = np.unique(np.concatenate([z1, z2]))
    return z


def compute_depth_profiles(result: InterpResult) -> pd.DataFrame:
    """Compute all depth-profile quantities (items 1-10) from a single
    already-computed InterpResult, reusing nearest_z_mask per depth."""
    z_values = build_depth_sweep_z()
    global_range = float(result.ref_values.max() - result.ref_values.min())

    rows = []
    for z in z_values:
        mask = nearest_coord_mask(result.points, 2, z)
        n = int(mask.sum())
        if n < 1:
            continue
        vtk_v = result.ref_values[mask]
        fem_v = result.case_values[mask]
        fb = result.fallback_mask[mask]
        diff = fem_v - vtk_v

        vtk_mean = float(np.mean(vtk_v))
        fem_mean = float(np.mean(fem_v))
        vtk_shape = vtk_v - vtk_mean
        fem_shape = fem_v - fem_mean
        shape_diff = fem_shape - vtk_shape
        shape_rmse = float(np.sqrt(np.mean(shape_diff**2)))

        if n > 1 and np.std(vtk_shape) > 0 and np.std(fem_shape) > 0:
            pearson_r = float(np.corrcoef(vtk_shape, fem_shape)[0, 1])
        else:
            pearson_r = float("nan")

        vtk_norm = np.linalg.norm(vtk_v)
        rel_l2 = float(np.linalg.norm(diff) / vtk_norm) if vtk_norm > 1e-12 else float("nan")
        rmse = float(np.sqrt(np.mean(diff**2)))

        rows.append(
            dict(
                z_nm=float(z),
                n_points=n,
                vtk_min=float(vtk_v.min()),
                vtk_mean=vtk_mean,
                vtk_max=float(vtk_v.max()),
                fem_min=float(fem_v.min()),
                fem_mean=fem_mean,
                fem_max=float(fem_v.max()),
                mean_signed_err=float(np.mean(diff)),
                mean_abs_err=float(np.mean(np.abs(diff))),
                max_abs_err=float(np.max(np.abs(diff))),
                rmse=rmse,
                rel_l2_error=rel_l2,
                normalized_rmse=rmse / global_range if global_range > 0 else float("nan"),
                valid_overlap_frac=float((~fb).mean()),
                nearest_fallback_frac=float(fb.mean()),
                shape_rmse=shape_rmse,
                pearson_r=pearson_r,
            )
        )
    df = pd.DataFrame(rows)
    return df


def save_depth_data(df: pd.DataFrame) -> tuple[Path, Path]:
    csv_path = DEPTH_DIR / "depth_profile_data.csv"
    npz_path = DEPTH_DIR / "depth_profile_data.npz"
    df.to_csv(csv_path, index=False)
    np.savez(npz_path, **{c: df[c].to_numpy() for c in df.columns})
    return csv_path, npz_path


def _mark_interfaces(ax) -> None:
    for zi in INTERFACE_Z_NM:
        ax.axvline(zi, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)


def plot_depth_profiles(df: pd.DataFrame) -> list[Path]:
    z = df["z_nm"].to_numpy()
    saved: list[Path] = []

    # Figure 1: potential envelopes (items 1 & 2), two panels sharing x
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax = axes[0]
    ax.plot(z, df["vtk_min"], color="#1f77b4", label="min")
    ax.plot(z, df["vtk_mean"], color="#000000", label="mean")
    ax.plot(z, df["vtk_max"], color="#d62728", label="max")
    _mark_interfaces(ax)
    ax.set_ylabel("VTK potential (V)")
    ax.set_title("Leah's VTK reference (basePotential): min/mean/max per depth slice")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(z, df["fem_min"], color="#1f77b4", label="min")
    ax.plot(z, df["fem_mean"], color="#000000", label="mean")
    ax.plot(z, df["fem_max"], color="#d62728", label="max")
    _mark_interfaces(ax)
    ax.set_xlabel("depth z (nm)")
    ax.set_ylabel("FEM potential (V)")
    ax.set_title(f"{CASE_LABEL}: min/mean/max per depth slice")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = DEPTH_DIR / "01_02_potential_envelopes_vs_depth.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Figure 2: error magnitudes (items 3, 4, 5) -- signed error, |err| mean/max, RMSE
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax = axes[0]
    ax.axhline(0, color="k", linewidth=0.6)
    ax.plot(z, df["mean_signed_err"], color="#9467bd")
    _mark_interfaces(ax)
    ax.set_ylabel("mean signed error\n(FEM - VTK) [V]")
    ax.set_title("Depth vs mean signed error (item 3)")

    ax = axes[1]
    ax.plot(z, df["mean_abs_err"], label="mean |error|", color="#2ca02c")
    ax.plot(z, df["max_abs_err"], label="max |error|", color="#ff7f0e")
    _mark_interfaces(ax)
    ax.set_ylabel("absolute error (V)")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Depth vs mean and max absolute error (item 4)")

    ax = axes[2]
    ax.plot(z, df["rmse"], color="#17becf")
    _mark_interfaces(ax)
    ax.set_xlabel("depth z (nm)")
    ax.set_ylabel("RMSE (V)")
    ax.set_yscale("log")
    ax.set_title("Depth vs RMSE (item 5)")
    fig.tight_layout()
    p = DEPTH_DIR / "03_04_05_error_magnitude_vs_depth.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Figure 3: relative-error metrics (items 6, 7)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax = axes[0]
    ax.plot(z, df["rel_l2_error"], color="#8c564b")
    _mark_interfaces(ax)
    ax.set_ylabel("relative L2 error\n||FEM-VTK|| / ||VTK|| (per slice)")
    ax.set_yscale("log")
    ax.set_title("Depth vs per-slice relative L2 error (item 6)")

    ax = axes[1]
    ax.plot(z, df["normalized_rmse"], color="#e377c2")
    _mark_interfaces(ax)
    ax.set_xlabel("depth z (nm)")
    ax.set_ylabel("RMSE / (global VTK max - min)")
    ax.set_title("Depth vs voltage-range-normalized RMSE (item 7)")
    fig.tight_layout()
    p = DEPTH_DIR / "06_07_relative_error_vs_depth.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Figure 4: interpolation validity sanity check (item 8)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(z, 100 * df["valid_overlap_frac"], label="valid overlap fraction [%]", color="#2ca02c")
    ax.plot(z, 100 * df["nearest_fallback_frac"], label="nearest-fallback fraction [%]", color="#d62728")
    _mark_interfaces(ax)
    ax.set_xlabel("depth z (nm)")
    ax.set_ylabel("fraction of slice points [%]")
    ax.set_ylim(-5, 105)
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("Depth vs valid-overlap / nearest-fallback fraction (item 8, reproducibility check)")
    fig.tight_layout()
    p = DEPTH_DIR / "08_interpolation_validity_vs_depth.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Figure 5: shape RMSE and Pearson r (items 9, 10) -- both computed on
    # mean-subtracted ("shape") values per depth slice, consistently.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax = axes[0]
    ax.plot(z, df["shape_rmse"], color="#bcbd22")
    _mark_interfaces(ax)
    ax.set_ylabel("shape RMSE (V)\n(mean-subtracted FEM vs VTK)")
    ax.set_yscale("log")
    ax.set_title("Depth vs mean-subtracted 'shape' error RMSE (item 9)")

    ax = axes[1]
    ax.plot(z, df["pearson_r"], color="#17becf")
    _mark_interfaces(ax)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("depth z (nm)")
    ax.set_ylabel("Pearson r\n(mean-subtracted values)")
    ax.set_title("Depth vs spatial Pearson correlation, VTK vs FEM (item 10; computed on mean-subtracted values, consistent with item 9)")
    fig.tight_layout()
    p = DEPTH_DIR / "09_10_shape_error_and_correlation_vs_depth.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


def pick_deep_z(df: pd.DataFrame) -> float:
    """Pick the deeper XY representative depth from the depth-profile results
    themselves, per task instructions: not the trivial near-bottom boundary,
    but wherever shape-error is most informative. We use the z (restricted to
    z > 60nm, i.e. below the documented device stack, so it's a distinct
    'deep buffer' comparison point rather than a repeat of the shallow
    device-region slice at z=35) with the maximum shape_rmse -- the depth
    where the FEM and VTK spatial *patterns* (independent of any constant
    offset) disagree the most.
    """
    sub = df[(df["z_nm"] > 60.0) & (df["z_nm"] < 2000.0)]
    if sub.empty:
        return 100.0
    idx = sub["shape_rmse"].idxmax()
    return float(df.loc[idx, "z_nm"])


def _layer_label(z: float) -> str:
    if z < 2:
        return "Si cap (0-2nm)"
    if z < 30:
        return "SiGe (2-30nm)"
    if z < 40:
        return "Si well (30-40nm)"
    if z < 43:
        return "SiGe barrier (40-43nm)"
    if z < 53:
        return "Si (43-53nm)"
    return "SiGe buffer (53-2053nm, bottom=-4.4V)"


def _masked_rel_err(vtk_v: np.ndarray, fem_v: np.ndarray, floor: float = REL_ERR_FLOOR_V) -> np.ndarray:
    rel = np.full_like(vtk_v, np.nan)
    sig = np.abs(vtk_v) >= floor
    rel[sig] = 100.0 * np.abs(fem_v[sig] - vtk_v[sig]) / np.abs(vtk_v[sig])
    return rel


def make_8panel_cross_section(
    result: InterpResult,
    axis: int,
    coord_value: float,
    plane_name: str,
    save_path: Path,
    tol: float | None = None,
    zoom_z: tuple[float, float] | None = None,
    title_suffix: str = "",
) -> Path | None:
    """Custom 8-panel cross-section figure (VTK, FEM, signed diff, abs error,
    masked relative error, VTK shape, FEM shape, shape diff) for one plane.

    axis=2 -> xy plane at z=coord_value (h=x, v=y)
    axis=1 -> xz plane at y=coord_value (h=x, v=z)
    axis=0 -> yz plane at x=coord_value (h=y, v=z)
    If zoom_z is given (only meaningful when z is the vertical axis), an
    additional vertical-range restriction is applied and color limits are
    recomputed from the zoomed data only.
    """
    mask = nearest_coord_mask(result.points, axis, coord_value, tol=tol)
    if mask.sum() < 4:
        print(f"  [skip] {plane_name}: only {mask.sum()} points")
        return None

    other_idx = [i for i in range(3) if i != axis]
    other_names = [{0: "x", 1: "y", 2: "z"}[i] for i in other_idx]
    h = result.points[mask, other_idx[0]]
    v = result.points[mask, other_idx[1]]
    vtk_v = result.ref_values[mask]
    fem_v = result.case_values[mask]

    z_is_vertical = other_names[1] == "z"
    if zoom_z is not None and z_is_vertical:
        zmask = (v >= zoom_z[0]) & (v <= zoom_z[1])
        h, v, vtk_v, fem_v = h[zmask], v[zmask], vtk_v[zmask], fem_v[zmask]
        if len(h) < 4:
            print(f"  [skip] {plane_name}: only {len(h)} points in zoom range")
            return None

    diff = fem_v - vtk_v
    abs_err = np.abs(diff)
    masked_rel = _masked_rel_err(vtk_v, fem_v)
    vtk_mean = float(np.mean(vtk_v))
    fem_mean = float(np.mean(fem_v))
    vtk_shape = vtk_v - vtk_mean
    fem_shape = fem_v - fem_mean
    shape_diff = fem_shape - vtk_shape

    # Shared potential color limits between VTK & FEM panels of THIS figure,
    # computed from THIS figure's own data (zoomed data uses its own range).
    pot_vmin = float(min(vtk_v.min(), fem_v.min()))
    pot_vmax = float(max(vtk_v.max(), fem_v.max()))

    diff_lim = float(np.nanpercentile(np.abs(diff), 99)) or 1e-9
    shape_lim = float(max(np.nanpercentile(np.abs(vtk_shape), 99), np.nanpercentile(np.abs(fem_shape), 99))) or 1e-9
    shape_diff_lim = float(np.nanpercentile(np.abs(shape_diff), 99)) or 1e-9
    abs_err_vmax = float(np.nanpercentile(abs_err, 99)) or 1e-9
    rel_finite = masked_rel[np.isfinite(masked_rel)]
    rel_vmax = float(np.nanpercentile(rel_finite, 99)) if rel_finite.size else 1.0

    fig, axes = plt.subplots(2, 4, figsize=(24, 11 if z_is_vertical else 10))
    axes = axes.ravel()

    panels = [
        (vtk_v, "Leah's VTK potential", "viridis", pot_vmin, pot_vmax, "V"),
        (fem_v, f"{CASE_LABEL} potential", "viridis", pot_vmin, pot_vmax, "V"),
        (diff, "Signed difference (FEM - VTK)", "RdBu_r", -diff_lim, diff_lim, "V"),
        (abs_err, "Absolute error |FEM - VTK|", "magma", 0, abs_err_vmax, "V"),
        (masked_rel, f"Masked relative error [%]\n(secondary; only |VTK|>={REL_ERR_FLOOR_V}V)", "magma", 0, rel_vmax, "%"),
        (vtk_shape, "VTK 'shape' (mean-subtracted)", "RdBu_r", -shape_lim, shape_lim, "V"),
        (fem_shape, "FEM 'shape' (mean-subtracted)", "RdBu_r", -shape_lim, shape_lim, "V"),
        (shape_diff, "Shape difference\n(FEM_shape - VTK_shape)", "RdBu_r", -shape_diff_lim, shape_diff_lim, "V"),
    ]
    for ax, (vals, title, cmap, vmin, vmax, unit) in zip(axes, panels):
        sc = ax.scatter(h, v, c=vals, cmap=cmap, s=6, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{other_names[0]} (nm)")
        ax.set_ylabel(f"{other_names[1]} (nm)")
        if z_is_vertical:
            ax.invert_yaxis()
            for zi in INTERFACE_Z_NM:
                if (zoom_z is None) or (zoom_z[0] <= zi <= zoom_z[1]):
                    ax.axhline(zi, color="gray", linestyle=":", linewidth=0.7, alpha=0.8)
        else:
            ax.set_aspect("equal")
        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(unit)

    fig.suptitle(f"{plane_name}{title_suffix}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def make_cross_sections(result: InterpResult, deep_z: float) -> list[Path]:
    saved: list[Path] = []

    # --- XY planes: z=35nm (Si well center, Leah's stated benchmark depth)
    # and the data-driven "deep_z" pick (max shape RMSE for z>60nm). ---
    for z, tag in [(35.0, "z35_well"), (deep_z, f"z{deep_z:g}_deep")]:
        p = XY_DIR / f"xy_8panel_{tag}.png"
        r = make_8panel_cross_section(
            result, axis=2, coord_value=z, plane_name=f"XY plane at z={z:.4g}nm ({_layer_label(z)})",
            save_path=p,
        )
        if r:
            saved.append(r)

    # --- XZ plane at y=0: reuse plot_cross_section (existing 4-panel), full range ---
    p = XZ_DIR / "xz_y0_reuse_4panel_full.png"
    plot_cross_section(result, axis="y", coord_value=0.0, label=CASE_LABEL, units=UNITS,
                        save_path=p, show=False)
    saved.append(p)
    # custom 8-panel, full range
    p = XZ_DIR / "xz_y0_8panel_full.png"
    r = make_8panel_cross_section(result, axis=1, coord_value=0.0,
                                   plane_name="XZ plane at y=0nm (full depth range)", save_path=p)
    if r:
        saved.append(r)
    # device-zoomed: reuse (still calls same function; caller restricts via subset not supported
    # by plot_cross_section directly, so we build a temporary zoomed InterpResult for the reuse call)
    zmask_full = nearest_coord_mask(result.points, 1, 0.0)
    zmask_zoom = zmask_full & (result.points[:, 2] >= 0) & (result.points[:, 2] <= 100)
    zoomed_result = InterpResult(
        points=result.points[zmask_zoom],
        ref_values=result.ref_values[zmask_zoom],
        case_values=result.case_values[zmask_zoom],
        fallback_mask=result.fallback_mask[zmask_zoom],
    )
    p = XZ_DIR / "xz_y0_reuse_4panel_devicezoom.png"
    plot_cross_section(zoomed_result, axis="y", coord_value=0.0, label=CASE_LABEL, units=UNITS,
                        save_path=p, show=False)
    saved.append(p)
    p = XZ_DIR / "xz_y0_8panel_devicezoom.png"
    r = make_8panel_cross_section(result, axis=1, coord_value=0.0,
                                   plane_name="XZ plane at y=0nm (device-region zoom, z in [0,100]nm)",
                                   save_path=p, zoom_z=(0.0, 100.0))
    if r:
        saved.append(r)

    # --- YZ plane at x=0: same treatment ---
    p = YZ_DIR / "yz_x0_reuse_4panel_full.png"
    plot_cross_section(result, axis="x", coord_value=0.0, label=CASE_LABEL, units=UNITS,
                        save_path=p, show=False)
    saved.append(p)
    p = YZ_DIR / "yz_x0_8panel_full.png"
    r = make_8panel_cross_section(result, axis=0, coord_value=0.0,
                                   plane_name="YZ plane at x=0nm (full depth range)", save_path=p)
    if r:
        saved.append(r)

    xmask_full = nearest_coord_mask(result.points, 0, 0.0)
    xmask_zoom = xmask_full & (result.points[:, 2] >= 0) & (result.points[:, 2] <= 100)
    zoomed_result_x = InterpResult(
        points=result.points[xmask_zoom],
        ref_values=result.ref_values[xmask_zoom],
        case_values=result.case_values[xmask_zoom],
        fallback_mask=result.fallback_mask[xmask_zoom],
    )
    p = YZ_DIR / "yz_x0_reuse_4panel_devicezoom.png"
    plot_cross_section(zoomed_result_x, axis="x", coord_value=0.0, label=CASE_LABEL, units=UNITS,
                        save_path=p, show=False)
    saved.append(p)
    p = YZ_DIR / "yz_x0_8panel_devicezoom.png"
    r = make_8panel_cross_section(result, axis=0, coord_value=0.0,
                                   plane_name="YZ plane at x=0nm (device-region zoom, z in [0,100]nm)",
                                   save_path=p, zoom_z=(0.0, 100.0))
    if r:
        saved.append(r)

    return saved


def run_all() -> dict:
    _ensure_dirs()
    result = load_result()

    print("Computing depth-profile sweep (~640 depths, single interpolation reused)...")
    df = compute_depth_profiles(result)
    csv_path, npz_path = save_depth_data(df)
    print(f"Saved depth-profile data: {csv_path}, {npz_path}")

    depth_plots = plot_depth_profiles(df)
    print(f"Saved {len(depth_plots)} depth-profile figures.")

    deep_z = pick_deep_z(df)
    print(f"Data-driven deep XY depth pick: z={deep_z:.4g}nm (max shape_rmse for z>60nm)")

    xsec_plots = make_cross_sections(result, deep_z)
    print(f"Saved {len(xsec_plots)} cross-section figures.")

    return {
        "depth_csv": csv_path,
        "depth_npz": npz_path,
        "depth_plots": depth_plots,
        "deep_z": deep_z,
        "xsec_plots": xsec_plots,
    }


def verify(outputs: dict) -> bool:
    print("\n--- Verification checklist ---")
    all_ok = True
    files_to_check = (
        [outputs["depth_csv"], outputs["depth_npz"]]
        + outputs["depth_plots"]
        + outputs["xsec_plots"]
    )
    for f in files_to_check:
        ok = Path(f).exists() and Path(f).stat().st_size > 0
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISSING'}] {f}")
    print(f"\nAll files present: {all_ok}")
    return all_ok


if __name__ == "__main__":
    outputs = run_all()
    ok = verify(outputs)
    if not ok:
        sys.exit(1)
    print("\nDone.")
