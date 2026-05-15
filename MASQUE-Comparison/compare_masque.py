#!/usr/bin/env python3
"""
MASQUE-Comparison/compare_masque.py
-------------------------------------
Clean, correct VTK vs XDMF/H5 comparison.

Strategy:
  - VTK:  scipy RegularGridInterpolator (trilinear) on the rectilinear VTK grid
  - XDMF: DOLFINx bb_tree + FEM evaluation (exact interpolant from FEM basis)
  Both evaluated on the SAME shared Cartesian x-y grid at each z-slice.
  This is a true 1-to-1 pointwise comparison.

Physics:
  V_gate = -1 V, V_bottom = -12 V
  R_disk = 10 nm, box 500x500x255 nm
  Layers: 50 nm SiGe (eps=13.05) / 5 nm sSi (eps=11.7) / 200 nm SiGe (eps=13.05)

Run via:
  ./MASQUE-Comparison/run_compare.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from mpi4py import MPI
import h5py
from dolfinx import fem, io
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

# ── Constants ──────────────────────────────────────────────────────────────────

PHI_VMIN = -12.0    # shared color scale for all phi plots
PHI_VMAX = -1.0
ACCEPT_Z_NM   = 155.0
ACCEPT_PCT    = 5.0     # "failed" threshold — will flag and report
PERCENT_FLOOR = 0.01    # 10 mV floor on |ref| to avoid divide-by-zero

SLICES_NM = [0.0, 10.0, 20.0, 51.0, 155.0, 255.0]


# ── VTK loading (pure ASCII, no pyvista) ──────────────────────────────────────

def _read_n_floats(lines, start, n):
    vals = []
    i = start
    while len(vals) < n and i < len(lines):
        vals.extend(lines[i].split())
        i += 1
    return np.array(vals[:n], dtype=np.float64), i


def load_vtk(vtk_path: Path) -> dict:
    """
    Parse an ASCII VTK RectilinearGrid.
    Returns dict with x, y, z arrays (nm) and phi_3d[ix,iy,iz] (V).
    """
    with open(vtk_path, "r") as f:
        lines = f.readlines()

    if not lines[0].strip().startswith("# vtk"):
        raise ValueError(f"Not a VTK file: {vtk_path}")
    if lines[2].strip().upper() != "ASCII":
        raise ValueError("Only ASCII VTK files supported")

    nx = ny = nz = None
    x_arr = y_arr = z_arr = phi_flat = None
    field_name = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        up = line.upper()
        if up.startswith("DIMENSIONS"):
            p = line.split()
            nx, ny, nz = int(p[1]), int(p[2]), int(p[3])
        elif up.startswith("X_COORDINATES"):
            n = int(line.split()[1])
            x_arr, i = _read_n_floats(lines, i + 1, n)
            continue
        elif up.startswith("Y_COORDINATES"):
            n = int(line.split()[1])
            y_arr, i = _read_n_floats(lines, i + 1, n)
            continue
        elif up.startswith("Z_COORDINATES"):
            n = int(line.split()[1])
            z_arr, i = _read_n_floats(lines, i + 1, n)
            continue
        elif up.startswith("SCALARS") and phi_flat is None:
            field_name = line.split()[1]
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if lines[j].strip().upper().startswith("LOOKUP_TABLE"):
                j += 1
            n_pts = nx * ny * nz
            phi_flat, i = _read_n_floats(lines, j, n_pts)
            continue
        i += 1

    if phi_flat is None:
        raise ValueError(f"Could not parse VTK field from {vtk_path}")

    phi_3d = phi_flat.reshape((nx, ny, nz), order="F")

    print(f"\n── VTK ───────────────────────────────────────────")
    print(f"  file    : {vtk_path}")
    print(f"  field   : {field_name}")
    print(f"  dims    : {nx} × {ny} × {nz}")
    print(f"  x range : [{x_arr.min():.2f}, {x_arr.max():.2f}] nm")
    print(f"  y range : [{y_arr.min():.2f}, {y_arr.max():.2f}] nm")
    print(f"  z range : [{z_arr.min():.2f}, {z_arr.max():.2f}] nm")
    print(f"  phi     : [{phi_3d.min():.6g}, {phi_3d.max():.6g}] V")

    return dict(x=x_arr, y=y_arr, z=z_arr, phi=phi_3d, field_name=field_name)


def make_vtk_interpolator(vtk: dict) -> RegularGridInterpolator:
    """scipy trilinear interpolator on the VTK rectilinear grid."""
    return RegularGridInterpolator(
        (vtk["x"], vtk["y"], vtk["z"]),
        vtk["phi"],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


# ── XDMF / H5 loading ────────────────────────────────────────────────────────

def _parse_xdmf_dataset(xdmf_path: Path, field_name: str) -> tuple[Path, str]:
    """Extract (h5_file, dataset_path) from XDMF XML for a named field."""
    tree = ET.parse(str(xdmf_path))
    root = tree.getroot()
    for attr in root.iter():
        if attr.tag.endswith("Attribute") and attr.attrib.get("Name") == field_name:
            for child in attr.iter():
                if child.tag.endswith("DataItem") and child.text:
                    text = child.text.strip()
                    if ".h5:" in text:
                        h5_name, dset = text.split(":", 1)
                        return (xdmf_path.parent / h5_name.strip()).resolve(), dset.strip()
    raise KeyError(f"Field '{field_name}' not found in {xdmf_path}")


def load_xdmf(xdmf_path: Path, field_name: str = "phi") -> tuple:
    """
    Load a DOLFINx CG1 FEM function from XDMF+H5.
    Returns (mesh, V, u).
    """
    comm = MPI.COMM_WORLD
    xdmf_path = xdmf_path.resolve()

    h5_file, dataset = _parse_xdmf_dataset(xdmf_path, field_name)
    print(f"\n── XDMF ──────────────────────────────────────────")
    print(f"  xdmf    : {xdmf_path}")
    print(f"  h5      : {h5_file}")
    print(f"  dataset : {dataset}")

    with io.XDMFFile(comm, str(xdmf_path), "r") as xf:
        msh = xf.read_mesh(name="mesh")

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V, name=field_name)

    with h5py.File(str(h5_file), "r") as h5:
        data = np.asarray(h5[dataset], dtype=np.float64).reshape(-1)

    if data.size != u.x.array.size:
        raise RuntimeError(
            f"DOF mismatch: XDMF expects {u.x.array.size} dofs, H5 has {data.size}. "
            f"Check the dataset path: {dataset}"
        )
    u.x.array[:] = data
    u.x.scatter_forward()

    coords_nm = V.tabulate_dof_coordinates() * 1e9
    phi = u.x.array
    print(f"  dofs    : {phi.size}")
    print(f"  x range : [{coords_nm[:,0].min():.2f}, {coords_nm[:,0].max():.2f}] nm")
    print(f"  y range : [{coords_nm[:,1].min():.2f}, {coords_nm[:,1].max():.2f}] nm")
    print(f"  z range : [{coords_nm[:,2].min():.2f}, {coords_nm[:,2].max():.2f}] nm")
    print(f"  phi     : [{phi.min():.6g}, {phi.max():.6g}] V")

    return msh, V, u


# ── FEM evaluation at arbitrary points ───────────────────────────────────────

def eval_fem_at_points(u: fem.Function, pts_m: np.ndarray) -> np.ndarray:
    """Evaluate DOLFINx FEM function at physical points (metres)."""
    msh = u.function_space.mesh
    tree = bb_tree(msh, msh.topology.dim)
    cand = compute_collisions_points(tree, pts_m)
    coll = compute_colliding_cells(msh, cand, pts_m)

    n = pts_m.shape[0]
    inside = np.zeros(n, dtype=bool)
    cells  = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        links = coll.links(i)
        if len(links) > 0:
            inside[i] = True
            cells[i]  = links[0]

    vals = np.full(n, np.nan, dtype=np.float64)
    if np.any(inside):
        v = np.asarray(u.eval(pts_m[inside], cells[inside])).reshape(-1)
        vals[inside] = v
    return vals


# ── Per-slice comparison ──────────────────────────────────────────────────────

def compare_at_z(
    z_req_nm: float,
    vtk: dict,
    vtk_interp: RegularGridInterpolator,
    u: fem.Function,
    grid_n: int,
    outdir: Path,
    label: str,
    qw_exclude: tuple[float, float] | None = None,
) -> dict:
    """
    Compare VTK and XDMF on the shared x-y grid at the nearest VTK z to z_req_nm.
    Returns dict of error metrics.
    """
    # Snap to nearest VTK z
    k = int(np.argmin(np.abs(vtk["z"] - z_req_nm)))
    z_nm = float(vtk["z"][k])

    vtk_zmin, vtk_zmax = float(vtk["z"].min()), float(vtk["z"].max())
    if z_req_nm < vtk_zmin or z_req_nm > vtk_zmax:
        print(f"  [skip] z={z_req_nm:.1f} nm outside VTK range [{vtk_zmin:.1f}, {vtk_zmax:.1f}]")
        return {}

    qw_excluded = (
        qw_exclude is not None
        and qw_exclude[0] <= z_req_nm <= qw_exclude[1]
    )
    excl_tag = "  [QW band-offset excluded]" if qw_excluded else ""
    print(f"\n  ── z = {z_req_nm:.1f} nm  →  VTK z = {z_nm:.4f} nm (k={k}){excl_tag} ──")

    # Shared x-y grid (VTK native bounds, uniform)
    xg = np.linspace(float(vtk["x"].min()), float(vtk["x"].max()), grid_n)
    yg = np.linspace(float(vtk["y"].min()), float(vtk["y"].max()), grid_n)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")

    # ── VTK: trilinear interpolation with scipy ───────────────────────────────
    pts_vtk = np.column_stack([XX.ravel(), YY.ravel(),
                                np.full(XX.size, z_nm)])
    phi_vtk_flat = vtk_interp(pts_vtk)
    phi_vtk = phi_vtk_flat.reshape(grid_n, grid_n)

    # ── XDMF: FEM evaluation at same physical points (metres) ────────────────
    pts_m = np.column_stack([XX.ravel() * 1e-9,
                              YY.ravel() * 1e-9,
                              np.full(XX.size, z_nm * 1e-9)])
    phi_fem_flat = eval_fem_at_points(u, pts_m)
    phi_fem = phi_fem_flat.reshape(grid_n, grid_n)

    # ── Mask: both fields must be finite ─────────────────────────────────────
    mask = np.isfinite(phi_vtk) & np.isfinite(phi_fem)
    n_valid   = int(mask.sum())
    n_total   = int(mask.size)
    coverage  = n_valid / max(n_total, 1)

    if n_valid == 0:
        print(f"    [WARN] no valid overlap — skipping")
        return {"z_req_nm": z_req_nm, "z_act_nm": z_nm, "n_valid": 0, "coverage": 0.0}

    ref = phi_vtk[mask]
    cmp = phi_fem[mask]
    diff = cmp - ref

    mae       = float(np.mean(np.abs(diff)))
    rms_abs   = float(np.sqrt(np.mean(diff**2)))
    max_abs   = float(np.max(np.abs(diff)))
    offset    = float(np.mean(diff))   # systematic bias

    denom     = np.maximum(np.abs(ref), PERCENT_FLOOR)
    pct       = 100.0 * np.abs(diff) / denom
    mean_pct  = float(np.mean(pct))
    rms_pct   = float(np.sqrt(np.mean(pct**2)))
    max_pct   = float(np.max(pct))
    p95_pct   = float(np.percentile(pct, 95))
    p99_pct   = float(np.percentile(pct, 99))

    l2_ref    = float(np.linalg.norm(ref))
    l2_rel    = float(np.linalg.norm(diff) / max(l2_ref, 1e-30))

    print(f"    coverage = {coverage:.3%}   n_valid = {n_valid}/{n_total}")
    print(f"    VTK phi  : [{np.nanmin(phi_vtk):.4f}, {np.nanmax(phi_vtk):.4f}] V")
    print(f"    FEM phi  : [{np.nanmin(phi_fem[mask]):.4f}, {np.nanmax(phi_fem[mask]):.4f}] V")
    print(f"    MAE      : {mae:.4e} V")
    print(f"    RMS abs  : {rms_abs:.4e} V")
    print(f"    max abs  : {max_abs:.4e} V")
    print(f"    offset   : {offset:.4e} V  (systematic bias)")
    print(f"    rel L2   : {l2_rel:.4f}")
    print(f"    mean %   : {mean_pct:.4f}%  ← primary metric")
    print(f"    rms  %   : {rms_pct:.4f}%")
    print(f"    max  %   : {max_pct:.4f}%")
    print(f"    p95  %   : {p95_pct:.4f}%")

    # ── 2D arrays for plotting ─────────────────────────────────────────────────
    diff_2d = np.full((grid_n, grid_n), np.nan)
    pct_2d  = np.full((grid_n, grid_n), np.nan)
    diff_2d[mask] = diff
    pct_2d[mask]  = pct

    _save_slice_plots(z_req_nm, z_nm, xg, yg, phi_vtk, phi_fem, diff_2d, pct_2d,
                      outdir, label, qw_excluded=qw_excluded)

    return {
        "label":        label,
        "z_req_nm":     z_req_nm,
        "z_act_nm":     z_nm,
        "k_vtk":        k,
        "n_valid":      n_valid,
        "coverage":     coverage,
        "vtk_min_V":    float(np.nanmin(phi_vtk)),
        "vtk_max_V":    float(np.nanmax(phi_vtk)),
        "fem_min_V":    float(np.nanmin(phi_fem[mask])),
        "fem_max_V":    float(np.nanmax(phi_fem[mask])),
        "offset_V":     offset,
        "mae_V":        mae,
        "rms_abs_V":    rms_abs,
        "max_abs_V":    max_abs,
        "l2_rel":       l2_rel,
        "mean_pct":     mean_pct,
        "rms_pct":      rms_pct,
        "max_pct":      max_pct,
        "p95_pct":      p95_pct,
        "p99_pct":      p99_pct,
        "accepted":     mean_pct < ACCEPT_PCT,
        "qw_excluded":  qw_excluded,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_slice_plots(z_req, z_act, xg, yg, phi_vtk, phi_fem, diff_2d, pct_2d,
                      outdir: Path, label: str, qw_excluded: bool = False) -> None:
    ext = [xg.min(), xg.max(), yg.min(), yg.max()]

    def _ticks(ax):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")

    excl_note = "\n⚠ QW band-offset region — excluded from verdict" if qw_excluded else ""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"{label}   z={z_req:.1f} nm → VTK z={z_act:.2f} nm{excl_note}",
                 fontsize=11, color="darkorange" if qw_excluded else "black")

    def _im(ax, data, title, cmap, vlo=None, vhi=None):
        im = ax.imshow(data.T, origin="lower", extent=ext, aspect="equal",
                       cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        _ticks(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if qw_excluded:
            ax.text(0.5, 0.97, "QW excluded", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color="darkorange",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    _im(axes[0,0], phi_vtk,  "VTK φ (V)",           "RdBu_r", PHI_VMIN, PHI_VMAX)
    _im(axes[0,1], phi_fem,  "DOLFINx FEM φ (V)",   "RdBu_r", PHI_VMIN, PHI_VMAX)
    _im(axes[1,0], diff_2d,  "FEM − VTK  (V)",      "seismic")
    _im(axes[1,1], pct_2d,   "% error  (floor=10mV)","hot_r", 0, None)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    safe_label = label.replace("/", "_")
    path = outdir / f"{safe_label}_z{z_req:.0f}nm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved: {path.name}")


def _save_lineplots(z_req_nm, vtk, vtk_interp, u, outdir, label):
    """
    Radial line cut through the gate center (x axis at y=0).
    VTK: nearest z slice directly from array.
    FEM: evaluated at matched z.
    """
    k = int(np.argmin(np.abs(vtk["z"] - z_req_nm)))
    z_nm = float(vtk["z"][k])

    rs_nm = np.linspace(-250, 250, 501)
    pts_vtk = np.column_stack([rs_nm, np.zeros_like(rs_nm), np.full_like(rs_nm, z_nm)])
    phi_vtk_line = vtk_interp(pts_vtk)

    pts_m = np.column_stack([rs_nm * 1e-9, np.zeros_like(rs_nm),
                              np.full_like(rs_nm, z_nm * 1e-9)])
    phi_fem_line = eval_fem_at_points(u, pts_m)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rs_nm, phi_vtk_line, "b-", lw=1.5, label="VTK (MaSQE)")
    ax.plot(rs_nm, phi_fem_line, "r--", lw=1.5, label="DOLFINx FEM")
    ax.set_xlabel("x (nm)  [y=0, through gate center]")
    ax.set_ylabel("φ (V)")
    ax.set_title(f"{label}  —  radial line cut at z={z_req_nm:.0f} nm")
    ax.axvline(-10, color="gray", lw=0.8, ls=":", label="disk edge ±10 nm")
    ax.axvline(+10, color="gray", lw=0.8, ls=":")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    safe_label = label.replace("/", "_")
    path = outdir / f"{safe_label}_z{z_req_nm:.0f}nm_linecut.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved: {path.name}")


# ── Summary output ────────────────────────────────────────────────────────────

def write_csv(rows, path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV: {path}")


def write_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  JSON: {path}")


def print_summary_table(all_rows):
    print("\n" + "="*90)
    print(f"  {'label':<40} {'z_req':>6}  {'coverage':>9}  {'MAE(V)':>10}  {'mean%':>8}  {'OK?':>6}")
    print("-"*90)
    for r in all_rows:
        ok = "✓ YES" if r.get("accepted") else "✗ NO"
        print(
            f"  {r.get('label','?'):<40} {r.get('z_req_nm',0):>6.1f}  "
            f"{r.get('coverage',0):>9.3%}  {r.get('mae_V',float('nan')):>10.4e}  "
            f"{r.get('mean_pct',float('nan')):>8.4f}  {ok:>6}"
        )
    print("="*90)


def plot_error_depth(all_rows, cases, outdir: Path):
    """Mean % error vs z-depth for each case."""
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (case_label, _) in enumerate(cases):
        rows = sorted([r for r in all_rows if r.get("label") == case_label],
                      key=lambda r: r.get("z_req_nm", 0))
        if not rows:
            continue
        zs   = [r["z_req_nm"] for r in rows]
        errs = [r.get("mean_pct", float("nan")) for r in rows]
        ax.plot(zs, errs, marker=markers[i % len(markers)], lw=1.5,
                label=case_label.replace("basepot_R10_", ""))

    ax.axhline(ACCEPT_PCT, color="blue", ls="--", lw=1.2, label=f"Target {ACCEPT_PCT:.0f}%")
    ax.axhline(1.0,        color="green", ls=":",  lw=1.0, label="Ideal 1%")
    ax.set_xlabel("z depth (nm)")
    ax.set_ylabel("Mean % error  (floor=10mV)")
    ax.set_title("Mean % error vs depth — VTK (MaSQE) vs DOLFINx FEM")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    path = outdir / "error_vs_depth.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Depth-error plot: {path}")


# ── Cases ─────────────────────────────────────────────────────────────────────

def default_cases(results_root: Path) -> list[tuple[str, Path]]:
    return [
        ("basepot_R10_exact_tet_h6_neg1_neg12",
         results_root / "basepot_R10_exact_tet_h6_neg1_neg12"
                      / "basepot_R10_exact_tet_h6_neg1_neg12.xdmf"),
        ("basepot_R10_exact_hex_h6_neg1_neg12",
         results_root / "basepot_R10_exact_hex_h6_neg1_neg12"
                      / "basepot_R10_exact_hex_h6_neg1_neg12.xdmf"),
        ("basepot_R10_L800_tet_h6_neg1_neg12",
         results_root / "basepot_R10_L800_tet_h6_neg1_neg12"
                      / "basepot_R10_L800_tet_h6_neg1_neg12.xdmf"),
        ("basepot_R10_L800_hex_h6_neg1_neg12",
         results_root / "basepot_R10_L800_hex_h6_neg1_neg12"
                      / "basepot_R10_L800_hex_h6_neg1_neg12.xdmf"),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run with plain python3, not mpiexec.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk",          required=True,
                    help="Path to basePotential3d.vtk")
    ap.add_argument("--results-root", default="results", dest="results_root",
                    help="Root dir containing per-case subdirs (default: results/)")
    ap.add_argument("--outdir",       default="MASQUE-Comparison/outputs")
    ap.add_argument("--grid-n",       type=int, default=201, dest="grid_n",
                    help="Points per axis on comparison grid (default 201)")
    ap.add_argument("--slices-nm",    nargs="+", type=float,
                    default=SLICES_NM, dest="slices_nm")
    ap.add_argument("--linecutz",     nargs="+", type=float,
                    default=[0.0, 155.0], dest="linecutz",
                    help="z slices for radial line-cut plots")
    ap.add_argument("--cases",        nargs="*", default=None,
                    help="Limit to specific case labels")
    ap.add_argument("--qw-exclude-nm", nargs=2, type=float, default=None,
                    metavar=("Z_LO", "Z_HI"), dest="qw_exclude_nm",
                    help="z range (nm) to mark as QW band-offset excluded "
                         "(e.g. --qw-exclude-nm 50 55). "
                         "These slices are plotted but not counted in PASS/FAIL.")
    args = ap.parse_args()

    vtk_path     = Path(args.vtk).resolve()
    results_root = Path(args.results_root).resolve()
    outdir       = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qw_exclude = tuple(args.qw_exclude_nm) if args.qw_exclude_nm else None
    if qw_exclude:
        print(f"  QW exclusion zone: z = [{qw_exclude[0]:.0f}, {qw_exclude[1]:.0f}] nm "
              f"(band-offset — excluded from PASS/FAIL verdict)")

    if not vtk_path.exists():
        print(f"ERROR: VTK not found: {vtk_path}", file=sys.stderr)
        sys.exit(1)

    # Load VTK once
    vtk        = load_vtk(vtk_path)
    vtk_interp = make_vtk_interpolator(vtk)

    # Build case list
    all_cases = default_cases(results_root)
    if args.cases:
        default_labels = {lb for lb, _ in all_cases}
        filtered = [(lb, p) for lb, p in all_cases if lb in args.cases]
        # Any name not in defaults: look up results_root/<name>/<name>.xdmf
        for name in args.cases:
            if name not in default_labels:
                xdmf = results_root / name / f"{name}.xdmf"
                filtered.append((name, xdmf))
        all_cases = filtered

    all_rows = []

    for case_label, xdmf_path in all_cases:
        xdmf_path = Path(str(xdmf_path).strip())
        if not xdmf_path.exists():
            print(f"\n[SKIP] {case_label}: {xdmf_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  CASE: {case_label}")
        print(f"{'='*60}")

        case_outdir = outdir / case_label
        case_outdir.mkdir(parents=True, exist_ok=True)

        try:
            msh, V, u = load_xdmf(xdmf_path, field_name="phi")
        except Exception as exc:
            print(f"  [ERROR] loading XDMF: {exc}")
            continue

        for z_nm in sorted(args.slices_nm):
            row = compare_at_z(z_nm, vtk, vtk_interp, u,
                               args.grid_n, case_outdir, case_label,
                               qw_exclude=qw_exclude)
            if row:
                all_rows.append(row)

        for z_nm in args.linecutz:
            _save_lineplots(z_nm, vtk, vtk_interp, u, case_outdir, case_label)

        write_csv([r for r in all_rows if r.get("label") == case_label],
                  case_outdir / f"{case_label}_metrics.csv")

    if not all_rows:
        print("\nNo results produced. Check case paths.")
        sys.exit(1)

    print_summary_table(all_rows)
    write_csv(all_rows,  outdir / "master_metrics.csv")
    write_json(all_rows, outdir / "master_metrics.json")
    plot_error_depth(all_rows, all_cases, outdir)

    # ── Acceptance verdict ──────────────────────────────────────────────────
    # Exclude QW band-offset region from verdict; use ALL non-excluded slices.
    verdict_rows = [r for r in all_rows if not r.get("qw_excluded", False)]
    qw_rows      = [r for r in all_rows if r.get("qw_excluded", False)]

    n_accepted = sum(1 for r in verdict_rows if r.get("accepted"))
    n_total_z  = len(verdict_rows)

    print(f"\n{'='*60}")
    print(f"  ACCEPTANCE (mean % < {ACCEPT_PCT:.0f}%  per slice, QW excluded)")
    if qw_rows:
        qw_zs = sorted({r['z_req_nm'] for r in qw_rows})
        print(f"  QW excluded (band-offset, not counted): z = {qw_zs} nm")
    print(f"  {n_accepted}/{n_total_z} slices passed across all cases")

    all_accepted = n_accepted == n_total_z and n_total_z > 0
    if all_accepted:
        print("  ✓  ALL NON-QW SLICES ACCEPTED")
    else:
        worst = max(verdict_rows, key=lambda r: r.get("mean_pct", 0), default={})
        print(f"  ✗  SOME SLICES NOT ACCEPTED")
        print(f"     Worst: {worst.get('label')} z={worst.get('z_req_nm')} nm "
              f"→ {worst.get('mean_pct', float('nan')):.3f}%")
    print(f"{'='*60}\n")

    write_json({
        "n_accepted":          n_accepted,
        "n_total":             n_total_z,
        "n_qw_excluded":       len(qw_rows),
        "all_accepted":        all_accepted,
        "accept_threshold_pct": ACCEPT_PCT,
        "qw_exclude_nm":       list(qw_exclude) if qw_exclude else None,
    }, outdir / "verdict.json")

    print(f"All outputs in: {outdir}/")


if __name__ == "__main__":
    main()
