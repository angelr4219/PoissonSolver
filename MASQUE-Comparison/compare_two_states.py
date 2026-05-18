#!/usr/bin/env python3
"""
MASQUE-Comparison/compare_two_states.py
-----------------------------------------
Side-by-side comparison of two physical states:

  State 0 — background (no gate):
    DOLFINx: sigma=-2e11 cm⁻², V_bottom=-12 V, no disk BC
    MaSQE:   basePotential3d.vtk

  State 1 — gate active:
    DOLFINx: sigma=-2e11 cm⁻², V_bottom=-12 V, V_gate=-1 V (disk R=10 nm)
    MaSQE:   totalPotential3d.vtk

For each z-slice this script produces:

  Panel A — absolute phi:
    [MaSQE bg | FEM bg | MaSQE gate | FEM gate]
    Shared colour scale so background vs gate is directly visible.

  Panel B — gate perturbation  Δφ = φ_gate − φ_bg:
    [MaSQE Δφ | FEM Δφ | FEM − MaSQE Δφ error | % error]
    Shows how the disk voltage decays with depth and whether both codes agree.

  Panel C — depth profile (1D):
    phi(r=0, z) from z=0 to bottom for both states and both codes.
    Plus Δφ(z) at the disk centre.

QW note: z=50-55 nm may show mismatch if MaSQE VTKs include band-edge offsets.
Those slices are plotted but annotated and excluded from the verdict.

Run via:
  ./MASQUE-Comparison/run_two_states.sh
"""
from __future__ import annotations

import argparse
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

# ── Constants ─────────────────────────────────────────────────────────────────
R_DISK_NM   = 10.0
V_GATE      = -1.0
V_BG        = -12.0
PHI_VMIN    = -12.0
PHI_VMAX    = -1.0
PCT_FLOOR   = 0.01   # 10 mV denominator floor
ACCEPT_PCT  = 5.0


# ── Shared VTK / FEM I/O (same as compare_masque.py) ─────────────────────────

def _read_n_floats(lines, start, n):
    vals = []
    i = start
    while len(vals) < n and i < len(lines):
        vals.extend(lines[i].split())
        i += 1
    return np.array(vals[:n], dtype=np.float64), i


def load_vtk(vtk_path: Path) -> dict:
    with open(vtk_path) as f:
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
        line = lines[i].strip(); up = line.upper()
        if up.startswith("DIMENSIONS"):
            p = line.split(); nx, ny, nz = int(p[1]), int(p[2]), int(p[3])
        elif up.startswith("X_COORDINATES"):
            x_arr, i = _read_n_floats(lines, i + 1, int(line.split()[1])); continue
        elif up.startswith("Y_COORDINATES"):
            y_arr, i = _read_n_floats(lines, i + 1, int(line.split()[1])); continue
        elif up.startswith("Z_COORDINATES"):
            z_arr, i = _read_n_floats(lines, i + 1, int(line.split()[1])); continue
        elif up.startswith("SCALARS") and phi_flat is None:
            field_name = line.split()[1]
            j = i + 1
            while j < len(lines) and not lines[j].strip(): j += 1
            if lines[j].strip().upper().startswith("LOOKUP_TABLE"): j += 1
            phi_flat, i = _read_n_floats(lines, j, nx * ny * nz); continue
        i += 1
    if phi_flat is None:
        raise ValueError(f"Could not parse VTK field from {vtk_path}")
    phi_3d = phi_flat.reshape((nx, ny, nz), order="F")
    print(f"  VTK {vtk_path.name}: {field_name} [{phi_3d.min():.4f}, {phi_3d.max():.4f}] V  "
          f"({nx}×{ny}×{nz})")
    return dict(x=x_arr, y=y_arr, z=z_arr, phi=phi_3d, field_name=field_name)


def make_interp(vtk: dict) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (vtk["x"], vtk["y"], vtk["z"]), vtk["phi"],
        method="linear", bounds_error=False, fill_value=np.nan,
    )


def _parse_xdmf_dataset(xdmf_path: Path, field_name: str):
    tree = ET.parse(str(xdmf_path)); root = tree.getroot()
    for attr in root.iter():
        if attr.tag.endswith("Attribute") and attr.attrib.get("Name") == field_name:
            for child in attr.iter():
                if child.tag.endswith("DataItem") and child.text:
                    text = child.text.strip()
                    if ".h5:" in text:
                        h5_name, dset = text.split(":", 1)
                        return (xdmf_path.parent / h5_name.strip()).resolve(), dset.strip()
    raise KeyError(f"Field '{field_name}' not found in {xdmf_path}")


def load_xdmf(xdmf_path: Path) -> tuple:
    comm = MPI.COMM_WORLD
    h5_file, dataset = _parse_xdmf_dataset(xdmf_path, "phi")
    with io.XDMFFile(comm, str(xdmf_path.resolve()), "r") as xf:
        msh = xf.read_mesh(name="mesh")
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V, name="phi")
    with h5py.File(str(h5_file), "r") as h5:
        data = np.asarray(h5[dataset], dtype=np.float64).reshape(-1)
    if data.size != u.x.array.size:
        raise RuntimeError(f"DOF mismatch: {data.size} vs {u.x.array.size}")
    u.x.array[:] = data; u.x.scatter_forward()
    phi = u.x.array
    print(f"  FEM {xdmf_path.name}: phi [{phi.min():.4f}, {phi.max():.4f}] V  "
          f"({phi.size} dofs)")
    return msh, V, u


def eval_fem_at_points(u: fem.Function, pts_m: np.ndarray) -> np.ndarray:
    msh = u.function_space.mesh
    tree = bb_tree(msh, msh.topology.dim)
    cand = compute_collisions_points(tree, pts_m)
    coll = compute_colliding_cells(msh, cand, pts_m)
    n = pts_m.shape[0]
    inside = np.zeros(n, dtype=bool)
    cells  = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        links = coll.links(i)
        if len(links): inside[i] = True; cells[i] = links[0]
    vals = np.full(n, np.nan)
    if np.any(inside):
        vals[inside] = np.asarray(u.eval(pts_m[inside], cells[inside])).reshape(-1)
    return vals


# ── Per-slice evaluation ───────────────────────────────────────────────────────

def eval_slice(vtk: dict, interp: RegularGridInterpolator,
               u: fem.Function, z_nm: float, grid_n: int) -> dict:
    """Return dict with phi_vtk, phi_fem (grid_n×grid_n), xg, yg in nm."""
    k = int(np.argmin(np.abs(vtk["z"] - z_nm)))
    z_act = float(vtk["z"][k])

    xg = np.linspace(float(vtk["x"].min()), float(vtk["x"].max()), grid_n)
    yg = np.linspace(float(vtk["y"].min()), float(vtk["y"].max()), grid_n)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")

    pts_vtk = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z_act)])
    phi_vtk = interp(pts_vtk).reshape(grid_n, grid_n)

    pts_m = np.column_stack([XX.ravel() * 1e-9, YY.ravel() * 1e-9,
                              np.full(XX.size, z_act * 1e-9)])
    phi_fem = eval_fem_at_points(u, pts_m).reshape(grid_n, grid_n)

    return dict(xg=xg, yg=yg, z_act=z_act, phi_vtk=phi_vtk, phi_fem=phi_fem)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _ticks(ax):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.set_xlabel("x (nm)", fontsize=7); ax.set_ylabel("y (nm)", fontsize=7)
    ax.tick_params(labelsize=6)


def _im(fig, ax, data, title, cmap, vlo=None, vhi=None, label=""):
    ext = [data["xg"].min(), data["xg"].max(), data["yg"].min(), data["yg"].max()]
    arr = data["arr"]
    im = ax.imshow(arr.T, origin="lower", extent=ext, aspect="equal",
                   cmap=cmap, vmin=vlo, vmax=vhi, interpolation="bilinear")
    ax.set_title(title, fontsize=8)
    _ticks(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    if label:
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


def _disk_circle(ax, cx_nm, cy_nm):
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(cx_nm + R_DISK_NM * np.cos(theta),
            cy_nm + R_DISK_NM * np.sin(theta),
            "k--", lw=0.8, alpha=0.6)


# ── Panel A: absolute phi (4 panels) ──────────────────────────────────────────

def plot_panel_A(z_req, bg, gate, outdir, label, cx_nm, cy_nm, qw_excluded):
    qw_tag = "  ⚠ QW excluded" if qw_excluded else ""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"φ absolute   z = {z_req:.0f} nm{qw_tag}\n"
        f"Left pair: background (no gate)   Right pair: gate active (V_gate={V_GATE} V)",
        fontsize=10, color="darkorange" if qw_excluded else "black"
    )

    vlo, vhi = PHI_VMIN, PHI_VMAX
    _im(fig, axes[0], dict(arr=bg["phi_vtk"],  xg=bg["xg"], yg=bg["yg"]),
        "MaSQE — background", "RdBu_r", vlo, vhi,
        label=f"[{np.nanmin(bg['phi_vtk']):.3f}, {np.nanmax(bg['phi_vtk']):.3f}] V")
    _im(fig, axes[1], dict(arr=bg["phi_fem"],  xg=bg["xg"], yg=bg["yg"]),
        "DOLFINx — background", "RdBu_r", vlo, vhi,
        label=f"[{np.nanmin(bg['phi_fem']):.3f}, {np.nanmax(bg['phi_fem']):.3f}] V")
    _im(fig, axes[2], dict(arr=gate["phi_vtk"], xg=gate["xg"], yg=gate["yg"]),
        "MaSQE — gate active", "RdBu_r", vlo, vhi,
        label=f"[{np.nanmin(gate['phi_vtk']):.3f}, {np.nanmax(gate['phi_vtk']):.3f}] V")
    _im(fig, axes[3], dict(arr=gate["phi_fem"], xg=gate["xg"], yg=gate["yg"]),
        "DOLFINx — gate active", "RdBu_r", vlo, vhi,
        label=f"[{np.nanmin(gate['phi_fem']):.3f}, {np.nanmax(gate['phi_fem']):.3f}] V")

    for ax in axes:
        _disk_circle(ax, cx_nm, cy_nm)

    fig.tight_layout()
    path = outdir / f"{label}_z{z_req:.0f}nm_A_phi.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    A: {path.name}")


# ── Panel B: gate perturbation Δφ ─────────────────────────────────────────────

def plot_panel_B(z_req, bg, gate, outdir, label, cx_nm, cy_nm, qw_excluded):
    xg, yg = bg["xg"], bg["yg"]

    delta_masque = gate["phi_vtk"] - bg["phi_vtk"]
    delta_fem    = gate["phi_fem"] - bg["phi_fem"]

    mask = np.isfinite(delta_masque) & np.isfinite(delta_fem)
    diff = delta_fem - delta_masque
    denom = np.maximum(np.abs(delta_masque), PCT_FLOOR)
    pct  = np.where(mask, 100.0 * np.abs(diff) / denom, np.nan)

    mean_pct = float(np.nanmean(pct[mask])) if mask.any() else float("nan")
    max_pct  = float(np.nanmax(pct[mask]))  if mask.any() else float("nan")

    # Symmetric colour limits around zero for delta
    dmax = max(np.nanmax(np.abs(delta_masque)), np.nanmax(np.abs(delta_fem)), 0.01)
    dmax = float(np.ceil(dmax * 10) / 10)

    qw_tag = "  ⚠ QW excluded" if qw_excluded else ""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Δφ = φ_gate − φ_bg   z = {z_req:.0f} nm{qw_tag}\n"
        f"Gate perturbation: how the -1 V disk decays with depth",
        fontsize=10, color="darkorange" if qw_excluded else "black"
    )

    _im(fig, axes[0], dict(arr=delta_masque, xg=xg, yg=yg),
        "MaSQE Δφ (V)", "seismic", -dmax, dmax,
        label=f"peak {np.nanmax(np.abs(delta_masque)):.3f} V")
    _im(fig, axes[1], dict(arr=delta_fem, xg=xg, yg=yg),
        "DOLFINx Δφ (V)", "seismic", -dmax, dmax,
        label=f"peak {np.nanmax(np.abs(delta_fem)):.3f} V")
    _im(fig, axes[2], dict(arr=np.where(mask, diff, np.nan), xg=xg, yg=yg),
        "FEM − MaSQE Δφ error (V)", "seismic")
    _im(fig, axes[3], dict(arr=pct, xg=xg, yg=yg),
        f"% error on Δφ   mean={mean_pct:.2f}%", "hot_r", 0, None)

    for ax in axes:
        _disk_circle(ax, cx_nm, cy_nm)

    fig.tight_layout()
    path = outdir / f"{label}_z{z_req:.0f}nm_B_delta.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    B: {path.name}")

    return {"mean_pct_delta": mean_pct, "max_pct_delta": max_pct,
            "qw_excluded": qw_excluded}


# ── Panel C: depth profile through disk centre ────────────────────────────────

def plot_panel_C(vtk_bg, vtk_gate, interp_bg, interp_gate,
                 u_bg, u_gate, outdir, label,
                 cx_nm, cy_nm, z_lo_nm, z_hi_nm, qw_exclude):
    z_vtk = vtk_bg["z"]
    z_mask = (z_vtk >= z_lo_nm) & (z_vtk <= z_hi_nm)
    zz = z_vtk[z_mask]

    # VTK: interpolate at disk centre vs depth
    pts_bg   = np.column_stack([np.full_like(zz, cx_nm),
                                 np.full_like(zz, cy_nm), zz])
    pts_gate = pts_bg.copy()
    phi_vtk_bg   = interp_bg(pts_bg)
    phi_vtk_gate = interp_gate(pts_gate)
    delta_vtk = phi_vtk_gate - phi_vtk_bg

    # FEM: evaluate at same z values
    pts_m = np.column_stack([np.full_like(zz, cx_nm * 1e-9),
                              np.full_like(zz, cy_nm * 1e-9),
                              zz * 1e-9])
    phi_fem_bg   = eval_fem_at_points(u_bg,   pts_m)
    phi_fem_gate = eval_fem_at_points(u_gate, pts_m)
    delta_fem    = phi_fem_gate - phi_fem_bg

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Depth profile at disk centre  (x={cx_nm:.0f}, y={cy_nm:.0f} nm)\n"
        f"Background vs gate-active — how disk voltage decays into semiconductor",
        fontsize=10
    )

    # ── left: absolute phi vs depth ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(zz, phi_vtk_bg,   "b-",  lw=1.5, label="MaSQE bg")
    ax.plot(zz, phi_vtk_gate, "b--", lw=1.5, label="MaSQE gate")
    ax.plot(zz, phi_fem_bg,   "r-",  lw=1.5, label="DOLFINx bg")
    ax.plot(zz, phi_fem_gate, "r--", lw=1.5, label="DOLFINx gate")
    if qw_exclude:
        ax.axvspan(qw_exclude[0], qw_exclude[1], alpha=0.15, color="orange",
                   label="QW band-offset zone")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("φ (V)")
    ax.set_title("φ absolute vs depth")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── middle: Δφ vs depth ───────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(zz, delta_vtk * 1e3, "b-",  lw=2, label="MaSQE Δφ")
    ax.plot(zz, delta_fem * 1e3, "r--", lw=2, label="DOLFINx Δφ")
    if qw_exclude:
        ax.axvspan(qw_exclude[0], qw_exclude[1], alpha=0.15, color="orange",
                   label="QW zone")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Δφ (mV)")
    ax.set_title("Gate perturbation Δφ(z) at disk centre\n(how -1V disk decays into semiconductor)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── right: error on Δφ vs depth ───────────────────────────────────────────
    ax = axes[2]
    valid = np.isfinite(delta_vtk) & np.isfinite(delta_fem) & (np.abs(delta_vtk) > PCT_FLOOR)
    pct_line = np.where(valid, 100.0 * np.abs(delta_fem - delta_vtk) / np.abs(delta_vtk), np.nan)
    ax.plot(zz, pct_line, "k-", lw=1.5, label="% error on Δφ")
    if qw_exclude:
        ax.axvspan(qw_exclude[0], qw_exclude[1], alpha=0.15, color="orange",
                   label="QW excluded")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("% error")
    ax.set_title("% error on gate perturbation Δφ vs depth")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"{label}_depth_profile.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    C: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run with plain python3 (not mpiexec).")

    ap = argparse.ArgumentParser(description="Two-state comparison: background vs gate-active.")
    ap.add_argument("--vtk-bg",    required=True,  help="basePotential3d.vtk (MaSQE background)")
    ap.add_argument("--vtk-gate",  required=True,  help="totalPotential3d.vtk (MaSQE gate active)")
    ap.add_argument("--xdmf-bg",   required=True,  help="DOLFINx XDMF: sigma only, no gate")
    ap.add_argument("--xdmf-gate", required=True,  help="DOLFINx XDMF: sigma + gate active")
    ap.add_argument("--outdir",    default="MASQUE-Comparison/outputs_two_states")
    ap.add_argument("--grid-n",    type=int, default=401, dest="grid_n",
                    help="Grid points per axis for 2D comparison (default 401)")
    ap.add_argument("--slices-nm", type=float, nargs="+", dest="slices_nm",
                    default=[0, 10, 20, 50, 55, 100, 155],
                    help="z depths (nm) for 2D slice panels (default: 0 10 20 50 55 100 155)")
    ap.add_argument("--qw-exclude-nm", nargs=2, type=float, default=[50.0, 55.0],
                    metavar=("Z_LO", "Z_HI"), dest="qw_exclude_nm",
                    help="QW band-offset exclusion zone in nm (default: 50 55)")
    ap.add_argument("--disk-cx-nm", type=float, default=250.0, dest="cx_nm")
    ap.add_argument("--disk-cy-nm", type=float, default=250.0, dest="cy_nm")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qw_exclude = tuple(args.qw_exclude_nm)

    print("\n=== Two-state comparison: background vs gate-active ===")
    print(f"  VTK bg   : {args.vtk_bg}")
    print(f"  VTK gate : {args.vtk_gate}")
    print(f"  FEM bg   : {args.xdmf_bg}")
    print(f"  FEM gate : {args.xdmf_gate}")
    print(f"  QW excluded: z=[{qw_exclude[0]:.0f}, {qw_exclude[1]:.0f}] nm")
    print(f"  grid-n   : {args.grid_n}")

    # ── Load everything ───────────────────────────────────────────────────────
    print("\n── Loading MaSQE VTKs ──")
    vtk_bg   = load_vtk(Path(args.vtk_bg))
    vtk_gate = load_vtk(Path(args.vtk_gate))
    interp_bg   = make_interp(vtk_bg)
    interp_gate = make_interp(vtk_gate)

    print("\n── Loading DOLFINx FEM ──")
    _, _, u_bg   = load_xdmf(Path(args.xdmf_bg))
    _, _, u_gate = load_xdmf(Path(args.xdmf_gate))

    # ── Per-slice panels A & B ────────────────────────────────────────────────
    label   = "two_states"
    results = []

    print("\n── Slice panels ──")
    for z_req in sorted(args.slices_nm):
        qw_excluded = qw_exclude[0] <= z_req <= qw_exclude[1]
        tag = " [QW excl]" if qw_excluded else ""
        print(f"\n  z = {z_req:.0f} nm{tag}")

        bg_slice   = eval_slice(vtk_bg,   interp_bg,   u_bg,   z_req, args.grid_n)
        gate_slice = eval_slice(vtk_gate, interp_gate, u_gate, z_req, args.grid_n)

        plot_panel_A(z_req, bg_slice, gate_slice, outdir, label,
                     args.cx_nm, args.cy_nm, qw_excluded)
        row = plot_panel_B(z_req, bg_slice, gate_slice, outdir, label,
                           args.cx_nm, args.cy_nm, qw_excluded)
        row["z_nm"] = z_req
        results.append(row)

    # ── Panel C: depth profile ─────────────────────────────────────────────────
    print("\n── Depth profile ──")
    coords = u_bg.function_space.mesh.geometry.x
    z_lo_nm = float(coords[:, 2].min()) * 1e9
    z_hi_nm = float(coords[:, 2].max()) * 1e9
    plot_panel_C(vtk_bg, vtk_gate, interp_bg, interp_gate,
                 u_bg, u_gate, outdir, label,
                 args.cx_nm, args.cy_nm,
                 z_lo_nm, z_hi_nm, qw_exclude)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict_rows = [r for r in results if not r["qw_excluded"]]
    qw_rows      = [r for r in results if r["qw_excluded"]]
    n_pass = sum(1 for r in verdict_rows if r["mean_pct_delta"] < ACCEPT_PCT)
    n_tot  = len(verdict_rows)

    print(f"\n{'='*60}")
    print(f"  VERDICT — Δφ agreement  (mean % < {ACCEPT_PCT:.0f}%)")
    if qw_rows:
        print(f"  QW excluded: z = {[r['z_nm'] for r in qw_rows]} nm")
    print(f"  {n_pass}/{n_tot} non-QW slices passed")
    for r in verdict_rows:
        tag = "✓" if r["mean_pct_delta"] < ACCEPT_PCT else "✗"
        print(f"    {tag}  z={r['z_nm']:5.0f} nm :  mean={r['mean_pct_delta']:.3f}%  "
              f"max={r['max_pct_delta']:.3f}%")
    print(f"{'='*60}\n")

    verdict = {
        "n_pass": n_pass, "n_total": n_tot,
        "n_qw_excluded": len(qw_rows),
        "all_passed": n_pass == n_tot and n_tot > 0,
        "accept_threshold_pct": ACCEPT_PCT,
        "qw_exclude_nm": list(qw_exclude),
        "slices": results,
    }
    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"Output: {outdir}/")


if __name__ == "__main__":
    main()
