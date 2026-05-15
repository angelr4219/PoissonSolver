#!/usr/bin/env python3
"""
analysis/compare_vtk_xdmf_disk10nm.py
--------------------------------------
Compare a MaSQE/Leah VTK RectilinearGrid reference against a DOLFINx
XDMF/H5 FE solution at requested z-slices.

Dependencies (all present in dolfinx/dolfinx:nightly):
  numpy, h5py, matplotlib, mpi4py, dolfinx

NO pyvista or scipy required.

For each slice the script builds a shared physical x-y grid, evaluates
both fields on it, and saves:
  - 4 PNG panels: VTK phi, XDMF phi, absolute difference, percent error
  - <basename>_comparison_summary.csv
  - <basename>_comparison_summary.json
  - README.txt

Usage via run_dolfinx.sh (mounts $PWD as /app):
  # First copy (or symlink) the VTK file into the project directory, then:
  ./run_dolfinx.sh analysis/compare_vtk_xdmf_disk10nm.py \\
      --vtk  basePotential3d.vtk \\
      --xdmf disk_R10nm_Q1e.xdmf \\
      --outdir outputs/disk10nm_comparison \\
      --slices-nm 0 5 10 20 155 255

  # Or mount the downloads folder too:
  docker run --rm -i \\
      -v "$PWD":/app -v "$HOME/Downloads/disk10nm":/vtk_data -w /app \\
      dolfinx/dolfinx:nightly \\
      sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- \\
      analysis/compare_vtk_xdmf_disk10nm.py \\
          --vtk /vtk_data/basePotential3d.vtk \\
          --xdmf disk_R10nm_Q1e.xdmf \\
          --outdir outputs/disk10nm_comparison \\
          --slices-nm 0 5 10 20 155 255
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive — works inside Docker with no display
import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from mpi4py import MPI
from dolfinx import fem, geometry, io


# ── MPI guard ──────────────────────────────────────────────────────────────

def _require_serial(comm: MPI.Intracomm) -> None:
    if comm.size != 1:
        raise RuntimeError(
            "This script supports serial execution only. "
            "Do NOT use mpiexec; just use plain python3 (or run_dolfinx.sh)."
        )


def _strip_path(p: str) -> Path:
    return Path(str(p).strip())


# ── VTK ASCII parser (no pyvista required) ─────────────────────────────────

def _read_n_floats(lines: list[str], start: int, n: int) -> tuple[np.ndarray, int]:
    """Read exactly n whitespace-separated floats starting at line index start."""
    vals: list[str] = []
    i = start
    while len(vals) < n and i < len(lines):
        vals.extend(lines[i].split())
        i += 1
    if len(vals) < n:
        raise ValueError(f"Expected {n} floats but only found {len(vals)}.")
    return np.array(vals[:n], dtype=np.float64), i


def load_vtk(vtk_path: Path, vtk_field: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Parse an ASCII VTK RectilinearGrid file.

    Returns (x, y, z, phi_3d, field_name).
    All coordinates are in the VTK file's native units (nm for basePotential3d.vtk).
    phi_3d has shape (nx, ny, nz), Fortran order matching VTK storage.
    """
    with open(vtk_path, "r") as f:
        lines = f.readlines()

    # check magic
    first = lines[0].strip() if lines else ""
    if not first.startswith("# vtk"):
        raise ValueError(f"Not a VTK file (first line: {first!r})")

    fmt_line = lines[2].strip().upper() if len(lines) > 2 else ""
    if fmt_line != "ASCII":
        raise ValueError(
            f"Only ASCII VTK files are supported (got format: {fmt_line!r}). "
            "Use pyvista for binary VTK files."
        )

    nx = ny = nz = None
    x_arr = y_arr = z_arr = None
    field_name: str | None = None
    phi_flat: np.ndarray | None = None
    found_fields: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()

        if upper.startswith("DIMENSIONS"):
            parts = line.split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])

        elif upper.startswith("X_COORDINATES"):
            n = int(line.split()[1])
            x_arr, i = _read_n_floats(lines, i + 1, n)
            continue

        elif upper.startswith("Y_COORDINATES"):
            n = int(line.split()[1])
            y_arr, i = _read_n_floats(lines, i + 1, n)
            continue

        elif upper.startswith("Z_COORDINATES"):
            n = int(line.split()[1])
            z_arr, i = _read_n_floats(lines, i + 1, n)
            continue

        elif upper.startswith("SCALARS"):
            parts = line.split()
            fn = parts[1]
            found_fields.append(fn)
            want = (vtk_field is None and phi_flat is None) or (fn == vtk_field)
            if want:
                field_name = fn
                # next non-blank line should be LOOKUP_TABLE
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].strip().upper().startswith("LOOKUP_TABLE"):
                    j += 1
                n_pts = (nx or 0) * (ny or 0) * (nz or 0)
                phi_flat, i = _read_n_floats(lines, j, n_pts)
                continue

        i += 1

    if nx is None or ny is None or nz is None:
        raise ValueError("VTK DIMENSIONS line not found.")
    if x_arr is None or y_arr is None or z_arr is None:
        raise ValueError("VTK coordinate arrays not found.")
    if phi_flat is None:
        avail = ", ".join(found_fields) or "(none)"
        raise KeyError(
            f"Field '{vtk_field}' not found. Available SCALARS: {avail}"
        )

    phi_3d = phi_flat.reshape((nx, ny, nz), order="F")
    return x_arr, y_arr, z_arr, phi_3d, field_name


def print_vtk_info(x, y, z, phi_3d, field_name: str, vtk_path: Path) -> None:
    print("\n── VTK reference ──────────────────────────────────────────────")
    print(f"  file       : {vtk_path}")
    print(f"  field      : {field_name}")
    print(f"  dimensions : {len(x)} × {len(y)} × {len(z)}")
    print(f"  x range    : [{x.min():.4g}, {x.max():.4g}]  (VTK units, likely nm)")
    print(f"  y range    : [{y.min():.4g}, {y.max():.4g}]")
    print(f"  z range    : [{z.min():.4g}, {z.max():.4g}]")
    print(f"  phi range  : [{phi_3d.min():.6g}, {phi_3d.max():.6g}]  V")


# ── trilinear interpolation (no scipy required) ────────────────────────────

def interp_rectilinear(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    phi: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """
    Vectorised trilinear interpolation on a rectilinear (possibly non-uniform) grid.

    pts : (N, 3) array of (x, y, z) query points in the same units as xs/ys/zs.
    Returns (N,) array; out-of-bounds points get NaN.
    """
    n = pts.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    ix = np.searchsorted(xs, pts[:, 0]) - 1
    iy = np.searchsorted(ys, pts[:, 1]) - 1
    iz = np.searchsorted(zs, pts[:, 2]) - 1

    valid = (
        (ix >= 0) & (ix < len(xs) - 1) &
        (iy >= 0) & (iy < len(ys) - 1) &
        (iz >= 0) & (iz < len(zs) - 1)
    )
    if not np.any(valid):
        return out

    vix = ix[valid]; viy = iy[valid]; viz = iz[valid]
    vp  = pts[valid]

    x0 = xs[vix]; x1 = xs[vix + 1]
    y0 = ys[viy]; y1 = ys[viy + 1]
    z0 = zs[viz]; z1 = zs[viz + 1]

    tx = (vp[:, 0] - x0) / (x1 - x0)
    ty = (vp[:, 1] - y0) / (y1 - y0)
    tz = (vp[:, 2] - z0) / (z1 - z0)

    out[valid] = (
        phi[vix,     viy,     viz    ] * (1-tx)*(1-ty)*(1-tz) +
        phi[vix + 1, viy,     viz    ] * tx    *(1-ty)*(1-tz) +
        phi[vix,     viy + 1, viz    ] * (1-tx)*ty    *(1-tz) +
        phi[vix + 1, viy + 1, viz    ] * tx    *ty    *(1-tz) +
        phi[vix,     viy,     viz + 1] * (1-tx)*(1-ty)*tz     +
        phi[vix + 1, viy,     viz + 1] * tx    *(1-ty)*tz     +
        phi[vix,     viy + 1, viz + 1] * (1-tx)*ty    *tz     +
        phi[vix + 1, viy + 1, viz + 1] * tx    *ty    *tz
    )
    return out


# ── XDMF / H5 loading ──────────────────────────────────────────────────────

def _parse_h5_from_xdmf(xdmf_path: Path, field_name: str) -> tuple[Path, str] | None:
    """Return (h5_file, dataset_path) from XDMF XML, or None."""
    try:
        tree = ET.parse(str(xdmf_path))
    except ET.ParseError:
        return None
    root = tree.getroot()
    for attr in root.iter():
        if attr.tag.endswith("Attribute") and attr.attrib.get("Name") == field_name:
            for child in attr.iter():
                if child.tag.endswith("DataItem") and child.text is not None:
                    text = child.text.strip()
                    if ".h5:" in text:
                        h5_name, dataset = text.split(":", 1)
                        h5_file = (xdmf_path.parent / h5_name.strip()).resolve()
                        return h5_file, dataset.strip()
    return None


def _discover_h5_dataset(h5_path: Path, field_name: str) -> str:
    """Walk H5 file to find a dataset for field_name."""
    candidates = [f"Function/{field_name}/0", f"Mesh/{field_name}", field_name]
    with h5py.File(h5_path, "r") as h5:
        for c in candidates:
            if c in h5:
                print(f"  [h5] found dataset: {c}")
                return c
        print(f"  [h5] standard paths not found; scanning top-level groups…")
        for key in h5.keys():
            grp = h5[key]
            if hasattr(grp, "keys"):
                for sub in grp.keys():
                    full = f"{key}/{sub}"
                    print(f"       {full}  shape={h5[full].shape}")
                    if field_name.lower() in sub.lower():
                        return full
    raise KeyError(
        f"Cannot locate field '{field_name}' in {h5_path}. "
        "Pass --h5-dataset explicitly, e.g. --h5-dataset Function/phi/0"
    )


def load_xdmf_function(
    xdmf_path: Path,
    h5_path: Path | None,
    field_name: str,
    h5_dataset: str | None,
) -> tuple:
    """Return (mesh, V, u, h5_path_used, dataset_used)."""
    comm = MPI.COMM_WORLD

    with io.XDMFFile(comm, str(xdmf_path), "r") as xdmf:
        msh = xdmf.read_mesh(name="mesh")

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V, name=field_name)

    if h5_path is None:
        result = _parse_h5_from_xdmf(xdmf_path, field_name)
        if result is not None:
            h5_path, inferred_ds = result
            print(f"  [xdmf] H5 inferred from XDMF XML: {h5_path}")
            if h5_dataset is None:
                h5_dataset = inferred_ds
        else:
            h5_path = xdmf_path.with_suffix(".h5")
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"Cannot find H5 file; tried {h5_path}. Pass --h5 explicitly."
                )
            print(f"  [xdmf] guessed H5: {h5_path}")

    h5_path = h5_path.resolve()

    if h5_dataset is None:
        result2 = _parse_h5_from_xdmf(xdmf_path, field_name)
        if result2 is not None:
            _, h5_dataset = result2
            print(f"  [xdmf] dataset from XDMF XML: {h5_dataset}")
        else:
            h5_dataset = _discover_h5_dataset(h5_path, field_name)

    with h5py.File(h5_path, "r") as h5:
        if h5_dataset not in h5:
            raise KeyError(f"Dataset '{h5_dataset}' not found in {h5_path}.")
        data = np.asarray(h5[h5_dataset], dtype=np.float64).reshape(-1)

    if data.size != u.x.array.size:
        raise RuntimeError(
            f"H5 dataset size {data.size} != CG1 dof count {u.x.array.size}. "
            "Check --field and --h5-dataset."
        )

    u.x.array[:] = data
    u.x.scatter_forward()
    return msh, V, u, h5_path, h5_dataset


def print_xdmf_info(msh, V, u: fem.Function, xdmf_path: Path, h5_path: Path, dataset: str, solver_scale: float) -> None:
    coords_nm = V.tabulate_dof_coordinates() * solver_scale
    phi = u.x.array
    print("\n── XDMF/H5 solver output ─────────────────────────────────────")
    print(f"  xdmf       : {xdmf_path}")
    print(f"  h5         : {h5_path}  ::  {dataset}")
    print(f"  dof count  : {phi.size}")
    print(f"  x range    : [{coords_nm[:,0].min():.4g}, {coords_nm[:,0].max():.4g}]  nm")
    print(f"  y range    : [{coords_nm[:,1].min():.4g}, {coords_nm[:,1].max():.4g}]  nm")
    print(f"  z range    : [{coords_nm[:,2].min():.4g}, {coords_nm[:,2].max():.4g}]  nm")
    print(f"  phi range  : [{phi.min():.6g}, {phi.max():.6g}]  V")
    print(f"  solver_scale: {solver_scale:.3g}  (× mesh coords → nm)")


# ── XDMF evaluation ────────────────────────────────────────────────────────

def eval_xdmf_at_points(u: fem.Function, pts_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate DOLFINx function at physical points (meters). Returns (values, inside)."""
    msh = u.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts_m)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts_m)

    n = pts_m.shape[0]
    inside = np.zeros(n, dtype=bool)
    cells = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        links = colliding.links(i)
        if len(links) > 0:
            inside[i] = True
            cells[i] = links[0]

    values = np.full(n, np.nan, dtype=np.float64)
    if np.any(inside):
        vals = u.eval(pts_m[inside], cells[inside])
        values[inside] = np.asarray(vals).reshape(-1)

    return values, inside


# ── per-slice comparison ───────────────────────────────────────────────────

def compare_slice(
    z_nm: float,
    vtk_x: np.ndarray,
    vtk_y: np.ndarray,
    vtk_z: np.ndarray,
    vtk_phi: np.ndarray,
    u: fem.Function,
    solver_scale: float,
    grid_n: int,
    percent_floor: float,
    outdir: Path,
    basename: str,
    save_npy: bool,
) -> dict | None:
    vtk_zmin, vtk_zmax = float(vtk_z.min()), float(vtk_z.max())

    if z_nm < vtk_zmin or z_nm > vtk_zmax:
        print(
            f"  [skip] z={z_nm:.1f} nm is outside VTK z range "
            f"[{vtk_zmin:.1f}, {vtk_zmax:.1f}] nm."
        )
        return None

    # nearest valid VTK z — never crashes regardless of requested value
    k_vtk = int(np.argmin(np.abs(vtk_z - z_nm)))
    z_actual_nm = float(vtk_z[k_vtk])

    print(
        f"\n  z_requested={z_nm:.1f} nm  →  nearest VTK z={z_actual_nm:.4f} nm  "
        f"(k={k_vtk}/{len(vtk_z)-1})"
    )

    # shared physical x-y grid (nm)
    gx = np.linspace(float(vtk_x.min()), float(vtk_x.max()), grid_n)
    gy = np.linspace(float(vtk_y.min()), float(vtk_y.max()), grid_n)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    GZ = np.full_like(GX, z_actual_nm)
    flat_nm = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)

    # evaluate VTK (trilinear, nm)
    phi_vtk_flat = interp_rectilinear(vtk_x, vtk_y, vtk_z, vtk_phi, flat_nm)
    phi_vtk = phi_vtk_flat.reshape(grid_n, grid_n)

    # evaluate solver (convert nm → solver units before querying)
    flat_solver = flat_nm * 1e-9 / solver_scale
    phi_fe_flat, inside_mask = eval_xdmf_at_points(u, flat_solver)
    phi_fe = phi_fe_flat.reshape(grid_n, grid_n)
    inside_2d = inside_mask.reshape(grid_n, grid_n)

    valid = np.isfinite(phi_vtk) & np.isfinite(phi_fe) & inside_2d
    n_valid = int(valid.sum())
    coverage = float(n_valid / max(valid.size, 1))

    if n_valid == 0:
        print(f"    [warn] no valid overlap — skipping plots.")
        return {"z_requested_nm": z_nm, "z_actual_nm": z_actual_nm, "k_vtk": k_vtk,
                "n_valid": 0, "coverage": 0.0}

    ref = phi_vtk[valid]
    cmp = phi_fe[valid]
    abs_err = np.abs(cmp - ref)
    denom   = np.maximum(np.abs(ref), percent_floor)
    pct_err = 100.0 * abs_err / denom

    mae      = float(np.mean(abs_err))
    rms_abs  = float(np.sqrt(np.mean(abs_err**2)))
    max_abs  = float(np.max(abs_err))
    mean_pct = float(np.mean(pct_err))
    rms_pct  = float(np.sqrt(np.mean(pct_err**2)))
    max_pct  = float(np.max(pct_err))
    ref_norm = float(np.linalg.norm(ref))
    g_rel_l2 = float(np.linalg.norm(cmp - ref) / max(ref_norm, percent_floor))

    print(
        f"    coverage={coverage:.3%}  mae={mae:.4e} V  rms={rms_abs:.4e} V  "
        f"mean_pct={mean_pct:.3f}%"
    )

    abs_2d = np.full((grid_n, grid_n), np.nan)
    pct_2d = np.full((grid_n, grid_n), np.nan)
    abs_2d[valid] = abs_err
    pct_2d[valid] = pct_err

    _save_slice_plots(z_nm, z_actual_nm, gx, gy, phi_vtk, phi_fe, abs_2d, pct_2d, outdir, basename)

    if save_npy:
        tag = f"{basename}_z{z_nm:.0f}nm"
        np.save(outdir / f"{tag}_phi_vtk.npy", phi_vtk)
        np.save(outdir / f"{tag}_phi_fe.npy",  phi_fe)
        np.save(outdir / f"{tag}_abs_err.npy", abs_2d)
        np.save(outdir / f"{tag}_pct_err.npy", pct_2d)

    return {
        "z_requested_nm":  z_nm,
        "z_actual_nm":     z_actual_nm,
        "k_vtk":           k_vtk,
        "n_valid":         n_valid,
        "n_total":         int(valid.size),
        "coverage":        coverage,
        "mae_V":           mae,
        "rms_abs_V":       rms_abs,
        "max_abs_V":       max_abs,
        "mean_pct_err":    mean_pct,
        "rms_pct_err":     rms_pct,
        "max_pct_err":     max_pct,
        "global_rel_l2":   g_rel_l2,
        "phi_vtk_min":     float(np.nanmin(phi_vtk)),
        "phi_vtk_max":     float(np.nanmax(phi_vtk)),
        "phi_fe_min":      float(np.nanmin(phi_fe[inside_2d])) if np.any(inside_2d) else float("nan"),
        "phi_fe_max":      float(np.nanmax(phi_fe[inside_2d])) if np.any(inside_2d) else float("nan"),
    }


# ── plotting ───────────────────────────────────────────────────────────────

def _save_slice_plots(
    z_req: float, z_act: float,
    gx: np.ndarray, gy: np.ndarray,
    phi_vtk: np.ndarray, phi_fe: np.ndarray,
    abs_err: np.ndarray, pct_err: np.ndarray,
    outdir: Path, basename: str,
) -> None:
    extent = [gx.min(), gx.max(), gy.min(), gy.max()]
    vmin = float(np.nanmin([phi_vtk.min(), np.nanmin(phi_fe)]))
    vmax = float(np.nanmax([phi_vtk.max(), np.nanmax(phi_fe)]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"z = {z_req:.1f} nm requested  →  VTK z = {z_act:.4f} nm",
        fontsize=12,
    )

    def _im(ax, data, title, cmap, vlo=None, vhi=None):
        im = ax.imshow(
            data.T, origin="lower", extent=extent,
            aspect="equal", cmap=cmap, vmin=vlo, vmax=vhi,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _im(axes[0, 0], phi_vtk, "VTK reference  φ (V)",     "RdBu_r", vmin, vmax)
    _im(axes[0, 1], phi_fe,  "XDMF solver   φ (V)",      "RdBu_r", vmin, vmax)
    _im(axes[1, 0], abs_err, "|φ_FE − φ_VTK|  (V)",      "hot_r")
    _im(axes[1, 1], pct_err, "% error  (floor applied)", "hot_r")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = outdir / f"{basename}_z{z_req:.0f}nm_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved: {path}")


# ── summary output ─────────────────────────────────────────────────────────

def write_summary_csv(rows: list[dict], outdir: Path, basename: str) -> Path:
    path = outdir / f"{basename}_comparison_summary.csv"
    if not rows:
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def write_summary_json(rows: list[dict], outdir: Path, basename: str) -> Path:
    path = outdir / f"{basename}_comparison_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return path


def write_readme(args: argparse.Namespace, rows: list[dict], outdir: Path, basename: str) -> Path:
    path = outdir / "README.txt"
    txt = textwrap.dedent(f"""\
        VTK vs XDMF Comparison — {basename}
        Generated by: analysis/compare_vtk_xdmf_disk10nm.py
        ============================================================
        VTK reference  : {args.vtk}
        VTK field      : {args.vtk_field or '(auto-detected)'}
        XDMF solver    : {args.xdmf}
        H5 file        : {args.h5 or '(inferred from XDMF)'}
        H5 dataset     : {args.h5_dataset or '(auto-discovered)'}
        Solver field   : {args.field}
        Solver scale   : {args.solver_scale}  (multiply DOLFINx coords → nm)
        Grid resolution: {args.grid_n} × {args.grid_n}
        Percent floor  : {args.percent_floor}
        Requested z    : {args.slices_nm} nm

        Slices processed: {len(rows)}
        ============================================================
        Per-slice results:
    """)
    for r in rows:
        z_req = r.get("z_requested_nm", "?")
        cov   = r.get("coverage", float("nan"))
        mae   = r.get("mae_V", float("nan"))
        rms   = r.get("rms_abs_V", float("nan"))
        pct   = r.get("mean_pct_err", float("nan"))
        txt += (
            f"  z={z_req:.1f} nm  coverage={cov:.2%}  "
            f"mae={mae:.4e} V  rms={rms:.4e} V  mean_pct={pct:.3f}%\n"
        )
    path.write_text(txt, encoding="utf-8")
    return path


def print_final_table(rows: list[dict]) -> None:
    print("\n" + "=" * 76)
    print(f"  {'z_req':>8}  {'z_act':>9}  {'coverage':>9}  {'MAE (V)':>12}  {'RMS (V)':>12}  {'mean_%':>8}")
    print("-" * 76)
    for r in rows:
        print(
            f"  {r.get('z_requested_nm', 0):>8.1f}  "
            f"{r.get('z_actual_nm', 0):>9.4f}  "
            f"{r.get('coverage', 0):>9.3%}  "
            f"{r.get('mae_V', float('nan')):>12.4e}  "
            f"{r.get('rms_abs_V', float('nan')):>12.4e}  "
            f"{r.get('mean_pct_err', float('nan')):>8.3f}"
        )
    print("=" * 76)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare MaSQE VTK reference vs DOLFINx XDMF/H5 at z-slices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--vtk",        required=True, help="ASCII VTK RectilinearGrid (basePotential3d.vtk)")
    ap.add_argument("--xdmf",       required=True, help="DOLFINx XDMF file (.xdmf)")
    ap.add_argument("--h5",         default=None,  help="Paired H5 file (inferred if omitted)")
    ap.add_argument("--field",      default="phi", help="Field name in XDMF/H5 (default: phi)")
    ap.add_argument("--h5-dataset", default=None,  dest="h5_dataset",
                    help="Explicit H5 dataset, e.g. Function/phi/0 (auto-found if omitted)")
    ap.add_argument("--vtk-field",  default=None,  dest="vtk_field",
                    help="SCALARS field name in VTK (auto-detected if omitted)")
    ap.add_argument("--outdir",     default="outputs/disk10nm_comparison")
    ap.add_argument("--basename",   default="disk10nm",
                    help="Prefix for output files (default: disk10nm)")
    ap.add_argument("--slices-nm",  nargs="+", type=float,
                    default=[0, 5, 10, 20, 155, 255], dest="slices_nm",
                    help="z values in nm to compare (default: 0 5 10 20 155 255)")
    ap.add_argument("--grid-n",     type=int, default=201, dest="grid_n",
                    help="Grid points per axis for shared x-y comparison grid (default: 201)")
    ap.add_argument("--solver-scale", type=float, default=1e9, dest="solver_scale",
                    help="Multiply DOLFINx coords by this to get nm (default: 1e9, i.e. solver uses metres)")
    ap.add_argument("--percent-floor", type=float, default=1e-12, dest="percent_floor",
                    help="Floor on |reference| for percent error denominator (default: 1e-12)")
    ap.add_argument("--save-npy",   action="store_true", dest="save_npy",
                    help="Also save .npy arrays for each slice")
    return ap.parse_args()


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    _require_serial(MPI.COMM_WORLD)
    args = parse_args()

    vtk_path  = _strip_path(args.vtk)
    xdmf_path = _strip_path(args.xdmf)
    h5_path   = _strip_path(args.h5) if args.h5 else None
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading VTK reference: {vtk_path}")
    vtk_x, vtk_y, vtk_z, vtk_phi, vtk_field_name = load_vtk(vtk_path, args.vtk_field)
    print_vtk_info(vtk_x, vtk_y, vtk_z, vtk_phi, vtk_field_name, vtk_path)

    print(f"\nLoading XDMF solver output: {xdmf_path}")
    msh, V, u, h5_used, ds_used = load_xdmf_function(
        xdmf_path, h5_path, args.field, args.h5_dataset
    )
    print_xdmf_info(msh, V, u, xdmf_path, h5_used, ds_used, args.solver_scale)

    print(f"\nComparing {len(args.slices_nm)} z-slice(s)…")
    rows: list[dict] = []
    for z_nm in sorted(args.slices_nm):
        result = compare_slice(
            z_nm=z_nm,
            vtk_x=vtk_x, vtk_y=vtk_y, vtk_z=vtk_z, vtk_phi=vtk_phi,
            u=u,
            solver_scale=args.solver_scale,
            grid_n=args.grid_n,
            percent_floor=args.percent_floor,
            outdir=outdir,
            basename=args.basename,
            save_npy=args.save_npy,
        )
        if result is not None:
            rows.append(result)

    csv_path  = write_summary_csv(rows, outdir, args.basename)
    json_path = write_summary_json(rows, outdir, args.basename)
    readme    = write_readme(args, rows, outdir, args.basename)

    print_final_table(rows)
    print(f"\nAll outputs in: {outdir}/")
    print(f"  CSV    : {csv_path}")
    print(f"  JSON   : {json_path}")
    print(f"  README : {readme}")


if __name__ == "__main__":
    main()
