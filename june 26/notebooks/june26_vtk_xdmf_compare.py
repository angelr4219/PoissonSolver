# %% [markdown]
# # June 26 — MaSQE VTK vs DOLFINx XDMF/H5 common-grid comparison
#
# **Workflow**
# 1. Read MaSQE `basePotential3d.vtk` (ASCII rectilinear grid, nm, −12 to −1 V).
# 2. Read each DOLFINx XDMF/H5 FEM case.
# 3. Build a common physical x-y grid in nm at each requested z-slice.
# 4. Evaluate both fields on that grid (trilinear for VTK, bb-tree for FEM).
# 5. Compute signed difference (FEM − VTK), absolute error, percent error.
# 6. Save PNG panels, line cuts, histograms, and per-slice CSV.gz files.
# 7. Write a summary CSV ranking all cases.
#
# **Unit convention**
# - VTK coordinates: nm (confirmed from file: x/y ±250 nm, z 0–255 nm).
# - DOLFINx mesh coordinates: metres (multiply by 1e9 to convert to nm).
#
# **Run inside Docker**:
# ```
# ./run_dolfinx.sh "june 26/notebooks/june26_vtk_xdmf_compare.py"
# ```
# Or open in VS Code and use the Jupyter cell runner (Ctrl+Shift+P → "Run All Cells").

# %% imports
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # works without a display (Docker)
import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py
except ImportError as _e:
    raise ImportError("h5py is required — install with: pip install h5py") from _e

try:
    import pandas as pd
except ImportError:
    pd = None   # optional; CSV export still works via stdlib csv module

try:
    from mpi4py import MPI
    from dolfinx import fem, geometry, io
    _DOLFINX_AVAILABLE = True
except ImportError:
    _DOLFINX_AVAILABLE = False
    print("[WARN] dolfinx not found — FEM evaluation will be skipped.")


# %% ==========================================================================
# CONFIGURATION — edit this section
# =============================================================================

# Root of the june 26 campaign folder.
# If running from repo root with run_dolfinx.sh, this is relative to /work.
REPO_ROOT = Path(__file__).parent.parent.parent   # ../../../ = repo root
JUNE_ROOT = REPO_ROOT / "june 26"

# Trusted MaSQE VTK reference.
VTK_PATH = JUNE_ROOT / "vtk" / "basePotential3d.vtk"
VTK_SCALAR = "basePotential"   # set None to auto-detect first SCALARS field

# DOLFINx FEM cases to compare.
# Add entries here once you have device-scale FEM outputs that match the
# MaSQE geometry (x: ±250 nm, y: ±250 nm, z: 0–255 nm, phi: −12 to −1 V).
#
# Example:
#   CASES = [
#       {
#           "label": "device_h10nm",
#           "xdmf":  JUNE_ROOT / "fem" / "device_h10nm" / "phi.xdmf",
#           "h5":    JUNE_ROOT / "fem" / "device_h10nm" / "phi.h5",
#           "field": "phi",           # name of the Attribute in XDMF / key in H5
#           "h5_dataset": None,       # None → auto-discover from XDMF XML or H5
#       },
#   ]
#
# NOTE: The existing files under fem/benchmark_results/ are POINT-CHARGE
# verification cases (±100 nm box, phi 0–0.16 V); they will not overlap the
# device VTK and are included here only as structural examples.
CASES: list[dict] = [
    # Disk gate R=10 nm — closer to MaSQE gate geometry (sub-10 nm gate, rest Neumann).
    # Solver: disk_gate_box style → z=0 is gate, z=255 nm is bulk (same as VTK) → z_flip=False.
    # Layer ordering: QW at z_dolfinx=50–55 nm (= z_vtk=200–205 nm from gate; physical QW
    # is at z_vtk=50–55 nm — wrong but small epsilon effect, acceptable for diagnostic run).
    {
        "label": "disk_R10nm_h6nm",
        "xdmf":  JUNE_ROOT / "fem" / "direct_R10_hex_h6_neg1_neg12_Lz255"
                            / "direct_R10_hex_h6_neg1_neg12_Lz255.xdmf",
        "h5":    JUNE_ROOT / "fem" / "direct_R10_hex_h6_neg1_neg12_Lz255"
                            / "direct_R10_hex_h6_neg1_neg12_Lz255.h5",
        "field":  "phi",
        "z_flip": False,   # disk case: z=0→gate, z=255→bulk — same as VTK
    },
    # Case A (uniform top gate, wrong physics for this VTK but kept for reference).
    # Solver: layered_disk_gate_box → z=0 is bulk, z=255 nm is gate → z_flip=True.
    # {
    #     "label": "caseA_h20nm_fulltop",
    #     "xdmf":  JUNE_ROOT / "fem" / "caseA_base_neg1_bottom_neg12" / "phi.xdmf",
    #     "h5":    JUNE_ROOT / "fem" / "caseA_base_neg1_bottom_neg12" / "phi.h5",
    #     "field": "phi",
    #     "z_flip": True,   # layered_disk_gate_box: z=0→bulk, z=255→gate — opposite of VTK
    # },
]

# Auto-discover XDMF cases from june 26/fem/ when CASES is empty.
# Disabled: CASES is now explicitly configured above.
AUTO_DISCOVER = False

# Active-minus-background FEM difference pairs.
# Each entry must reference two labels that appear in CASES.
# The "delta" field = phi_active - phi_background.
DELTA_PAIRS: list[dict] = [
    # Example:
    # {
    #     "name":             "disk_perturbation",
    #     "active_label":     "disk_active_h5nm",
    #     "background_label": "background_h5nm",
    # },
]

# z-slices to compare (in nm).  Must lie within the VTK z range [0, 255].
Z_SLICES_NM = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 155.0, 200.0, 255.0]

# Shared x-y grid resolution (points per axis) for each z-slice.
GRID_N = 201

# Percent-error denominator floor [V].  Prevents blowup near phi ≈ 0.
PERCENT_FLOOR = 0.01   # 10 mV — appropriate for a –12 V to –1 V device

# Fixed colorbar for raw potential plots.
# None → automatic symmetric range from slice data.
FIXED_PHI_CLIM = (-12.0, -1.0)   # match MaSQE VTK range

# Fixed percent-error colorbar.
FIXED_PCT_CLIM = (0.0, 5.0)   # 0–5 %

# Save per-slice point-wise CSV.gz (can be large; disable if disk space is tight).
SAVE_CSV_GZ = True

# Attempt-directory prefix inside june 26/outputs/.
ATTEMPT_PREFIX = "comparison_attempt"


# %% ==========================================================================
# Attempt output directory
# =============================================================================

def _next_attempt(base: Path, prefix: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob(f"{prefix}_*"))
    nums = [int(m.group(1)) for p in existing
            if (m := re.search(r"_(\d+)$", p.name))]
    n = max(nums) + 1 if nums else 1
    out = base / f"{prefix}_{n:03d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


OUTDIR = _next_attempt(JUNE_ROOT / "outputs", ATTEMPT_PREFIX)
print(f"Attempt output directory: {OUTDIR}")


# %% ==========================================================================
# Pure-Python ASCII VTK RectilinearGrid reader
# (no pyvista required — works inside Docker with no extra deps)
# =============================================================================

def _read_n_floats(lines: list[str], start: int, n: int) -> tuple[np.ndarray, int]:
    vals: list[str] = []
    i = start
    while len(vals) < n and i < len(lines):
        vals.extend(lines[i].split())
        i += 1
    if len(vals) < n:
        raise ValueError(f"Expected {n} floats starting at line {start}, got {len(vals)}")
    return np.array(vals[:n], dtype=np.float64), i


def load_vtk_rectilinear(
    vtk_path: Path,
    scalar: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Parse an ASCII VTK RectilinearGrid.

    Returns (x_nm, y_nm, z_nm, phi[x,y,z], field_name).
    phi has shape (nx, ny, nz) with Fortran-order index matching VTK storage.
    Coordinates are assumed to be in nm (verified for basePotential3d.vtk).
    """
    vtk_path = Path(vtk_path)
    if not vtk_path.exists():
        raise FileNotFoundError(vtk_path)

    with open(vtk_path, "r") as fh:
        lines = fh.readlines()

    if not lines[0].strip().startswith("# vtk"):
        raise ValueError(f"Not a VTK file: {vtk_path}")
    if lines[2].strip().upper() != "ASCII":
        raise ValueError("Only ASCII VTK files are supported (got BINARY). Use pyvista instead.")

    nx = ny = nz = None
    x_arr = y_arr = z_arr = None
    field_name: str | None = None
    phi_flat: np.ndarray | None = None
    found_scalars: list[str] = []

    i = 0
    while i < len(lines):
        tok = lines[i].strip().upper()

        if tok.startswith("DIMENSIONS"):
            parts = lines[i].split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])

        elif tok.startswith("X_COORDINATES"):
            n = int(lines[i].split()[1])
            x_arr, i = _read_n_floats(lines, i + 1, n)
            continue

        elif tok.startswith("Y_COORDINATES"):
            n = int(lines[i].split()[1])
            y_arr, i = _read_n_floats(lines, i + 1, n)
            continue

        elif tok.startswith("Z_COORDINATES"):
            n = int(lines[i].split()[1])
            z_arr, i = _read_n_floats(lines, i + 1, n)
            continue

        elif tok.startswith("SCALARS"):
            parts = lines[i].split()
            fn = parts[1]
            found_scalars.append(fn)
            want = (scalar is None and phi_flat is None) or (fn == scalar)
            if want:
                field_name = fn
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].strip().upper().startswith("LOOKUP_TABLE"):
                    j += 1
                n_pts = (nx or 0) * (ny or 0) * (nz or 0)
                phi_flat, i = _read_n_floats(lines, j, n_pts)
                continue

        i += 1

    if nx is None:
        raise ValueError("DIMENSIONS line not found in VTK file.")
    if x_arr is None or y_arr is None or z_arr is None:
        raise ValueError("Coordinate arrays not found in VTK file.")
    if phi_flat is None:
        raise KeyError(
            f"Scalar {scalar!r} not found. Available: {found_scalars}"
        )

    phi_3d = phi_flat.reshape((nx, ny, nz), order="F")
    return x_arr, y_arr, z_arr, phi_3d, field_name


# %% ==========================================================================
# Trilinear interpolation on the VTK rectilinear grid
# =============================================================================

def interp_rectilinear(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    phi: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """
    Boundary-inclusive vectorised trilinear interpolation on a rectilinear grid.

    pts : (N, 3) array of (x, y, z) query points in the same units as xs/ys/zs.
    Returns (N,) array.

    Points strictly outside the grid (beyond a small float tolerance) return NaN.
    Points exactly on a boundary face — including z = 0, z = z_max, x = ±x_max,
    y = ±y_max — are handled correctly by clipping before index lookup, so
    they never produce NaN.
    """
    pts = np.asarray(pts, dtype=np.float64)
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
    out = np.full(len(pts), np.nan, dtype=np.float64)

    # Tolerance: 1 ppm of the grid span, sufficient to absorb floating-point
    # jitter while not accepting genuinely out-of-domain points.
    tol = 1e-6
    inside = (
        (px >= xs[0] - tol) & (px <= xs[-1] + tol) &
        (py >= ys[0] - tol) & (py <= ys[-1] + tol) &
        (pz >= zs[0] - tol) & (pz <= zs[-1] + tol)
    )
    if not np.any(inside):
        return out

    # Clip to exact grid bounds so boundary points map to the last valid cell.
    qx = np.clip(px[inside], xs[0], xs[-1])
    qy = np.clip(py[inside], ys[0], ys[-1])
    qz = np.clip(pz[inside], zs[0], zs[-1])

    i = np.clip(np.searchsorted(xs, qx, side="right") - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, qy, side="right") - 1, 0, len(ys) - 2)
    k = np.clip(np.searchsorted(zs, qz, side="right") - 1, 0, len(zs) - 2)

    tx = np.clip((qx - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0)
    ty = np.clip((qy - ys[j]) / (ys[j + 1] - ys[j]), 0.0, 1.0)
    tz = np.clip((qz - zs[k]) / (zs[k + 1] - zs[k]), 0.0, 1.0)

    c00 = phi[i,     j,     k    ] * (1-tx) + phi[i+1, j,     k    ] * tx
    c10 = phi[i,     j + 1, k    ] * (1-tx) + phi[i+1, j + 1, k    ] * tx
    c01 = phi[i,     j,     k + 1] * (1-tx) + phi[i+1, j,     k + 1] * tx
    c11 = phi[i,     j + 1, k + 1] * (1-tx) + phi[i+1, j + 1, k + 1] * tx

    c0 = c00 * (1-ty) + c10 * ty
    c1 = c01 * (1-ty) + c11 * ty

    out[inside] = c0 * (1-tz) + c1 * tz
    return out


# %% ==========================================================================
# XDMF / H5 loading helpers
# =============================================================================

def _parse_h5_ref_from_xdmf(xdmf_path: Path, field: str) -> tuple[Path, str] | None:
    """Extract (h5_file_path, dataset_path) from XDMF XML for the named field."""
    try:
        tree = ET.parse(str(xdmf_path))
    except ET.ParseError:
        return None
    for attr in tree.getroot().iter():
        if attr.tag.endswith("Attribute") and attr.attrib.get("Name") == field:
            for child in attr.iter():
                if child.tag.endswith("DataItem") and child.text and ".h5:" in child.text:
                    h5_name, ds = child.text.strip().split(":", 1)
                    return (xdmf_path.parent / h5_name.strip()).resolve(), ds.strip()
    return None


def _find_h5_dataset(h5_path: Path, field: str) -> str:
    """Walk H5 to find the most likely dataset for a named field."""
    priorities = [
        f"Function/{field}/0",
        f"Function/{field}",
        field,
    ]
    with h5py.File(h5_path, "r") as h5:
        for p in priorities:
            if p in h5:
                return p
        # Fall back: find any dataset whose path contains the field name.
        hits: list[tuple[int, str]] = []
        def _vis(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset) and field.lower() in name.lower():
                hits.append((obj.size, name))
        h5.visititems(_vis)
        if hits:
            return sorted(hits, reverse=True)[0][1]
        # Last resort: largest numeric 1-D-ish dataset.
        candidates: list[tuple[int, str]] = []
        def _all(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                candidates.append((int(np.prod(obj.shape)), name))
        h5.visititems(_all)
        if not candidates:
            raise KeyError(f"No numeric dataset found in {h5_path}")
        return sorted(candidates, reverse=True)[0][1]


def load_fem_case(
    xdmf_path: Path,
    h5_path: Path | None,
    field: str = "phi",
    h5_dataset: str | None = None,
) -> tuple:
    """
    Load a DOLFINx XDMF/H5 case.

    Returns (mesh, V, u, h5_path_used, dataset_used).
    u is a CG1 Function with the phi values loaded from H5.
    """
    if not _DOLFINX_AVAILABLE:
        raise RuntimeError("dolfinx is not available; cannot load FEM case.")

    comm = MPI.COMM_WORLD
    xdmf_path = Path(xdmf_path)

    # Read mesh — try common mesh names.
    last_err: Exception | None = None
    mesh = None
    for mesh_name in ["mesh", "Grid", "sphere_refined_box", "domain", ""]:
        try:
            with io.XDMFFile(comm, str(xdmf_path), "r") as xf:
                mesh = xf.read_mesh(name=mesh_name) if mesh_name else xf.read_mesh()
            break
        except Exception as e:
            last_err = e

    if mesh is None:
        raise RuntimeError(f"Could not read mesh from {xdmf_path}: {last_err}")

    V = fem.functionspace(mesh, ("Lagrange", 1))
    u = fem.Function(V, name=field)

    # Resolve H5 path and dataset.
    if h5_path is None:
        parsed = _parse_h5_ref_from_xdmf(xdmf_path, field)
        if parsed is not None:
            h5_path, inferred_ds = parsed
            if h5_dataset is None:
                h5_dataset = inferred_ds
        else:
            h5_path = xdmf_path.with_suffix(".h5")
            if not h5_path.exists():
                raise FileNotFoundError(f"Cannot locate H5 for {xdmf_path}")

    h5_path = Path(h5_path).resolve()

    if h5_dataset is None:
        parsed2 = _parse_h5_ref_from_xdmf(xdmf_path, field)
        h5_dataset = parsed2[1] if parsed2 else _find_h5_dataset(h5_path, field)

    with h5py.File(h5_path, "r") as h5:
        data = np.asarray(h5[h5_dataset], dtype=np.float64).reshape(-1)

    if data.size != u.x.array.size:
        raise RuntimeError(
            f"H5 data size {data.size} ≠ DOF count {u.x.array.size} "
            f"(dataset={h5_dataset}, h5={h5_path})"
        )

    u.x.array[:] = data
    u.x.scatter_forward()
    return mesh, V, u, h5_path, h5_dataset


# %% ==========================================================================
# DOLFINx point evaluation
# =============================================================================

def eval_fem_at_points_nm(
    u: "fem.Function",
    pts_nm: np.ndarray,
    batch: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate DOLFINx function at query points in nm.

    DOLFINx mesh is in metres → pts_m = pts_nm * 1e-9.

    Returns (values, inside_mask).
    """
    if not _DOLFINX_AVAILABLE:
        raise RuntimeError("dolfinx not available")

    mesh = u.function_space.mesh
    tdim = mesh.topology.dim
    n = pts_nm.shape[0]

    values = np.full(n, np.nan, dtype=np.float64)
    inside = np.zeros(n, dtype=bool)

    bb = geometry.bb_tree(mesh, tdim)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        pts_m = pts_nm[start:end] * 1e-9   # nm → metres

        cands   = geometry.compute_collisions_points(bb, pts_m)
        collide = geometry.compute_colliding_cells(mesh, cands, pts_m)

        chunk_vals  = np.full(end - start, np.nan, dtype=np.float64)
        chunk_ins   = np.zeros(end - start, dtype=bool)
        eval_pts:   list[np.ndarray] = []
        eval_cells: list[int]        = []
        eval_idx:   list[int]        = []

        for i in range(end - start):
            links = collide.links(i)
            if len(links):
                eval_pts.append(pts_m[i])
                eval_cells.append(int(links[0]))
                eval_idx.append(i)
                chunk_ins[i] = True

        if eval_pts:
            ep = np.asarray(eval_pts, dtype=np.float64)
            ec = np.asarray(eval_cells, dtype=np.int32)
            vals = np.asarray(u.eval(ep, ec)).reshape(len(eval_idx), -1)[:, 0]
            for li, vi in enumerate(eval_idx):
                chunk_vals[vi] = float(vals[li])

        values[start:end] = chunk_vals
        inside[start:end] = chunk_ins

    return values, inside


def fem_bbox_nm(mesh: "mesh.Mesh") -> tuple[np.ndarray, np.ndarray]:
    """Return (mins_nm, maxs_nm) bounding box of a DOLFINx mesh."""
    coords_m = np.asarray(mesh.geometry.x, dtype=float)
    coords_nm = coords_m * 1e9
    return coords_nm.min(axis=0), coords_nm.max(axis=0)


# %% ==========================================================================
# Auto-discovery of XDMF/H5 cases
# =============================================================================

def _discover_cases(fem_root: Path) -> list[dict]:
    cases: list[dict] = []
    seen: set[Path] = set()
    for xdmf in sorted(fem_root.rglob("*.xdmf")) + sorted(fem_root.rglob("*.xmdf")):
        key = xdmf.resolve()
        if key in seen:
            continue
        seen.add(key)
        h5 = None
        for suf in [".h5", ".hdf5"]:
            c = xdmf.with_suffix(suf)
            if c.exists():
                h5 = c
                break
        if h5 is None:
            continue
        # Derive a short label from the parent folder + stem.
        label = f"{xdmf.parent.name}/{xdmf.stem}"
        # Guess field name from XDMF XML.
        field = "phi"
        try:
            tree = ET.parse(str(xdmf))
            attrs = tree.getroot().findall(".//Attribute")
            if attrs:
                field = attrs[0].attrib.get("Name", "phi")
        except Exception:
            pass
        cases.append({"label": label, "xdmf": xdmf, "h5": h5, "field": field})
    return cases


if not CASES and AUTO_DISCOVER:
    auto = _discover_cases(JUNE_ROOT / "fem")
    if auto:
        print(f"Auto-discovered {len(auto)} XDMF case(s):")
        for c in auto:
            print(f"  {c['label']}  ({c['field']})")
        CASES = auto
    else:
        print("[INFO] No XDMF/H5 pairs found in june 26/fem/.")
        print("       Run scripts/stage_june26.py first, then add device-scale FEM outputs.")


# %% ==========================================================================
# Plotting helpers
# =============================================================================

def _imshow(ax, arr2d, gx, gy, title, cmap, clim=None, cbar_label=""):
    extent = [gx.min(), gx.max(), gy.min(), gy.max()]
    kwargs = dict(origin="lower", extent=extent, aspect="equal",
                  cmap=cmap, interpolation="nearest")
    if clim:
        kwargs.update(vmin=clim[0], vmax=clim[1])
    im = ax.imshow(arr2d.T, **kwargs)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x [nm]", fontsize=8)
    ax.set_ylabel("y [nm]", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=7)
    return im


def save_slice_panel(
    gx: np.ndarray,
    gy: np.ndarray,
    phi_vtk: np.ndarray,
    phi_fem: np.ndarray,
    z_actual: float,
    label: str,
    outpath: Path,
) -> None:
    diff = phi_fem - phi_vtk
    pct = 100.0 * np.abs(diff) / np.maximum(np.abs(phi_vtk), PERCENT_FLOOR)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{label}  z = {z_actual:.3g} nm", fontsize=10)

    # Shared raw-potential limits
    raw_clim = FIXED_PHI_CLIM
    if raw_clim is None:
        valid = np.concatenate([phi_vtk[np.isfinite(phi_vtk)].ravel(),
                                phi_fem[np.isfinite(phi_fem)].ravel()])
        if valid.size:
            raw_clim = (float(np.nanpercentile(valid, 1)),
                        float(np.nanpercentile(valid, 99)))

    # Symmetric diff limits
    fd = diff[np.isfinite(diff)].ravel()
    dlim = float(np.percentile(np.abs(fd), 99)) if fd.size else 0.1
    dlim = max(dlim, 1e-9)

    _imshow(axes[0, 0], phi_vtk, gx, gy, "VTK reference  φ (V)",
            "RdBu_r", raw_clim, "V")
    _imshow(axes[0, 1], phi_fem, gx, gy, "DOLFINx FEM  φ (V)",
            "RdBu_r", raw_clim, "V")
    _imshow(axes[1, 0], diff,    gx, gy, "FEM − VTK  [V]",
            "coolwarm", (-dlim, dlim), "V")
    _imshow(axes[1, 1], pct,     gx, gy, "percent error  [%]",
            "hot_r", FIXED_PCT_CLIM, "%")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_linecut(
    gx: np.ndarray,
    gy: np.ndarray,
    phi_vtk: np.ndarray,
    phi_fem: np.ndarray,
    z_actual: float,
    label: str,
    axis: str,
    outpath: Path,
) -> None:
    if axis == "x":
        mid = len(gy) // 2
        x_vals, ref_line, fem_line = gx, phi_vtk[:, mid], phi_fem[:, mid]
        xlabel = "x [nm]"
        midlabel = f"y = {gy[mid]:.3g} nm"
    else:
        mid = len(gx) // 2
        x_vals, ref_line, fem_line = gy, phi_vtk[mid, :], phi_fem[mid, :]
        xlabel = "y [nm]"
        midlabel = f"x = {gx[mid]:.3g} nm"

    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.plot(x_vals, ref_line, label="VTK reference")
    ax.plot(x_vals, fem_line, "--", label="DOLFINx FEM")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("φ [V]")
    ax.set_title(f"{label}  z = {z_actual:.3g} nm  ({midlabel})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def save_histogram(diff2d: np.ndarray, label: str, z_actual: float, outpath: Path) -> None:
    d = diff2d[np.isfinite(diff2d)].ravel()
    if d.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.hist(d, bins=80, edgecolor="none")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("FEM − VTK [V]")
    ax.set_ylabel("count")
    ax.set_title(f"{label}  z = {z_actual:.3g} nm  (signed diff histogram)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# %% ==========================================================================
# Per-slice metrics
# =============================================================================

def slice_metrics(
    phi_vtk: np.ndarray,
    phi_fem: np.ndarray,
    inside: np.ndarray,
    label: str,
    z_nm: float,
    z_actual: float,
) -> dict:
    mask = np.isfinite(phi_vtk) & np.isfinite(phi_fem) & inside
    n_valid = int(mask.sum())
    n_total = int(phi_vtk.size)

    base = dict(label=label, z_requested_nm=z_nm, z_actual_nm=z_actual,
                n_total=n_total, n_valid=n_valid,
                coverage=n_valid / n_total if n_total else 0.0)

    if n_valid == 0:
        return {**base, **{k: float("nan") for k in
                           ["mae_V", "rms_V", "max_abs_V", "mean_pct", "rms_pct",
                            "p95_pct", "p99_pct", "max_pct", "rel_L2", "bias_V"]}}

    r = phi_vtk[mask]
    t = phi_fem[mask]
    diff = t - r
    absD = np.abs(diff)
    denom = np.maximum(np.abs(r), PERCENT_FLOOR)
    pct = 100.0 * absD / denom

    ref_norm = float(np.linalg.norm(r))
    rel_L2   = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else float("nan")

    return {
        **base,
        "mae_V":     float(np.mean(absD)),
        "rms_V":     float(np.sqrt(np.mean(diff**2))),
        "max_abs_V": float(np.max(absD)),
        "mean_pct":  float(np.mean(pct)),
        "rms_pct":   float(np.sqrt(np.mean(pct**2))),
        "p95_pct":   float(np.percentile(pct, 95)),
        "p99_pct":   float(np.percentile(pct, 99)),
        "max_pct":   float(np.max(pct)),
        "rel_L2":    rel_L2,
        "bias_V":    float(np.mean(diff)),
    }


def write_csv_gz(path: Path, gx, gy, z_actual, phi_vtk, phi_fem, inside) -> None:
    import gzip
    path.parent.mkdir(parents=True, exist_ok=True)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    diff = phi_fem - phi_vtk
    pct  = 100.0 * np.abs(diff) / np.maximum(np.abs(phi_vtk), PERCENT_FLOOR)
    fin  = np.isfinite(phi_vtk) & np.isfinite(phi_fem) & inside.reshape(phi_vtk.shape)

    rows = zip(
        GX.ravel(), GY.ravel(),
        np.full(GX.size, z_actual),
        phi_vtk.ravel(), phi_fem.ravel(),
        diff.ravel(), pct.ravel(), fin.ravel(),
    )
    with gzip.open(path, "wt", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x_nm", "y_nm", "z_nm",
                     "phi_vtk_V", "phi_fem_V",
                     "diff_fem_minus_vtk_V", "pct_error", "finite_pair"])
        wr.writerows(rows)


# %% ==========================================================================
# Load VTK reference
# =============================================================================

print(f"\nLoading VTK reference: {VTK_PATH}")
if not VTK_PATH.exists():
    raise FileNotFoundError(
        f"{VTK_PATH}\n"
        "Run scripts/stage_june26.py first to copy basePotential3d.vtk into june 26/vtk/"
    )

vtk_x, vtk_y, vtk_z, vtk_phi, vtk_field = load_vtk_rectilinear(VTK_PATH, VTK_SCALAR)

print(f"  field      : {vtk_field}")
print(f"  dimensions : {len(vtk_x)} × {len(vtk_y)} × {len(vtk_z)}")
print(f"  x range    : [{vtk_x.min():.4g}, {vtk_x.max():.4g}] nm")
print(f"  y range    : [{vtk_y.min():.4g}, {vtk_y.max():.4g}] nm")
print(f"  z range    : [{vtk_z.min():.4g}, {vtk_z.max():.4g}] nm")
print(f"  phi range  : [{vtk_phi.min():.6g}, {vtk_phi.max():.6g}] V")


# %% ==========================================================================
# Main comparison loop
# =============================================================================

all_metrics: list[dict] = []
loaded_fem: dict[str, dict] = {}   # label → {"mesh", "u", "mins_nm", "maxs_nm"}

if not CASES:
    print("\n[INFO] CASES is empty. Add device-scale FEM outputs to CASES in the CONFIG section.")

for idx, case in enumerate(CASES, start=1):
    label   = str(case["label"])
    xdmf_p  = Path(case["xdmf"])
    h5_p    = Path(case["h5"])   if case.get("h5")   else None
    field   = str(case.get("field",      "phi"))
    h5_ds   = case.get("h5_dataset",    None)

    safe_label = re.sub(r"[^\w\-.]+", "_", label).strip("_") or f"case{idx:02d}"
    case_dir   = OUTDIR / f"{idx:02d}_{safe_label}"
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"[{idx}/{len(CASES)}] {label}")
    print(f"{'='*80}")
    print(f"  xdmf  : {xdmf_p}")
    print(f"  h5    : {h5_p}")
    print(f"  field : {field}")

    if not _DOLFINX_AVAILABLE:
        print("  [SKIP] dolfinx not available")
        continue

    try:
        mesh, V, u, h5_used, ds_used = load_fem_case(xdmf_p, h5_p, field, h5_ds)
    except Exception as e:
        print(f"  [ERROR] could not load FEM case: {e}")
        continue

    fem_min, fem_max = fem_bbox_nm(mesh)
    phi_range = (float(u.x.array.min()), float(u.x.array.max()))
    print(f"  dataset   : {ds_used}")
    print(f"  DOFs      : {u.x.array.size}")
    print(f"  FEM x     : [{fem_min[0]:.4g}, {fem_max[0]:.4g}] nm")
    print(f"  FEM y     : [{fem_min[1]:.4g}, {fem_max[1]:.4g}] nm")
    print(f"  FEM z     : [{fem_min[2]:.4g}, {fem_max[2]:.4g}] nm")
    print(f"  phi range : [{phi_range[0]:.6g}, {phi_range[1]:.6g}] V")

    loaded_fem[label] = {"mesh": mesh, "u": u, "min": fem_min, "max": fem_max}

    # Common x-y grid = intersection of VTK and FEM bounding boxes.
    gx_lo = max(float(vtk_x.min()), float(fem_min[0]))
    gx_hi = min(float(vtk_x.max()), float(fem_max[0]))
    gy_lo = max(float(vtk_y.min()), float(fem_min[1]))
    gy_hi = min(float(vtk_y.max()), float(fem_max[1]))

    if gx_hi <= gx_lo or gy_hi <= gy_lo:
        print(f"  [WARN] x-y bounding boxes do not overlap — skipping slices.")
        print(f"         VTK  x:[{vtk_x.min():.3g},{vtk_x.max():.3g}]  y:[{vtk_y.min():.3g},{vtk_y.max():.3g}]")
        print(f"         FEM  x:[{fem_min[0]:.3g},{fem_max[0]:.3g}]  y:[{fem_min[1]:.3g},{fem_max[1]:.3g}]")
        print(f"         Likely unit mismatch — FEM coords look like metres, not nm.")
        print(f"         Check that the DOLFINx mesh was built in metres (÷ 1e9 from nm).")
        continue

    gx = np.linspace(gx_lo, gx_hi, GRID_N)
    gy = np.linspace(gy_lo, gy_hi, GRID_N)

    print(f"\n  Common x-y grid: [{gx_lo:.3g}, {gx_hi:.3g}] × [{gy_lo:.3g}, {gy_hi:.3g}] nm")

    # Per-slice.
    for z_req in Z_SLICES_NM:
        gz_lo = max(float(vtk_z.min()), float(fem_min[2]))
        gz_hi = min(float(vtk_z.max()), float(fem_max[2]))

        if z_req < gz_lo or z_req > gz_hi:
            continue

        # Snap to nearest VTK z grid point.
        k = int(np.argmin(np.abs(vtk_z - z_req)))
        z_act = float(vtk_z[k])

        print(f"\n  z = {z_req:.1f} nm  (VTK grid: {z_act:.4f} nm)", end="", flush=True)

        GX, GY = np.meshgrid(gx, gy, indexing="ij")

        # VTK convention: z=0 is the gate (−1 V), z=255 nm is the bulk (−12 V).
        # DOLFINx convention depends on the solver used to generate the case:
        #   z_flip=True  → FEM z=0 is bulk, z=H is gate (layered_disk_gate_box style)
        #                   requires z_dolfinx = vtk_z_max − z_vtk
        #   z_flip=False → FEM z=0 is gate, z=H is bulk (disk_gate_box style)
        #                   same convention as VTK, no transform needed
        # Specify per case in CASES via "z_flip": True/False (default False).
        z_flip = bool(case.get("z_flip", False))
        GZ_vtk = np.full_like(GX, z_act)
        GZ_fem = np.full_like(GX, float(vtk_z[-1]) - z_act if z_flip else z_act)

        pts_vtk = np.stack([GX.ravel(), GY.ravel(), GZ_vtk.ravel()], axis=1)  # (N,3) nm
        pts_fem = np.stack([GX.ravel(), GY.ravel(), GZ_fem.ravel()], axis=1)  # (N,3) nm

        t0 = time.perf_counter()

        # VTK evaluation (trilinear).
        phi_vtk_flat = interp_rectilinear(vtk_x, vtk_y, vtk_z, vtk_phi, pts_vtk)
        phi_vtk_2d   = phi_vtk_flat.reshape(GRID_N, GRID_N)

        # FEM evaluation (DOLFINx bb-tree, pts in nm → converted to m inside).
        phi_fem_flat, ins_flat = eval_fem_at_points_nm(u, pts_fem)
        phi_fem_2d = phi_fem_flat.reshape(GRID_N, GRID_N)
        ins_2d     = ins_flat.reshape(GRID_N, GRID_N)

        dt = time.perf_counter() - t0

        m = slice_metrics(phi_vtk_2d, phi_fem_2d, ins_2d, label, z_req, z_act)
        all_metrics.append(m)

        print(
            f"  coverage={m['coverage']:.2%}  "
            f"rms={m['rms_V']:.4e} V  "
            f"p95%={m['p95_pct']:.2f}%  "
            f"({dt:.1f}s)"
        )

        z_dir = case_dir / f"z{z_req:.0f}nm"
        z_dir.mkdir(parents=True, exist_ok=True)
        slug = f"{safe_label}_z{z_req:.0f}nm"

        save_slice_panel(gx, gy, phi_vtk_2d, phi_fem_2d, z_act, label,
                         z_dir / f"{slug}_panel.png")
        save_linecut(gx, gy, phi_vtk_2d, phi_fem_2d, z_act, label, "x",
                     z_dir / f"{slug}_linecut_x.png")
        save_linecut(gx, gy, phi_vtk_2d, phi_fem_2d, z_act, label, "y",
                     z_dir / f"{slug}_linecut_y.png")
        save_histogram(phi_fem_2d - phi_vtk_2d, label, z_act,
                       z_dir / f"{slug}_hist.png")

        if SAVE_CSV_GZ:
            write_csv_gz(z_dir / f"{slug}_points.csv.gz",
                         gx, gy, z_act, phi_vtk_2d, phi_fem_2d, ins_2d)

print("\nMain comparison loop complete.")


# %% ==========================================================================
# Summary CSV and ranking
# =============================================================================

summary_csv = OUTDIR / "slice_summary.csv"
if all_metrics:
    keys = list(all_metrics[0].keys())
    with open(summary_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(all_metrics)

    print(f"\nSaved slice summary: {summary_csv}")
    print(f"\n{'label':<35} {'z_req':>7} {'coverage':>9} {'rms_V':>11} {'p95%':>8}")
    print("-" * 75)
    for m in sorted(all_metrics, key=lambda r: (r["label"], r["z_requested_nm"])):
        print(
            f"{m['label'][:35]:<35} {m['z_requested_nm']:>7.1f}"
            f" {m['coverage']:>9.2%} {m['rms_V']:>11.4e} {m['p95_pct']:>8.2f}"
        )
else:
    print("\n[INFO] No metrics collected (no CASES ran successfully).")


# %% ==========================================================================
# Ranking figure (best p95% across all z-slices)
# =============================================================================

if all_metrics:
    # Best p95% for each case (across slices).
    by_case: dict[str, list[float]] = {}
    for m in all_metrics:
        if np.isfinite(m["p95_pct"]):
            by_case.setdefault(m["label"], []).append(m["p95_pct"])

    if by_case:
        labels_sorted = sorted(by_case, key=lambda l: min(by_case[l]))
        best_p95 = [min(by_case[l]) for l in labels_sorted]

        fig, ax = plt.subplots(figsize=(max(6, len(labels_sorted) * 1.2), 5), dpi=130)
        ax.bar(range(len(labels_sorted)), best_p95)
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("best p95 percent error [%]")
        ax.set_title("Ranking: best p95 percent error (min across z-slices)")
        ax.axhline(1.0, color="r", linewidth=1, linestyle="--", label="1% target")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTDIR / "ranking_best_p95.png", dpi=130)
        plt.close(fig)
        print(f"Saved ranking figure: {OUTDIR / 'ranking_best_p95.png'}")


# %% ==========================================================================
# FEM active-minus-background difference pairs
# =============================================================================

for pair in DELTA_PAIRS:
    act_label = pair["active_label"]
    bg_label  = pair["background_label"]
    name      = str(pair.get("name", f"{act_label}_minus_{bg_label}"))
    safe_name = re.sub(r"[^\w\-.]+", "_", name).strip("_")

    if act_label not in loaded_fem or bg_label not in loaded_fem:
        print(f"\n[SKIP] delta pair '{name}': one or both cases not loaded.")
        continue

    u_act = loaded_fem[act_label]["u"]
    u_bg  = loaded_fem[bg_label]["u"]

    # Common grid = intersection of both FEM bboxes.
    lo = np.maximum(loaded_fem[act_label]["min"], loaded_fem[bg_label]["min"])
    hi = np.minimum(loaded_fem[act_label]["max"], loaded_fem[bg_label]["max"])
    if np.any(hi <= lo):
        print(f"\n[WARN] delta pair '{name}': bounding boxes do not overlap, skipping.")
        continue

    gx = np.linspace(lo[0], hi[0], GRID_N)
    gy = np.linspace(lo[1], hi[1], GRID_N)

    pair_dir = OUTDIR / f"delta_{safe_name}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Delta pair: {name}")
    print(f"  active     : {act_label}")
    print(f"  background : {bg_label}")
    print(f"{'='*80}")

    delta_metrics: list[dict] = []

    for z_req in Z_SLICES_NM:
        if z_req < lo[2] or z_req > hi[2]:
            continue

        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        GZ = np.full_like(GX, z_req)
        pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)

        phi_act, ins_act = eval_fem_at_points_nm(u_act, pts)
        phi_bg,  ins_bg  = eval_fem_at_points_nm(u_bg,  pts)

        delta = (phi_act - phi_bg).reshape(GRID_N, GRID_N)
        fin = (ins_act & ins_bg).reshape(GRID_N, GRID_N)

        d = delta[np.isfinite(delta) & fin]
        row = {
            "pair": name, "active": act_label, "background": bg_label,
            "z_nm": z_req,
            "delta_min_V":  float(d.min())              if d.size else float("nan"),
            "delta_max_V":  float(d.max())              if d.size else float("nan"),
            "delta_mean_V": float(d.mean())             if d.size else float("nan"),
            "delta_rms_V":  float(np.sqrt((d**2).mean())) if d.size else float("nan"),
        }
        delta_metrics.append(row)

        print(f"  z={z_req:.0f}nm  delta:[{row['delta_min_V']:.4e},{row['delta_max_V']:.4e}] V"
              f"  rms={row['delta_rms_V']:.4e} V")

        z_dir = pair_dir / f"z{z_req:.0f}nm"
        z_dir.mkdir(parents=True, exist_ok=True)

        fd = delta[np.isfinite(delta)]
        dlim = float(np.percentile(np.abs(fd), 99)) if fd.size else 0.01
        dlim = max(dlim, 1e-12)

        fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
        im = ax.imshow(
            delta.T, origin="lower",
            extent=[gx.min(), gx.max(), gy.min(), gy.max()],
            aspect="equal", cmap="coolwarm", vmin=-dlim, vmax=dlim,
        )
        ax.set_title(f"{name}  z = {z_req:.0f} nm\nactive − background [V]", fontsize=9)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        plt.colorbar(im, ax=ax).set_label("Δφ [V]")
        fig.tight_layout()
        fig.savefig(z_dir / f"{safe_name}_z{z_req:.0f}nm_delta.png", dpi=130)
        plt.close(fig)

    if delta_metrics:
        dp = pair_dir / "delta_summary.csv"
        with open(dp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(delta_metrics[0].keys()))
            wr.writeheader()
            wr.writerows(delta_metrics)
        print(f"  Saved: {dp}")


# %% ==========================================================================
# Run manifest
# =============================================================================

manifest = {
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "vtk": str(VTK_PATH),
    "vtk_field": vtk_field,
    "vtk_x_range_nm": [float(vtk_x.min()), float(vtk_x.max())],
    "vtk_y_range_nm": [float(vtk_y.min()), float(vtk_y.max())],
    "vtk_z_range_nm": [float(vtk_z.min()), float(vtk_z.max())],
    "vtk_phi_range_V": [float(vtk_phi.min()), float(vtk_phi.max())],
    "grid_n": GRID_N,
    "z_slices_nm": Z_SLICES_NM,
    "percent_floor_V": PERCENT_FLOOR,
    "n_cases": len(CASES),
    "n_metrics_rows": len(all_metrics),
    "outdir": str(OUTDIR),
}
mf_path = OUTDIR / "run_manifest.json"
with open(mf_path, "w") as fh:
    json.dump(manifest, fh, indent=2)

print(f"\nRun manifest: {mf_path}")
print(f"All outputs:  {OUTDIR}")
print("Done.")
