"""
analysis/xdmf_utils.py
-----------------------
Shared helpers for loading DOLFINx XDMF/H5 results and evaluating FEM functions
at arbitrary points. Imported by compare_background_cases.py, compare_delta_cases.py,
and compare_gate_vs_background.py.

Requires: dolfinx, h5py, numpy (all available in dolfinx/dolfinx:nightly)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpi4py import MPI
import h5py
from dolfinx import fem, io
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points


# ── XDMF loading ──────────────────────────────────────────────────────────────

def _find_h5_dataset(xdmf_path: Path, field_name: str) -> tuple[Path, str]:
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


def load_xdmf_field(xdmf_path: Path, field_name: str = "phi") -> fem.Function:
    """Load a single scalar field from a DOLFINx XDMF/H5 pair."""
    comm = MPI.COMM_WORLD
    xdmf_path = Path(xdmf_path).resolve()
    h5_file, dataset = _find_h5_dataset(xdmf_path, field_name)

    with io.XDMFFile(comm, str(xdmf_path), "r") as xf:
        msh = xf.read_mesh(name="mesh")

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V, name=field_name)

    with h5py.File(str(h5_file), "r") as h5:
        data = np.asarray(h5[dataset], dtype=np.float64).reshape(-1)

    if data.size != u.x.array.size:
        raise RuntimeError(f"DOF mismatch: {data.size} vs {u.x.array.size}")

    u.x.array[:] = data
    u.x.scatter_forward()

    phi = u.x.array
    coords = V.tabulate_dof_coordinates()
    print(f"  loaded {field_name} from {xdmf_path.name}: "
          f"[{phi.min():.4f}, {phi.max():.4f}]  ({phi.size} dofs)  "
          f"z=[{coords[:,2].min()*1e9:.0f}, {coords[:,2].max()*1e9:.0f}] nm")
    return u


# ── FEM evaluation ────────────────────────────────────────────────────────────

def eval_at_points(u: fem.Function, pts_m: np.ndarray) -> np.ndarray:
    """Evaluate a DOLFINx FEM function at physical points (metres)."""
    msh = u.function_space.mesh
    tree = bb_tree(msh, msh.topology.dim)
    cand = compute_collisions_points(tree, pts_m)
    coll = compute_colliding_cells(msh, cand, pts_m)

    n = pts_m.shape[0]
    inside = np.zeros(n, dtype=bool)
    cells  = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        links = coll.links(i)
        if len(links):
            inside[i] = True
            cells[i]  = links[0]

    vals = np.full(n, np.nan)
    if np.any(inside):
        vals[inside] = np.asarray(u.eval(pts_m[inside], cells[inside])).reshape(-1)
    return vals


def eval_on_grid(u: fem.Function, z_nm: float, grid_n: int,
                 x_range_nm: tuple, y_range_nm: tuple) -> tuple:
    """
    Evaluate u on an (grid_n × grid_n) Cartesian xy grid at fixed z=z_nm.
    Returns (xg_nm, yg_nm, phi_2d) where phi_2d shape is (grid_n, grid_n).
    """
    xg = np.linspace(x_range_nm[0], x_range_nm[1], grid_n) * 1e-9
    yg = np.linspace(y_range_nm[0], y_range_nm[1], grid_n) * 1e-9
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z_nm * 1e-9)])
    phi_flat = eval_at_points(u, pts)
    return xg * 1e9, yg * 1e9, phi_flat.reshape(grid_n, grid_n)


def mesh_xy_range(u: fem.Function) -> tuple:
    """Return (x_min, x_max, y_min, y_max) in nm from mesh coordinates."""
    coords = u.function_space.mesh.geometry.x
    return (coords[:,0].min()*1e9, coords[:,0].max()*1e9,
            coords[:,1].min()*1e9, coords[:,1].max()*1e9)


# ── Depth profile ─────────────────────────────────────────────────────────────

def eval_depth_line(u: fem.Function, cx_nm: float, cy_nm: float,
                    npts: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """phi along the vertical line (cx, cy) from z=0 to z=Lz. Returns (z_nm, phi)."""
    coords = u.function_space.mesh.geometry.x
    z_min = coords[:,2].min()
    z_max = coords[:,2].max()
    zs = np.linspace(z_min, z_max, npts)
    pts = np.column_stack([np.full(npts, cx_nm*1e-9),
                           np.full(npts, cy_nm*1e-9), zs])
    return zs * 1e9, eval_at_points(u, pts)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _ticks(ax):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.set_xlabel("x (nm)", fontsize=8)
    ax.set_ylabel("y (nm)", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_slice_panel(fig, axes_row, xg, yg, arrays: list[tuple[str, np.ndarray]],
                     z_nm: float, shared_vmin=None, shared_vmax=None, cmap="RdBu_r"):
    """
    Plot multiple 2D arrays in a row of axes.
    arrays: list of (title, 2d_array) pairs.
    """
    vlo = shared_vmin if shared_vmin is not None else min(np.nanmin(a) for _, a in arrays)
    vhi = shared_vmax if shared_vmax is not None else max(np.nanmax(a) for _, a in arrays)
    for ax, (title, arr) in zip(axes_row, arrays):
        ext = [xg.min(), xg.max(), yg.min(), yg.max()]
        im = ax.imshow(arr.T, origin="lower", extent=ext, aspect="equal",
                       cmap=cmap, vmin=vlo, vmax=vhi, interpolation="bilinear")
        ax.set_title(f"{title}\nz={z_nm:.0f} nm", fontsize=8)
        _ticks(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)


def disk_circle(ax, cx_nm, cy_nm, r_nm, **kw):
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(cx_nm + r_nm*np.cos(theta), cy_nm + r_nm*np.sin(theta),
            **{"color":"k", "ls":"--", "lw":0.8, "alpha":0.6, **kw})
