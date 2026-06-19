"""Compare a reference VTK potential field against one or more DOLFINx XDMF/H5 outputs.

Workflow: read the VTK reference once, read each XDMF case, linearly interpolate the
XDMF field onto the VTK reference points (treating VTK as ground truth), compute
absolute/relative error globally and on z-slices, and plot side-by-side comparisons.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import griddata

EPS_FLOOR = 1e-12  # avoids division-by-near-zero in relative error


@dataclass
class FieldData:
    name: str
    points: np.ndarray  # (N, 3)
    values: np.ndarray  # (N,)


@dataclass
class XdmfCase:
    label: str
    xdmf_path: Path
    field: str | None = None  # None => first point-data array
    scale: float = 1.0  # multiply coordinates by this (e.g. unit conversion)


@dataclass
class InterpResult:
    points: np.ndarray
    ref_values: np.ndarray
    case_values: np.ndarray
    fallback_mask: np.ndarray  # True where nearest-neighbor fallback was used (outside hull)

    @property
    def nearest_fallback_frac(self) -> float:
        return float(self.fallback_mask.mean()) if len(self.fallback_mask) else float("nan")


def read_vtk_reference(vtk_path: Path, field: str | None = None) -> FieldData:
    vtk_path = Path(vtk_path)
    if not vtk_path.exists():
        raise FileNotFoundError(f"VTK reference not found: {vtk_path}")
    mesh = pv.read(vtk_path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if field is None:
        names = list(mesh.point_data.keys())
        if not names:
            raise ValueError(f"No point data arrays found in {vtk_path}")
        field = names[0]
    elif field not in mesh.point_data:
        raise KeyError(
            f"Field '{field}' not in VTK point data. Available: {list(mesh.point_data.keys())}"
        )
    values = np.asarray(mesh.point_data[field]).reshape(-1)
    points = np.asarray(mesh.points)
    return FieldData(name=field, points=points, values=values)


def write_paraview_friendly_xdmf(xdmf_path: Path, out_path: Path | None = None) -> Path:
    """Re-pack a dolfinx-written XDMF/H5 checkpoint into a single flat Grid with no
    xi:include and no per-field Temporal Collections.

    dolfinx's XDMFFile writes one Topology+Geometry plus a separate
    GridType="Collection" CollectionType="Temporal" wrapper *per field*, each
    referencing the mesh via xi:include/xpointer. That's valid XDMF3, but both
    ParaView's and VTK's XDMF readers are unreliable at resolving it (ParaView shows
    an empty/blank view; pyvista's reader segfaults outright — see
    read_xdmf_case's docstring). Repacking into one inline Grid with N plain
    Attribute elements fixes both.
    """
    xdmf_path = Path(xdmf_path)
    out_path = Path(out_path) if out_path else xdmf_path.with_name(xdmf_path.stem + "_pv.xdmf")

    tree = ET.parse(xdmf_path)
    root = tree.getroot()

    mesh_grid = root.find(".//Grid[@GridType='Uniform']")
    topo = mesh_grid.find("Topology")
    geom = mesh_grid.find("Geometry")
    topo_item = topo.find("DataItem")
    geom_item = geom.find("DataItem")
    h5_name = topo_item.text.split(":")[0].strip()

    attrs = []
    for attr in root.findall(".//Attribute"):
        name = attr.get("Name")
        dim = attr.find("DataItem").get("Dimensions")
        ref = attr.find("DataItem").text.strip()
        attrs.append((name, dim, ref))

    attr_xml = "\n".join(
        f"""        <Attribute Name="{name}" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{dim}" Format="HDF">{ref}</DataItem>
        </Attribute>"""
        for name, dim, ref in attrs
    )

    xdmf_txt = f"""<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="{mesh_grid.get('Name')}" GridType="Uniform">
      <Topology TopologyType="{topo.get('TopologyType')}" NumberOfElements="{topo.get('NumberOfElements')}" NodesPerElement="{topo.get('NodesPerElement')}">
        <DataItem Dimensions="{topo_item.get('Dimensions')}" NumberType="Int" Format="HDF">{topo_item.text.strip()}</DataItem>
      </Topology>
      <Geometry GeometryType="{geom.get('GeometryType')}">
        <DataItem Dimensions="{geom_item.get('Dimensions')}" Format="HDF">{geom_item.text.strip()}</DataItem>
      </Geometry>
      <Time Value="0" />
{attr_xml}
    </Grid>
  </Domain>
</Xdmf>
"""
    out_path.write_text(xdmf_txt)
    return out_path


def _hdf_path_to_dataset(h5_root: Path, ref: str) -> np.ndarray:
    """Resolve an XDMF 'file.h5:/group/dataset' reference into a numpy array."""
    fname, dset_path = ref.split(":")
    fname = fname.strip()
    h5_path = h5_root.parent / fname
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f[dset_path.strip()])


def read_xdmf_case(case: XdmfCase) -> FieldData:
    """Read an XDMF+H5 pair without going through VTK's XDMF reader.

    VTK/pyvista's XDMF3 reader segfaults on the xi:include + temporal-collection
    pattern that dolfinx's XDMFFile writes, so we parse the small XML descriptor
    ourselves and pull arrays straight out of the HDF5 file. Handles both:
      - dolfinx unstructured meshes (Geometry GeometryType="XYZ")
      - the ORIGIN_DXDYDZ uniform grid format written by fft_solve's XDMF writer
    """
    xdmf_path = case.xdmf_path
    if not xdmf_path.exists():
        raise FileNotFoundError(f"XDMF case '{case.label}' not found: {xdmf_path}")

    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    grids = root.findall(".//Grid[@GridType='Uniform']")
    if not grids:
        raise ValueError(f"No Uniform Grid found in {xdmf_path}")
    grid = grids[0]

    geom = grid.find("Geometry")
    if geom is None:
        raise ValueError(f"No Geometry element in {xdmf_path}")
    geom_type = geom.get("GeometryType")

    if geom_type == "XYZ":
        data_item = geom.find("DataItem")
        points = _hdf_path_to_dataset(xdmf_path, data_item.text.strip())
    elif geom_type == "ORIGIN_DXDYDZ":
        topo = grid.find("Topology")
        dims = [int(d) for d in topo.get("Dimensions").split()]
        items = geom.findall("DataItem")
        origin = [float(v) for v in items[0].text.split()]
        spacing = [float(v) for v in items[1].text.split()]
        axes = [
            np.arange(n) * spacing[i] + origin[i] for i, n in enumerate(dims)
        ]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    else:
        raise ValueError(f"Unsupported GeometryType '{geom_type}' in {xdmf_path}")

    attrs = grid.findall(".//Attribute")
    if not attrs:
        # dolfinx checkpoint format nests the real grid under a Collection;
        # fall back to searching the whole tree.
        attrs = root.findall(".//Attribute")
    available = [a.get("Name") for a in attrs]
    field = case.field
    if field is None:
        if not attrs:
            raise ValueError(f"No Attribute (field) found in {xdmf_path}")
        chosen = attrs[0]
    else:
        matches = [a for a in attrs if a.get("Name") == field]
        if not matches:
            raise KeyError(f"Field '{field}' not in '{case.label}'. Available: {available}")
        chosen = matches[0]
    field_name = chosen.get("Name")
    values = _hdf_path_to_dataset(xdmf_path, chosen.find("DataItem").text.strip()).reshape(-1)

    points = points * case.scale
    if len(values) != len(points):
        raise ValueError(
            f"'{case.label}': {len(points)} points but {len(values)} field values "
            f"(field='{field_name}') — mesh/attribute mismatch in {xdmf_path}"
        )
    return FieldData(name=field_name, points=points, values=values)


def check_bbox_overlap(ref: FieldData, case: FieldData, label: str) -> None:
    ref_lo, ref_hi = ref.points.min(axis=0), ref.points.max(axis=0)
    case_lo, case_hi = case.points.min(axis=0), case.points.max(axis=0)
    overlap_lo = np.maximum(ref_lo, case_lo)
    overlap_hi = np.minimum(ref_hi, case_hi)
    overlap_extent = overlap_hi - overlap_lo
    if np.any(overlap_extent <= 0):
        warnings.warn(
            f"[{label}] bounding boxes do NOT overlap on at least one axis.\n"
            f"  ref  bbox: {ref_lo} .. {ref_hi}\n"
            f"  case bbox: {case_lo} .. {case_hi}\n"
            f"  This usually means a unit-scale mismatch (e.g. forgot to convert nm <-> m)."
        )
        return
    ref_extent = ref_hi - ref_lo
    coverage = overlap_extent / np.where(ref_extent == 0, 1, ref_extent)
    if np.any(coverage < 0.5):
        warnings.warn(
            f"[{label}] case covers only {coverage.min():.0%} of the reference extent "
            f"on the worst axis — interpolation will rely heavily on nearest-neighbor "
            f"fallback outside the overlap region."
        )


def interpolate_case_onto_ref(
    ref: FieldData, case: FieldData, label: str, check_bbox: bool = True
) -> InterpResult:
    if check_bbox:
        check_bbox_overlap(ref, case, label)

    linear = griddata(case.points, case.values, ref.points, method="linear")
    nan_mask = np.isnan(linear)
    if nan_mask.any():
        nearest = griddata(case.points, case.values, ref.points[nan_mask], method="nearest")
        linear[nan_mask] = nearest

    return InterpResult(
        points=ref.points,
        ref_values=ref.values,
        case_values=linear,
        fallback_mask=nan_mask,
    )


def compute_metrics(
    result: InterpResult,
    label: str,
    rel_floor: float = EPS_FLOOR,
    pct_mask_threshold: float | None = None,
) -> dict:
    """pct_mask_threshold: if set, percent-error stats also computed restricted to
    |ref| > pct_mask_threshold, since percent error is meaningless where the
    reference field is itself near zero (e.g. far from a localized charge)."""
    diff = result.case_values - result.ref_values
    abs_err = np.abs(diff)
    denom = np.maximum(np.abs(result.ref_values), rel_floor)
    pct_err = 100.0 * abs_err / denom

    ref_norm = np.linalg.norm(result.ref_values)
    rel_l2 = (
        np.linalg.norm(diff) / ref_norm if ref_norm > rel_floor else float("nan")
    )

    metrics = {
        "case": label,
        "n_points": len(diff),
        "nearest_fallback_frac": result.nearest_fallback_frac,
        "abs_err_mean": np.nanmean(abs_err),
        "abs_err_max": np.nanmax(abs_err),
        "rel_l2_error": rel_l2,
        "pct_err_mean": np.nanmean(pct_err),
        "pct_err_median": np.nanmedian(pct_err),
        "pct_err_max": np.nanmax(pct_err),
    }

    if pct_mask_threshold is not None:
        sig_mask = np.abs(result.ref_values) > pct_mask_threshold
        n_sig = int(sig_mask.sum())
        metrics["pct_err_mean_masked"] = (
            np.nanmean(pct_err[sig_mask]) if n_sig else float("nan")
        )
        metrics["pct_err_max_masked"] = (
            np.nanmax(pct_err[sig_mask]) if n_sig else float("nan")
        )
        metrics["n_points_above_threshold"] = n_sig

    return metrics


def nearest_z_mask(points: np.ndarray, z_target: float, tol: float | None = None) -> np.ndarray:
    z = points[:, 2]
    if tol is None:
        uz = np.unique(z)
        spacing = np.median(np.diff(np.sort(uz))) if len(uz) > 1 else 1.0
        tol = max(spacing * 0.5, 1e-9)
    return np.abs(z - z_target) <= tol


def plot_slice_pair(
    result: InterpResult,
    z_target: float,
    label: str,
    units: str = "",
    tol: float | None = None,
    save_path: Path | None = None,
    show: bool = True,
):
    mask = nearest_z_mask(result.points, z_target, tol=tol)
    if mask.sum() < 4:
        warnings.warn(f"[{label}] z={z_target}: only {mask.sum()} points in slice, skipping plot")
        return None

    x, y = result.points[mask, 0], result.points[mask, 1]
    ref_v = result.ref_values[mask]
    case_v = result.case_values[mask]
    diff = case_v - ref_v
    denom = np.maximum(np.abs(ref_v), EPS_FLOOR)
    pct = 100.0 * np.abs(diff) / denom

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    panels = [
        (ref_v, "VTK reference (ground truth)", "viridis"),
        (case_v, f"{label}", "viridis"),
        (diff, "Absolute error (case - ref)", "RdBu_r"),
        (pct, "Relative error [%]", "magma"),
    ]
    for ax, (vals, title, cmap) in zip(axes, panels):
        vmax = np.nanpercentile(np.abs(vals), 99) if title.startswith(("Absolute",)) else None
        kwargs = dict(c=vals, cmap=cmap, s=8)
        if vmax:
            kwargs.update(vmin=-vmax, vmax=vmax)
        sc = ax.scatter(x, y, **kwargs)
        ax.set_title(f"{title}\nz={z_target:.3g}{units}")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_linecut(
    result: InterpResult,
    z_target: float,
    label: str,
    axis: str = "x",
    y0: float = 0.0,
    tol_perp: float | None = None,
    units: str = "",
    save_path: Path | None = None,
    show: bool = True,
):
    mask = nearest_z_mask(result.points, z_target)
    pts = result.points[mask]
    ref_v = result.ref_values[mask]
    case_v = result.case_values[mask]

    perp_idx = 1 if axis == "x" else 0
    perp_coord = pts[:, perp_idx]
    if tol_perp is None:
        tol_perp = max(np.std(perp_coord) * 0.05, 1e-9)
    line_mask = np.abs(perp_coord - y0) <= tol_perp
    if line_mask.sum() < 2:
        warnings.warn(f"[{label}] linecut z={z_target}: not enough points, skipping")
        return None

    axis_idx = 0 if axis == "x" else 1
    order = np.argsort(pts[line_mask, axis_idx])
    s = pts[line_mask, axis_idx][order]
    rv = ref_v[line_mask][order]
    cv = case_v[line_mask][order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, rv, "o-", label="VTK reference", ms=3)
    ax.plot(s, cv, "s--", label=label, ms=3)
    ax.set_xlabel(f"{axis} {units}")
    ax.set_ylabel("potential")
    ax.set_title(f"Linecut along {axis} at z={z_target:.3g}{units}")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def run_comparison(
    vtk_path: Path,
    cases: Sequence[XdmfCase],
    z_slices: Sequence[float],
    vtk_field: str | None = None,
    units: str = "",
    out_dir: Path | None = None,
    show_plots: bool = True,
    pct_mask_threshold: float | None = None,
) -> pd.DataFrame:
    """Run the full VTK-vs-XDMF comparison for an arbitrary number of XDMF cases.

    Returns a DataFrame with one row per (case, slice) plus one 'global' row per case.
    """
    ref = read_vtk_reference(vtk_path, field=vtk_field)
    print(f"Reference '{vtk_path.name}': field='{ref.name}', {len(ref.points)} points")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        print(f"\n--- case: {case.label} ({case.xdmf_path.name}) ---")
        case_data = read_xdmf_case(case)
        result = interpolate_case_onto_ref(ref, case_data, case.label)

        global_metrics = compute_metrics(result, case.label, pct_mask_threshold=pct_mask_threshold)
        global_metrics["slice_z"] = "global"
        rows.append(global_metrics)
        print(
            f"  global: rel_L2={global_metrics['rel_l2_error']:.4e}  "
            f"mean%={global_metrics['pct_err_mean']:.3g}  "
            f"nearest_fallback={global_metrics['nearest_fallback_frac']:.1%}"
        )

        for z in z_slices:
            slice_path = out_dir / f"slice_{case.label}_z{z:g}.png" if out_dir else None
            plot_slice_pair(result, z, case.label, units=units, save_path=slice_path, show=show_plots)

            mask = nearest_z_mask(result.points, z)
            if mask.sum() < 4:
                continue
            slice_result = InterpResult(
                points=result.points[mask],
                ref_values=result.ref_values[mask],
                case_values=result.case_values[mask],
                fallback_mask=result.fallback_mask[mask],
            )
            slice_metrics = compute_metrics(slice_result, case.label, pct_mask_threshold=pct_mask_threshold)
            slice_metrics["slice_z"] = z
            rows.append(slice_metrics)

    df = pd.DataFrame(rows)
    if out_dir is not None:
        df.to_csv(out_dir / "comparison_metrics.csv", index=False)
    return df
