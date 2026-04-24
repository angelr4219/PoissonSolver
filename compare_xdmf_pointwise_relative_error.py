#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from mpi4py import MPI

from dolfinx import fem, geometry, io


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare the same scalar CG1 field stored in two XDMF/HDF5 outputs by "
            "sampling both at the same physical points and computing pointwise relative error."
        )
    )
    p.add_argument("--xdmf_a", type=str, required=True, help="First XDMF file")
    p.add_argument("--xdmf_b", type=str, required=True, help="Second XDMF file")
    p.add_argument("--field", type=str, default="phi", help="Field name to compare, default: phi")
    p.add_argument(
        "--reference",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Use file a or b as the reference mesh/field for sampling points"
    )
    p.add_argument(
        "--rel_floor",
        type=float,
        default=1.0e-12,
        help=(
            "Relative denominator floor as a fraction of max(|reference|, 1). "
            "Used to avoid blow-up near zero."
        ),
    )
    p.add_argument("--outdir", type=str, default="Results/field_compare")
    p.add_argument("--basename", type=str, default="field_compare")
    return p.parse_args()


def require_serial(comm: MPI.Intracomm) -> None:
    if comm.size != 1:
        raise RuntimeError(
            "This comparison script currently supports serial execution only. "
            "Run it with plain python3 inside Docker, not mpiexec."
        )


def parse_h5_target_from_xdmf(xdmf_path: str | Path, field_name: str) -> tuple[Path, str]:
    """
    Parse the HDF5 target path for an Attribute Name=<field_name> in an XDMF file.

    Returns
    -------
    h5_file : Path
        Path to the HDF5 file referenced by the XDMF.
    h5_dataset : str
        Internal HDF5 dataset path.
    """
    xdmf_path = Path(xdmf_path)
    tree = ET.parse(xdmf_path)
    root = tree.getroot()

    for attr in root.iter():
        if attr.tag.endswith("Attribute") and attr.attrib.get("Name") == field_name:
            for child in attr.iter():
                if child.tag.endswith("DataItem") and child.text is not None:
                    text = child.text.strip()
                    if ".h5:" in text:
                        h5_name, dataset = text.split(":", 1)
                        h5_file = (xdmf_path.parent / h5_name).resolve()
                        return h5_file, dataset.strip()

    raise RuntimeError(f"Could not find field '{field_name}' in XDMF file: {xdmf_path}")


def load_cg1_function_from_xdmf(xdmf_path: str | Path, field_name: str) -> tuple:
    """
    Read mesh from XDMF and reconstruct a scalar CG1 function from the linked HDF5 dataset.
    """
    comm = MPI.COMM_WORLD
    require_serial(comm)

    xdmf_path = Path(xdmf_path).resolve()
    with io.XDMFFile(comm, str(xdmf_path), "r") as xdmf:
        msh = xdmf.read_mesh(name="mesh")

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = fem.Function(V, name=field_name)

    h5_file, h5_dataset = parse_h5_target_from_xdmf(xdmf_path, field_name)
    with h5py.File(h5_file, "r") as h5:
        data = np.array(h5[h5_dataset])

    data = np.asarray(data).reshape(-1)

    if data.size != u.x.array.size:
        raise RuntimeError(
            f"Dataset size mismatch for field '{field_name}'. "
            f"HDF5 has {data.size} entries, but reconstructed CG1 function has {u.x.array.size} dofs."
        )

    u.x.array[:] = data.astype(u.x.array.dtype, copy=False)
    u.x.scatter_forward()
    return msh, V, u, h5_file, h5_dataset


def sample_function_at_points(u: fem.Function, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate function u at physical points.

    Returns
    -------
    values : (n,) ndarray
        Sampled values, NaN where outside mesh.
    inside : (n,) bool ndarray
        Whether each point was found inside a cell.
    """
    msh = u.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    inside = np.zeros(points.shape[0], dtype=bool)
    cells = np.full(points.shape[0], -1, dtype=np.int32)

    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            inside[i] = True
            cells[i] = links[0]

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if np.any(inside):
        vals = u.eval(points[inside], cells[inside])
        values[inside] = np.asarray(vals).reshape(-1)

    return values, inside


def build_reference_point_data(V_ref, u_ref: fem.Function) -> tuple[np.ndarray, np.ndarray]:
    """
    For scalar CG1, compare on reference mesh dof coordinates.
    """
    coords = V_ref.tabulate_dof_coordinates()
    coords = np.asarray(coords, dtype=np.float64).reshape((-1, coords.shape[-1]))
    vals = np.asarray(u_ref.x.array, dtype=np.float64).copy()

    if coords.shape[0] != vals.size:
        raise RuntimeError(
            f"Reference CG1 coordinate/value mismatch: {coords.shape[0]} coords vs {vals.size} values."
        )

    return coords, vals


def compute_stats(ref_vals: np.ndarray, cmp_vals: np.ndarray, rel_floor_frac: float) -> dict:
    valid = np.isfinite(ref_vals) & np.isfinite(cmp_vals)
    if not np.any(valid):
        raise RuntimeError("No valid comparison points were found.")

    ref = ref_vals[valid]
    cmpv = cmp_vals[valid]

    abs_err = np.abs(cmpv - ref)

    ref_scale = max(float(np.max(np.abs(ref))), 1.0)
    denom_floor = rel_floor_frac * ref_scale
    denom = np.maximum(np.abs(ref), denom_floor)
    rel_err = abs_err / denom

    l2_abs = math.sqrt(float(np.mean(abs_err**2)))
    l2_rel = math.sqrt(float(np.mean(rel_err**2)))

    global_rel_l2 = float(np.linalg.norm(cmpv - ref) / max(np.linalg.norm(ref), denom_floor))

    stats = {
        "n_valid": int(valid.sum()),
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "rms_abs_error": l2_abs,
        "max_rel_error": float(np.max(rel_err)),
        "mean_rel_error": float(np.mean(rel_err)),
        "rms_rel_error": l2_rel,
        "p95_rel_error": float(np.percentile(rel_err, 95.0)),
        "global_rel_l2_discrete": global_rel_l2,
        "ref_scale_used": ref_scale,
        "denom_floor_used": denom_floor,
        "valid_mask": valid,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }
    return stats


def write_csv(
    csv_path: Path,
    points: np.ndarray,
    ref_vals: np.ndarray,
    cmp_vals: np.ndarray,
    valid_mask: np.ndarray,
    abs_err: np.ndarray,
    rel_err: np.ndarray,
) -> None:
    valid_points = points[valid_mask]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "reference_value", "compared_value", "abs_error", "rel_error"])
        j = 0
        for i in range(points.shape[0]):
            if valid_mask[i]:
                p = valid_points[j]
                writer.writerow([p[0], p[1], p[2], ref_vals[i], cmp_vals[i], abs_err[j], rel_err[j]])
                j += 1


def write_error_xdmf(
    outdir: Path,
    basename: str,
    msh_ref,
    V_ref,
    ref_vals: np.ndarray,
    cmp_vals: np.ndarray,
    valid_mask: np.ndarray,
    abs_err_valid: np.ndarray,
    rel_err_valid: np.ndarray,
) -> Path:
    cmp_on_ref = fem.Function(V_ref, name="compared_on_reference_points")
    abs_error = fem.Function(V_ref, name="abs_error")
    rel_error = fem.Function(V_ref, name="rel_error")
    valid_points = fem.Function(V_ref, name="valid_points")

    cmp_arr = np.full(ref_vals.shape, np.nan, dtype=np.float64)
    abs_arr = np.full(ref_vals.shape, np.nan, dtype=np.float64)
    rel_arr = np.full(ref_vals.shape, np.nan, dtype=np.float64)
    valid_arr = np.zeros(ref_vals.shape, dtype=np.float64)

    cmp_arr[valid_mask] = cmp_vals[valid_mask]
    abs_arr[valid_mask] = abs_err_valid
    rel_arr[valid_mask] = rel_err_valid
    valid_arr[valid_mask] = 1.0

    cmp_on_ref.x.array[:] = cmp_arr
    abs_error.x.array[:] = abs_arr
    rel_error.x.array[:] = rel_arr
    valid_points.x.array[:] = valid_arr

    cmp_on_ref.x.scatter_forward()
    abs_error.x.scatter_forward()
    rel_error.x.scatter_forward()
    valid_points.x.scatter_forward()

    xdmf_path = outdir / f"{basename}_pointwise_error.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh_ref)
        xdmf.write_function(cmp_on_ref)
        xdmf.write_function(abs_error)
        xdmf.write_function(rel_error)
        xdmf.write_function(valid_points)
    return xdmf_path


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    require_serial(comm)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    msh_a, V_a, u_a, h5_a, dset_a = load_cg1_function_from_xdmf(args.xdmf_a, args.field)
    msh_b, V_b, u_b, h5_b, dset_b = load_cg1_function_from_xdmf(args.xdmf_b, args.field)

    if args.reference == "a":
        msh_ref, V_ref, u_ref = msh_a, V_a, u_a
        u_cmp = u_b
        ref_label = "a"
        cmp_label = "b"
    else:
        msh_ref, V_ref, u_ref = msh_b, V_b, u_b
        u_cmp = u_a
        ref_label = "b"
        cmp_label = "a"

    ref_points, ref_vals = build_reference_point_data(V_ref, u_ref)
    cmp_vals, inside_cmp = sample_function_at_points(u_cmp, ref_points)

    n_inside = int(np.count_nonzero(inside_cmp))
    n_total = int(ref_points.shape[0])

    stats = compute_stats(ref_vals, cmp_vals, args.rel_floor)

    summary = {
        "field": args.field,
        "reference_file": str(args.xdmf_a if args.reference == "a" else args.xdmf_b),
        "compared_file": str(args.xdmf_b if args.reference == "a" else args.xdmf_a),
        "reference_label": ref_label,
        "compared_label": cmp_label,
        "reference_h5_dataset": f"{h5_a}:{dset_a}" if args.reference == "a" else f"{h5_b}:{dset_b}",
        "compared_h5_dataset": f"{h5_b}:{dset_b}" if args.reference == "a" else f"{h5_a}:{dset_a}",
        "n_reference_points": n_total,
        "n_points_inside_compared_mesh": n_inside,
        "coverage_fraction": float(n_inside / max(n_total, 1)),
        "max_abs_error": stats["max_abs_error"],
        "mean_abs_error": stats["mean_abs_error"],
        "rms_abs_error": stats["rms_abs_error"],
        "max_rel_error": stats["max_rel_error"],
        "mean_rel_error": stats["mean_rel_error"],
        "rms_rel_error": stats["rms_rel_error"],
        "p95_rel_error": stats["p95_rel_error"],
        "global_rel_l2_discrete": stats["global_rel_l2_discrete"],
        "ref_scale_used": stats["ref_scale_used"],
        "denom_floor_used": stats["denom_floor_used"],
    }

    csv_path = outdir / f"{args.basename}_pointwise.csv"
    write_csv(
        csv_path,
        ref_points,
        ref_vals,
        cmp_vals,
        stats["valid_mask"],
        stats["abs_err"],
        stats["rel_err"],
    )

    error_xdmf = write_error_xdmf(
        outdir,
        args.basename,
        msh_ref,
        V_ref,
        ref_vals,
        cmp_vals,
        stats["valid_mask"],
        stats["abs_err"],
        stats["rel_err"],
    )

    summary["csv"] = str(csv_path)
    summary["error_xdmf"] = str(error_xdmf)

    summary_json = outdir / f"{args.basename}_summary.json"
    summary_txt = outdir / f"{args.basename}_summary.txt"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(summary_txt, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n=== Pointwise comparison summary ===")
    print(f"field                      : {args.field}")
    print(f"reference                  : {summary['reference_file']}")
    print(f"compared                   : {summary['compared_file']}")
    print(f"reference points           : {summary['n_reference_points']}")
    print(f"inside compared mesh       : {summary['n_points_inside_compared_mesh']}")
    print(f"coverage fraction          : {summary['coverage_fraction']:.12e}")
    print()
    print(f"max abs error              : {summary['max_abs_error']:.12e}")
    print(f"mean abs error             : {summary['mean_abs_error']:.12e}")
    print(f"rms abs error              : {summary['rms_abs_error']:.12e}")
    print()
    print(f"max rel error              : {summary['max_rel_error']:.12e}")
    print(f"mean rel error             : {summary['mean_rel_error']:.12e}")
    print(f"rms rel error              : {summary['rms_rel_error']:.12e}")
    print(f"p95 rel error              : {summary['p95_rel_error']:.12e}")
    print(f"global rel L2 (discrete)   : {summary['global_rel_l2_discrete']:.12e}")
    print()
    print("=== Wrote files ===")
    print(f"  csv                      : {csv_path}")
    print(f"  error xdmf               : {error_xdmf}")
    print(f"  summary txt              : {summary_txt}")
    print(f"  summary json             : {summary_json}")


if __name__ == "__main__":
    main()
