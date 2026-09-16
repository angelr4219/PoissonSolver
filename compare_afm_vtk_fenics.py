#!/usr/bin/env python3

"""
Compare the AFM-tip + Si/SiGe double-well FEniCSx solution
(sige_afm_tip_two_probes.py) against a reference VTK (e.g. Leah's
basePotential3d(1).vtk). The VTK is treated as ground truth, per the
established MaSQE/FEniCS validation workflow in this repo.

Run inspect_base_vtk.py on the VTK FIRST -- this script assumes you
already know its coordinate convention (z=0 = sample surface here,
confirmed for basePotential3d(1).vtk) and bottom-face voltage.

Method: evaluate the live FEniCS Function directly at the VTK's own
native grid points via DOLFINx mesh collision detection -- not
scipy.griddata on already-exported nodal values. This avoids
convex-hull/nearest-neighbor fallback entirely: either a VTK point
falls inside a mesh cell (exact FEM evaluation) or it's outside the
FEM domain (reported, not silently guessed).

Outputs (under --outdir):
  solved_vs_reference.xdmf   ONE combined file: phi, phi_reference,
                              diff, relative_error, relative_permittivity,
                              material_id, facet_tags -- open once in
                              ParaView, color by whichever field you want.
  centerline_comparison.png  phi(0,0,z): reference vs FEM vs diff
  slice_z###nm.png           per z-slice: raw ref, raw FEM, diff, rel err
                              (matched color limits within each pair)
  compare_stats.json         global + per-slice error stats (mean/RMS/
                              median/95th-pct/max, masked + unmasked)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, io
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sige_afm_tip_two_probes import (
    build_geometry, solve as tip_solve,
    FACET_TIP, FACET_BOTTOM, FACET_PROBE1, FACET_PROBE2, FACET_OUTER,
)

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def read_vtk_rectilinear(path):
    """Legacy ASCII VTK RECTILINEAR_GRID reader (single scalar field)."""
    with open(path) as f:
        lines = f.readlines()
    i = [0]

    def nxt():
        l = lines[i[0]]
        i[0] += 1
        return l

    nxt()  # vtk DataFile Version
    nxt()  # title
    nxt()  # ASCII
    nxt()  # DATASET RECTILINEAR_GRID
    nx, ny, nz = (int(v) for v in nxt().split()[1:4])

    def read_coords(n_expected):
        header = nxt().split()
        n = int(header[1])
        assert n == n_expected
        vals = []
        while len(vals) < n:
            vals.extend(float(v) for v in nxt().split())
        return np.array(vals, dtype=np.float64)

    x = read_coords(nx)
    y = read_coords(ny)
    z = read_coords(nz)

    npts = int(nxt().split()[1])  # POINT_DATA n
    field_name = nxt().split()[1]  # SCALARS name float
    nxt()  # LOOKUP_TABLE default

    vals = []
    while len(vals) < npts:
        vals.extend(float(v) for v in nxt().split())
    vals = np.array(vals, dtype=np.float64)
    data = vals.reshape((nz, ny, nx))  # VTK order: x fastest, then y, then z
    return x, y, z, data, field_name


def eval_at_points(domain, phi, points):
    bb = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)
    out = np.full(points.shape[0], np.nan)
    cells, idx_on_proc = [], []
    for i in range(points.shape[0]):
        links_i = colliding.links(i)
        if len(links_i) > 0:
            idx_on_proc.append(i)
            cells.append(links_i[0])
    if idx_on_proc:
        pts = points[idx_on_proc]
        vals = phi.eval(pts, np.array(cells, dtype=np.int32))[:, 0]
        for local_i, gi in enumerate(idx_on_proc):
            out[gi] = vals[local_i]
    return out


def error_stats(diff, rel_err, mask):
    finite = np.isfinite(diff)
    finite_masked = finite & mask
    out = {
        "n_points": int(finite.sum()),
        "n_points_above_floor": int(finite_masked.sum()),
        "abs_error": {
            "mean_V": float(np.nanmean(np.abs(diff[finite]))) if finite.any() else None,
            "rms_V": float(np.sqrt(np.nanmean(diff[finite] ** 2))) if finite.any() else None,
            "max_V": float(np.nanmax(np.abs(diff[finite]))) if finite.any() else None,
        },
        "rel_error_masked": {
            "mean": float(np.nanmean(rel_err[finite_masked])) if finite_masked.any() else None,
            "median": float(np.nanmedian(rel_err[finite_masked])) if finite_masked.any() else None,
            "p95": float(np.nanpercentile(rel_err[finite_masked], 95)) if finite_masked.any() else None,
            "max": float(np.nanmax(rel_err[finite_masked])) if finite_masked.any() else None,
        },
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vtk", default=str(Path.home() / "Downloads" / "basePotential3d(1).vtk"))

    p.add_argument("--lx", type=float, default=300.0)
    p.add_argument("--ly", type=float, default=300.0)
    p.add_argument("--air-height", type=float, default=260.0,
                   help="Must exceed gap+2*tip_radius+cone_height+shaft_height or the shaft gets clipped")
    p.add_argument("--eps-air", type=float, default=1.0)
    p.add_argument("--eps-si", type=float, default=11.7)
    p.add_argument("--eps-sige", type=float, default=12.0)
    p.add_argument("--gap", type=float, default=30.0,
                   help="Tip apex height above sample surface [nm]")
    p.add_argument("--tip-radius", type=float, default=20.0)
    p.add_argument("--cone-height", type=float, default=100.0)
    p.add_argument("--shank-radius", type=float, default=60.0)
    p.add_argument("--shaft-radius", type=float, default=60.0,
                   help="Should match --shank-radius to connect flush")
    p.add_argument("--shaft-height", type=float, default=60.0)
    p.add_argument("--tip-voltage", type=float, default=1.0,
                   help="NOT yet calibrated to the reference's actual tip bias -- treat as free parameter")
    p.add_argument("--bottom-voltage", type=float, default=-4.4,
                   help="Should match the VTK's own bottom face -- check with inspect_base_vtk.py first")
    p.add_argument("--probe1-radius", type=float, default=10.0)
    p.add_argument("--probe2-radius", type=float, default=10.0)
    p.add_argument("--h-apex", type=float, default=1.0)
    p.add_argument("--h-device", type=float, default=2.0)
    p.add_argument("--h-near", type=float, default=5.0)
    p.add_argument("--h-bottom", type=float, default=100.0)
    p.add_argument("--degree", type=int, default=1)

    p.add_argument("--rel-err-floor-frac", type=float, default=0.01,
                   help="Relative error is only computed where |vtk phi| exceeds this fraction "
                        "of the vtk field's peak |value| -- avoids meaningless huge percentages "
                        "from dividing by near-zero reference values")
    p.add_argument("--slices-nm", nargs="+", type=float,
                   default=[0, 2, 30, 35, 40, 43, 48, 53])
    p.add_argument("--outdir", default="Results/compare_afm_vtk_fenics")
    args = p.parse_args()

    outdir = Path(args.outdir)
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    # ------------------------------------------------------------------
    # Load the reference VTK
    # ------------------------------------------------------------------
    x_nm, y_nm, z_nm, vtk_data, field_name = read_vtk_rectilinear(args.vtk)
    if RANK == 0:
        print(f"[VTK] {args.vtk}: field={field_name}, "
              f"grid {len(x_nm)}x{len(y_nm)}x{len(z_nm)}, "
              f"X[{x_nm.min():.0f},{x_nm.max():.0f}] Y[{y_nm.min():.0f},{y_nm.max():.0f}] "
              f"Z[{z_nm.min():.0f},{z_nm.max():.0f}] nm, "
              f"vtk phi range [{vtk_data.min():.4f},{vtk_data.max():.4f}] V")
        print("      (run inspect_base_vtk.py on this file if you haven't confirmed its "
              "coordinate convention and bottom-face voltage)")

    # ------------------------------------------------------------------
    # Build + solve the AFM-tip + Si/SiGe FEM geometry (nm coordinates).
    # ------------------------------------------------------------------
    domain, cell_tags, facet_tags, z_probe1, z_probe2, z_bottom = build_geometry(args, COMM)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)

    if RANK == 0:
        print("[FEM] Facet tag integrity (confirms tip/gates/bottom actually exist in the mesh):")
    for name, marker in [("AFM tip", FACET_TIP), ("Bottom", FACET_BOTTOM),
                          ("Probe 1 (gate)", FACET_PROBE1), ("Probe 2 (gate)", FACET_PROBE2),
                          ("Outer", FACET_OUTER)]:
        count = COMM.allreduce(len(facet_tags.find(marker)), op=MPI.SUM)
        if RANK == 0:
            print(f"    {name:16s} tag={marker:3d} facets={count:,}")
        if count == 0:
            raise RuntimeError(f"{name} has zero facets -- geometry/tagging bug, not a solver issue.")

    phi, epsilon, material_id, V, problem = tip_solve(domain, cell_tags, facet_tags, args)

    if RANK == 0:
        print(f"[FEM] tip_voltage={args.tip_voltage}V bottom_voltage={args.bottom_voltage}V "
              f"gap={args.gap}nm z_bottom={z_bottom}nm")

    # ------------------------------------------------------------------
    # Reference field interpolated onto the FEM mesh's own dof coords,
    # so phi / phi_reference / diff / relative_error / material fields
    # / facet_tags can all be viewed together in ONE ParaView file.
    # ------------------------------------------------------------------
    vtk_interp = RegularGridInterpolator(
        (z_nm, y_nm, x_nm), vtk_data, bounds_error=False, fill_value=np.nan,
    )
    dof_xyz_fem = V.tabulate_dof_coordinates()
    phi_ref_on_fem = vtk_interp(dof_xyz_fem[:, [2, 1, 0]])  # (z,y,x) order

    phi_ref_fn = fem.Function(V, name="phi_reference")
    phi_ref_fn.x.array[:] = np.nan_to_num(phi_ref_on_fem, nan=0.0).astype(PETSc.ScalarType)

    diff_fn = fem.Function(V, name="diff")
    diff_fn.x.array[:] = phi.x.array - phi_ref_fn.x.array

    relerr_fn = fem.Function(V, name="relative_error")
    denom = np.abs(phi_ref_fn.x.array)
    floor = args.rel_err_floor_frac * np.abs(vtk_data).max()
    relerr_fn.x.array[:] = np.abs(diff_fn.x.array) / np.where(denom > floor, denom, np.nan)

    facet_tags.name = "facet_tags"
    combined_path = outdir / "solved_vs_reference.xdmf"
    with io.XDMFFile(COMM, combined_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(phi_ref_fn)
        xdmf.write_function(diff_fn)
        xdmf.write_function(relerr_fn)
        xdmf.write_function(epsilon)
        xdmf.write_function(material_id)
        try:
            xdmf.write_meshtags(facet_tags, domain.geometry)
        except TypeError:
            xdmf.write_meshtags(facet_tags)

    if RANK == 0:
        print(f"[FEM] Wrote {combined_path} -- phi, phi_reference, diff, relative_error, "
              f"relative_permittivity, material_id, facet_tags, all in one file.")

    # ------------------------------------------------------------------
    # Evaluate FEM directly at the VTK's own native grid points (exact
    # DOLFINx collision-detection evaluation, no griddata/nearest-fallback).
    # ------------------------------------------------------------------
    X, Y, Z = np.meshgrid(x_nm, y_nm, z_nm, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    fem_vals_flat = eval_at_points(domain, phi, points)
    fem_vals = fem_vals_flat.reshape(len(x_nm), len(y_nm), len(z_nm))  # (nx,ny,nz)
    fem_vals = np.transpose(fem_vals, (2, 1, 0))  # -> (nz,ny,nx), matches vtk_data

    if RANK != 0:
        return

    diff = fem_vals - vtk_data
    rel_err = np.abs(diff) / np.abs(vtk_data)
    floor = args.rel_err_floor_frac * np.abs(vtk_data).max()
    mask = np.abs(vtk_data) > floor

    np.savez(outdir / "comparison_arrays.npz",
             x_nm=x_nm, y_nm=y_nm, z_nm=z_nm,
             vtk_data=vtk_data, fem_vals=fem_vals, diff=diff, rel_err=rel_err, mask=mask)

    n_outside = int((~np.isfinite(diff)).sum())
    stats = {
        "vtk_file": args.vtk,
        "tip_voltage_V": args.tip_voltage,
        "bottom_voltage_V": args.bottom_voltage,
        "gap_nm": args.gap,
        "rel_err_floor_V": float(floor),
        "n_points_total": int(diff.size),
        "n_points_outside_fem_domain": n_outside,
        "global": error_stats(diff, rel_err, mask),
        "per_slice": {},
    }

    cx = len(x_nm) // 2  # x=0
    cy = len(y_nm) // 2  # y=0

    # ------------------------------------------------------------------
    # Centerline comparison: phi(0,0,z) reference vs FEM vs diff, from
    # the AFM tip apex through both probe depths to the back gate.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(z_nm, vtk_data[:, cy, cx], label="Reference (VTK)", lw=2)
    axes[0].plot(z_nm, fem_vals[:, cy, cx], label="FEniCSx", lw=2, ls="--")
    for zline, name in [(z_probe1, "Probe 1"), (z_probe2, "Probe 2")]:
        axes[0].axvline(zline, color="gray", ls=":", lw=1)
        axes[0].text(zline, axes[0].get_ylim()[1], name, fontsize=8, ha="center", va="bottom")
    axes[0].set_ylabel("phi [V]")
    axes[0].set_title("Centerline phi(0,0,z): tip apex -> surface -> device -> back gate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(z_nm, fem_vals[:, cy, cx] - vtk_data[:, cy, cx], color="tab:red")
    axes[1].set_xlabel("z [nm]")
    axes[1].set_ylabel("FEM - Reference [V]")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "centerline_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-slice 4-panel comparison plots, matched color limits within
    # each pair so visual agreement corresponds to numerical agreement.
    # ------------------------------------------------------------------
    for z_target in args.slices_nm:
        iz = int(np.argmin(np.abs(z_nm - z_target)))
        z_actual = z_nm[iz]

        vtk_slice = vtk_data[iz]
        fem_slice = fem_vals[iz]
        diff_slice = diff[iz]
        relerr_slice = rel_err[iz]
        mask_slice = mask[iz]

        stats["per_slice"][f"z{z_actual:.1f}nm"] = error_stats(diff_slice, relerr_slice, mask_slice)

        vmin, vmax = np.nanmin(vtk_slice), np.nanmax(vtk_slice)
        finite_slice = np.isfinite(diff_slice)
        dmax = np.nanmax(np.abs(diff_slice)) if finite_slice.any() else 1.0
        extent = [x_nm.min(), x_nm.max(), y_nm.min(), y_nm.max()]

        relerr_masked = np.where(mask_slice, relerr_slice, np.nan) * 100

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, data, title, vlo, vhi, cmap in [
            (axes[0], vtk_slice, "Reference (VTK)", vmin, vmax, "viridis"),
            (axes[1], fem_slice, "FEniCSx", vmin, vmax, "viridis"),
            (axes[2], diff_slice, "FEM - Reference", -dmax, dmax, "RdBu_r"),
            (axes[3], relerr_masked, "Relative error [%] (masked)", 0, None, "magma"),
        ]:
            im = ax.imshow(data, origin="lower", extent=extent, aspect="equal",
                            vmin=vlo, vmax=vhi, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("x [nm]")
            ax.set_ylabel("y [nm]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"z = {z_actual:.1f} nm")
        fig.tight_layout()
        fig.savefig(outdir / f"slice_z{z_actual:.0f}nm.png", dpi=150)
        plt.close(fig)

    with open(outdir / "compare_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    g = stats["global"]
    print("=" * 78)
    print("COMPARISON SUMMARY")
    print("=" * 78)
    print(f"  VTK points               : {stats['n_points_total']:,}")
    print(f"  FEM-evaluated points     : {stats['n_points_total'] - n_outside:,}")
    print(f"  points outside FEM       : {n_outside:,}")
    print(f"  rel-err floor            : |ref| > {floor:.4f} V "
          f"({args.rel_err_floor_frac*100:.1f}% of peak |ref|)")
    print()
    print("  absolute error:")
    print(f"      mean                  : {g['abs_error']['mean_V']:.4f} V")
    print(f"      RMS                   : {g['abs_error']['rms_V']:.4f} V")
    print(f"      maximum               : {g['abs_error']['max_V']:.4f} V")
    print()
    print("  relative error (masked, |ref| above floor):")
    print(f"      mean                  : {g['rel_error_masked']['mean']*100:.2f} %")
    print(f"      median                : {g['rel_error_masked']['median']*100:.2f} %")
    print(f"      95th percentile       : {g['rel_error_masked']['p95']*100:.2f} %")
    print(f"      maximum               : {g['rel_error_masked']['max']*100:.2f} %")
    print()
    print(f"  potential range: VTK [{vtk_data.min():.4f},{vtk_data.max():.4f}] V, "
          f"FEniCS [{np.nanmin(fem_vals):.4f},{np.nanmax(fem_vals):.4f}] V")
    print()
    print("NOTE: --tip-voltage is not yet calibrated against the reference's actual AFM bias.")
    print("Do not conclude anything about mesh/geometry accuracy until the centerline and")
    print("z-slice comparisons show voltage-independent shape agreement.")
    print(f"Wrote: {outdir}/solved_vs_reference.xdmf, centerline_comparison.png, "
          f"slice_z###nm.png, compare_stats.json")


if __name__ == "__main__":
    main()
