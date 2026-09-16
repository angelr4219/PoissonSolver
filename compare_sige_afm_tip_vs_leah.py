#!/usr/bin/env python3

"""
Compare the AFM-tip + Si/SiGe double-well FEniCSx solution
(sige_afm_tip_two_probes.py) against Leah's MaSQE reference
(basePotential3d(1).vtk).

Leah's benchmark (per her email): top well at 30 nm depth, well width
10 nm -> compare potential linecuts at a depth of 35 nm. That is
EXACTLY sige_afm_tip_two_probes.py's Layer 3 (Si, z=30->40 nm) and
Probe 1 (z=35 nm) -- confirmed by inspecting the VTK directly: grid is
300x300nm laterally (x,y in [-150,150]nm) and z in [0,2053]nm with
0.5nm spacing around 25-58nm, matching this repo's six-layer stack
exactly. The VTK's bottom face (z=2053nm) is uniformly -4.4V, and the
top face (z=0) is peaked at the center (~-1.235V) decaying to ~-0.06V
at the lateral edges -- the signature of a real 3D AFM tip in air
above the sample, not a flat top BC. Match --bottom-voltage to -4.4
when calling this script; --tip-voltage is NOT yet calibrated to
Leah's actual tip bias and should be treated as a free parameter to
fit against this comparison's residual.

Method (same as compare_basepotential_vs_masqe.py): evaluate the FEM
solution directly at the VTK's own native grid points (no
interpolation error on the FEM side), diff = FEM - VTK, relative
error = |FEM - VTK| / |VTK|.

Outputs (under --outdir):
  linecut_z035nm.png     the linecut Leah explicitly asked for
  slice_z###nm.png       per-slice 4-panel comparison (raw VTK, raw FEM, diff, rel err)
  compare_stats.json     global + per-slice error stats
  comparison_arrays.npz  raw grids for further analysis
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
    """Legacy ASCII VTK RECTILINEAR_GRID reader (same format/parser as
    POISSONSOLVER/compare_basepotential_vs_masqe.py's read_vtk_rectilinear,
    duplicated here so this script has no cross-repo import dependency)."""
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


def main():
    # Re-declares sige_afm_tip_two_probes.py's exact geometry/solver flags
    # (same names/defaults) plus this script's own comparison flags, since
    # build_geometry()/solve() there expect an argparse Namespace with
    # those specific attribute names.
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
                   help="NOT yet calibrated to Leah's actual tip bias -- treat as free parameter")
    p.add_argument("--bottom-voltage", type=float, default=-4.4,
                   help="Matches the VTK's own bottom face (-4.4V, read directly from basePotential3d(1).vtk)")
    p.add_argument("--probe1-radius", type=float, default=10.0)
    p.add_argument("--probe2-radius", type=float, default=10.0)
    p.add_argument("--h-apex", type=float, default=1.0)
    p.add_argument("--h-device", type=float, default=2.0)
    p.add_argument("--h-near", type=float, default=5.0)
    p.add_argument("--h-bottom", type=float, default=100.0)
    p.add_argument("--degree", type=int, default=1)

    p.add_argument("--slices-nm", nargs="+", type=float,
                   default=[0, 10, 20, 30, 35, 40, 48, 53, 99])
    p.add_argument("--outdir", default="Results/compare_afm_tip_vs_leah")
    args = p.parse_args()

    outdir = Path(args.outdir)
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    COMM.barrier()

    # ------------------------------------------------------------------
    # Load Leah's reference VTK
    # ------------------------------------------------------------------
    x_nm, y_nm, z_nm, vtk_data, field_name = read_vtk_rectilinear(args.vtk)
    if RANK == 0:
        print(f"[VTK] {args.vtk}: field={field_name}, "
              f"grid {len(x_nm)}x{len(y_nm)}x{len(z_nm)}, "
              f"X[{x_nm.min():.0f},{x_nm.max():.0f}] Y[{y_nm.min():.0f},{y_nm.max():.0f}] "
              f"Z[{z_nm.min():.0f},{z_nm.max():.0f}] nm, "
              f"vtk phi range [{vtk_data.min():.4f},{vtk_data.max():.4f}] V")

    # ------------------------------------------------------------------
    # Build + solve the AFM-tip + Si/SiGe FEM geometry (nm coordinates,
    # matches sige_afm_tip_two_probes.py exactly).
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
    # Write the raw FEM solution to XDMF/H5 FIRST, before any comparison,
    # so it can be opened and inspected directly in ParaView (geometry,
    # tip position, material layers, solved potential) independent of
    # whether the voltage calibration against Leah's VTK is right yet.
    # ------------------------------------------------------------------
    potential_path = outdir / "potential.xdmf"
    with io.XDMFFile(COMM, potential_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)

    materials_path = outdir / "materials.xdmf"
    with io.XDMFFile(COMM, materials_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(epsilon)
        xdmf.write_function(material_id)

    facet_tags.name = "facet_tags"
    facet_tags_path = outdir / "facet_tags.xdmf"
    with io.XDMFFile(COMM, facet_tags_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        try:
            xdmf.write_meshtags(facet_tags, domain.geometry)
        except TypeError:
            xdmf.write_meshtags(facet_tags)

    if RANK == 0:
        print(f"[FEM] Wrote {potential_path}, {materials_path}, {facet_tags_path}")
        print("      Open potential.xdmf in ParaView to inspect phi (and the tip/geometry)")
        print("      before trusting any comparison numbers below.")

    # ------------------------------------------------------------------
    # Also write Leah's VTK potential (linearly interpolated onto the
    # FEM mesh's own dof coordinates) plus diff/relative_error as real
    # fields on the SAME mesh, so both solutions and their disagreement
    # can be viewed side-by-side / sliced continuously in ParaView --
    # same pattern as POISSONSOLVER/compare_basepotential_vs_masqe.py.
    # ------------------------------------------------------------------
    vtk_interp = RegularGridInterpolator(
        (z_nm, y_nm, x_nm), vtk_data, bounds_error=False, fill_value=np.nan,
    )
    dof_xyz_fem = V.tabulate_dof_coordinates()  # nm, (x,y,z) columns
    phi_masqe_on_fem = vtk_interp(dof_xyz_fem[:, [2, 1, 0]])  # (z,y,x) order

    phi_masqe_fn = fem.Function(V, name="phi_masqe")
    phi_masqe_fn.x.array[:] = np.nan_to_num(phi_masqe_on_fem, nan=0.0).astype(PETSc.ScalarType)

    diff_fn = fem.Function(V, name="diff")
    diff_fn.x.array[:] = phi.x.array - phi_masqe_fn.x.array

    relerr_fn = fem.Function(V, name="relative_error")
    denom = np.abs(phi_masqe_fn.x.array)
    relerr_fn.x.array[:] = np.abs(diff_fn.x.array) / np.where(denom > 1e-6, denom, np.nan)

    comparison_path = outdir / "comparison.xdmf"
    with io.XDMFFile(COMM, comparison_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(phi_masqe_fn)
        xdmf.write_function(diff_fn)
        xdmf.write_function(relerr_fn)

    if RANK == 0:
        print(f"[FEM] Wrote {comparison_path} -- phi, phi_masqe (Leah's VTK interpolated "
              f"onto this mesh), diff, relative_error, all viewable/sliceable in ParaView.")

    # ------------------------------------------------------------------
    # Evaluate FEM directly at the VTK's own native grid points.
    # Domain here is in nm (unlike compare_basepotential_vs_masqe.py,
    # which rescales to meters) -- sige_afm_tip_two_probes.py builds its
    # gmsh geometry in raw nm, so evaluate points in nm too.
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

    np.savez(outdir / "comparison_arrays.npz",
             x_nm=x_nm, y_nm=y_nm, z_nm=z_nm,
             vtk_data=vtk_data, fem_vals=fem_vals, diff=diff, rel_err=rel_err)

    finite = np.isfinite(diff)
    stats = {
        "vtk_file": args.vtk,
        "tip_voltage_V": args.tip_voltage,
        "bottom_voltage_V": args.bottom_voltage,
        "gap_nm": args.gap,
        "n_points_compared": int(finite.sum()),
        "n_points_outside_fem_domain": int((~finite).sum()),
        "global": {
            "max_abs_diff_V": float(np.nanmax(np.abs(diff))),
            "mean_abs_diff_V": float(np.nanmean(np.abs(diff))),
            "rmse_V": float(np.sqrt(np.nanmean(diff[finite] ** 2))),
            "max_rel_err": float(np.nanmax(rel_err[finite])),
            "mean_rel_err": float(np.nanmean(rel_err[finite])),
        },
        "per_slice": {},
    }

    cx = len(x_nm) // 2  # x=0
    cy = len(y_nm) // 2  # y=0

    # ------------------------------------------------------------------
    # The linecut Leah explicitly asked for: phi(x, y=0, z=35nm)
    # ------------------------------------------------------------------
    iz35 = int(np.argmin(np.abs(z_nm - 35.0)))
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(x_nm, vtk_data[iz35, cy, :], label="MaSQE (Leah)", lw=2)
    axes[0].plot(x_nm, fem_vals[iz35, cy, :], label="FEniCSx", lw=2, ls="--")
    axes[0].set_ylabel("phi [V]")
    axes[0].set_title(f"Linecut at z={z_nm[iz35]:.1f} nm, y=0 (top well, per Leah's request)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_nm, fem_vals[iz35, cy, :] - vtk_data[iz35, cy, :], color="tab:red")
    axes[1].set_xlabel("x [nm]")
    axes[1].set_ylabel("FEM - MaSQE [V]")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "linecut_z035nm.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-slice 4-panel comparison plots
    # ------------------------------------------------------------------
    for z_target in args.slices_nm:
        iz = int(np.argmin(np.abs(z_nm - z_target)))
        z_actual = z_nm[iz]

        vtk_slice = vtk_data[iz]
        fem_slice = fem_vals[iz]
        diff_slice = diff[iz]
        relerr_slice = rel_err[iz]

        finite_slice = np.isfinite(diff_slice)
        stats["per_slice"][f"z{z_actual:.1f}nm"] = {
            "max_abs_diff_V": float(np.nanmax(np.abs(diff_slice))) if finite_slice.any() else None,
            "mean_abs_diff_V": float(np.nanmean(np.abs(diff_slice))) if finite_slice.any() else None,
            "max_rel_err": float(np.nanmax(relerr_slice[finite_slice])) if finite_slice.any() else None,
            "mean_rel_err": float(np.nanmean(relerr_slice[finite_slice])) if finite_slice.any() else None,
        }

        vmin, vmax = np.nanmin(vtk_slice), np.nanmax(vtk_slice)
        dmax = np.nanmax(np.abs(diff_slice)) if finite_slice.any() else 1.0
        extent = [x_nm.min(), x_nm.max(), y_nm.min(), y_nm.max()]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, data, title, vlo, vhi, cmap in [
            (axes[0], vtk_slice, "MaSQE (Leah)", vmin, vmax, "viridis"),
            (axes[1], fem_slice, "FEniCSx", vmin, vmax, "viridis"),
            (axes[2], diff_slice, "FEM - MaSQE", -dmax, dmax, "RdBu_r"),
            (axes[3], relerr_slice * 100, "Relative error [%]", 0, None, "magma"),
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

    print("=" * 78)
    print("COMPARISON RESULTS (AFM tip + Si/SiGe vs Leah's MaSQE VTK)")
    print("=" * 78)
    print(f"  max_rel_err  = {stats['global']['max_rel_err']*100:.2f} %")
    print(f"  mean_rel_err = {stats['global']['mean_rel_err']*100:.2f} %")
    print(f"  mean_abs_diff = {stats['global']['mean_abs_diff_V']:.4f} V")
    print(f"  max_abs_diff  = {stats['global']['max_abs_diff_V']:.4f} V")
    print()
    print("NOTE: --tip-voltage is not yet calibrated against Leah's actual AFM bias.")
    print("If error is large and roughly uniform (not concentrated at the tip apex),")
    print("sweep --tip-voltage until the z=35nm linecut center matches MaSQE's value")
    print("before concluding anything about mesh/geometry accuracy.")
    print(f"Wrote: {outdir}/linecut_z035nm.png, slice_z###nm.png, compare_stats.json")


if __name__ == "__main__":
    main()
