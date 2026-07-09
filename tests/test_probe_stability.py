"""
test_probe_stability.py

Track physically meaningful device quantities across mesh refinements.

Global norms do not reveal whether the potential landscape — quantum-dot minima,
inter-dot barriers, gate cross-talk — is converged.  This test extracts those
quantities directly and flags instabilities between mesh strategies.

Same 4-gate geometry as test_four_square_refinement_box.py (standalone, no import).

Geometry (SI internally):
  Box:    Lx=300nm, Ly=300nm, Lz=150nm
  Domain: x in [-150,150]nm, y in [-150,150]nm, z in [0,150]nm
  z=0  : gate plane  (Dirichlet gates)
  z=Lz : bottom contact (phi=0)
  Sides / top outside gates: Neumann

Gates (60x60nm, 30nm gaps):
  G1 centre (-45,-45)nm  +0.25V
  G2 centre (+45,-45)nm  +0.10V
  G3 centre (-45,+45)nm  -0.10V
  G4 centre (+45,+45)nm  +0.25V

Usage:
  python tests/test_probe_stability.py [--quick|--full]
         [--outdir PATH] [--write-xdmf] [--save-csv] [--max-dofs N]
"""
from __future__ import annotations

import sys
import time
import argparse
import csv
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poisson.refinement import RefinementBox, build_refined_mesh_3d
from poisson.io_utils import write_field_xdmf

# ── Geometry constants (SI) ──────────────────────────────────────────────────
NM = 1e-9

Lx = 300 * NM
Ly = 300 * NM
Lz = 150 * NM

W    = 60 * NM   # gate side length
Z_QW = 40 * NM   # quantum-well proxy depth

GATES = [
    (-45 * NM, -45 * NM, +0.25),   # G1
    (+45 * NM, -45 * NM, +0.10),   # G2
    (-45 * NM, +45 * NM, -0.10),   # G3
    (+45 * NM, +45 * NM, +0.25),   # G4
]

REFBOX_CX = 0.0
REFBOX_CY = 0.0
REFBOX_CZ = Lz / 2
REFBOX_LX = 200 * NM
REFBOX_LY = 200 * NM
REFBOX_LZ =  80 * NM

TOL = 1e-12   # geometric tolerance

# Cases: (name, h_coarse, h_fine_box_or_None, in_quick, in_full)
CASE_DEFS = [
    ("A", 20 * NM, None,     True,  True),
    ("B", 10 * NM, None,     True,  True),
    ("C",  5 * NM, None,     False, True),
    ("D", 20 * NM, 10 * NM,  True,  True),
    ("E", 20 * NM,  5 * NM,  True,  True),
]

# Instability thresholds (mode-specific — see main())
WARN_MV_QUICK = 5.0    # quick mode: WARNING if any quantity exceeds this
FAIL_MV_QUICK = 10.0   # quick mode: FAIL (exit 1) if any quantity exceeds this
FAIL_MV_FULL  = 5.0    # full mode:  FAIL (exit 1) if any quantity exceeds this


# ── Mesh / FEM helpers (copied from refinement workflow, standalone) ──────────
def make_boxes(h_fine_box):
    if h_fine_box is None:
        return []
    return [RefinementBox(
        cx=REFBOX_CX, cy=REFBOX_CY, cz=REFBOX_CZ,
        lx=REFBOX_LX, ly=REFBOX_LY, lz=REFBOX_LZ,
        h_fine=h_fine_box,
    )]


def build_mesh(comm, h_coarse, boxes):
    return build_refined_mesh_3d(comm, Lx, Ly, Lz, h_coarse, boxes)


def build_function_space(mesh):
    from dolfinx import fem
    return fem.functionspace(mesh, ("Lagrange", 1))


def build_bcs(mesh, V):
    from dolfinx import fem
    from dolfinx.fem import locate_dofs_geometrical

    bcs = []
    for (x0, y0, voltage) in GATES:
        def gate_marker(x, _x0=x0, _y0=y0):
            return (
                np.isclose(x[2], 0.0, atol=TOL) &
                (x[0] >= _x0 - W / 2 - TOL) & (x[0] <= _x0 + W / 2 + TOL) &
                (x[1] >= _y0 - W / 2 - TOL) & (x[1] <= _y0 + W / 2 + TOL)
            )
        dofs = locate_dofs_geometrical(V, gate_marker)
        bcs.append(fem.dirichletbc(np.float64(voltage), dofs, V))

    def bottom_marker(x):
        return np.isclose(x[2], Lz, atol=TOL)

    bottom_dofs = locate_dofs_geometrical(V, bottom_marker)
    bcs.append(fem.dirichletbc(np.float64(0.0), bottom_dofs, V))
    return bcs


def solve_poisson(mesh, V, bcs):
    from dolfinx import fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    import ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    f = fem.Constant(mesh, default_scalar_type(0.0))
    L = f * v * ufl.dx

    problem = LinearProblem(
        a, L,
        petsc_options_prefix="probe_poisson_",
        bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
        },
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


def eval_points(mesh, phi, pts_array: np.ndarray) -> np.ndarray:
    """Evaluate phi at N points. Returns shape-(N,) array; NaN where not found."""
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    pts = np.asarray(pts_array, dtype=np.float64)
    n   = len(pts)
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, pts)

    results = np.full(n, np.nan)
    ep_list, ec_list, idx_list = [], [], []

    for i in range(n):
        links = colliding_cells.links(i)
        if len(links) > 0:
            ep_list.append(pts[i])
            ec_list.append(links[0])
            idx_list.append(i)

    if ep_list:
        ep = np.array(ep_list, dtype=np.float64)
        ec = np.array(ec_list, dtype=np.int32)
        vals = phi.eval(ep, ec)
        for j, idx in enumerate(idx_list):
            results[idx] = float(vals[j] if vals.ndim == 1 else vals[j, 0])

    return results


def eval_single(mesh, phi, pt: tuple) -> float:
    """Evaluate phi at one point. Returns NaN if not found."""
    arr = eval_points(mesh, phi, np.array([list(pt)], dtype=np.float64))
    return float(arr[0])


# ── Device-quantity extraction ───────────────────────────────────────────────
def extract_quantities(mesh, phi, V) -> dict:
    """
    Extract all device quantities of interest from a solved phi field.
    Returns a dict with float values (NaN where evaluation failed).
    """
    # 1-2: phi_min / phi_max over the z=40nm QW-proxy slice
    #      Use DOF coordinates to find owned DOFs near z=40nm (within ±2nm).
    dof_coords = V.tabulate_dof_coordinates()          # (n_local_dofs, 3)
    phi_vals   = phi.x.array[: len(dof_coords)]        # owned values

    mask_z40  = np.abs(dof_coords[:, 2] - Z_QW) < 2 * NM
    phi_z40   = phi_vals[mask_z40]
    phi_min   = float(np.min(phi_z40))  if len(phi_z40) > 0 else np.nan
    phi_max   = float(np.max(phi_z40))  if len(phi_z40) > 0 else np.nan

    # 3-6: phi directly under each gate centre at z=40nm
    phi_G1 = eval_single(mesh, phi, (-45*NM, -45*NM, Z_QW))
    phi_G2 = eval_single(mesh, phi, (+45*NM, -45*NM, Z_QW))
    phi_G3 = eval_single(mesh, phi, (-45*NM, +45*NM, Z_QW))
    phi_G4 = eval_single(mesh, phi, (+45*NM, +45*NM, Z_QW))

    # 7-8: barrier midpoints in z=40nm plane
    barrier_12 = eval_single(mesh, phi, (   0.0, -45*NM, Z_QW))  # G1-G2 midpoint
    barrier_13 = eval_single(mesh, phi, (-45*NM,    0.0, Z_QW))  # G1-G3 midpoint

    # 9: phi at lateral centre
    phi_center = eval_single(mesh, phi, (0.0, 0.0, Z_QW))

    # 10: bottom centre — should be ~0 (bottom BC check)
    phi_bottom_center = eval_single(mesh, phi, (0.0, 0.0, 140*NM))

    # 11: z-profile at G1 centre, z from 5 to 140nm (30 points)
    zs_G1 = np.linspace(5*NM, 140*NM, 30)
    pts_G1_z = np.column_stack([
        np.full(30, -45*NM),
        np.full(30, -45*NM),
        zs_G1,
    ])
    z_profile = eval_points(mesh, phi, pts_G1_z)
    z_profile_min   = float(np.nanmin(z_profile))
    z_profile_max   = float(np.nanmax(z_profile))
    # value at z=40nm: nearest sample index
    idx_40nm = int(np.argmin(np.abs(zs_G1 - Z_QW)))
    z_profile_at40  = float(z_profile[idx_40nm]) if not np.isnan(z_profile[idx_40nm]) else phi_G1

    # 12-13: derived quantities
    delta_phi_12 = phi_G1 - phi_G2                    # gate asymmetry
    delta_barrier = barrier_12 - phi_G1               # barrier height above G1

    return {
        "phi_min":           phi_min,
        "phi_max":           phi_max,
        "phi_G1":            phi_G1,
        "phi_G2":            phi_G2,
        "phi_G3":            phi_G3,
        "phi_G4":            phi_G4,
        "barrier_12":        barrier_12,
        "barrier_13":        barrier_13,
        "phi_center":        phi_center,
        "phi_bottom_center": phi_bottom_center,
        "z_profile_min":     z_profile_min,
        "z_profile_max":     z_profile_max,
        "z_profile_at40nm":  z_profile_at40,
        "delta_phi_12":      delta_phi_12,
        "delta_barrier":     delta_barrier,
        # Underlying slice arrays for XDMF/CSV export
        "_dof_coords_z40":   dof_coords[mask_z40],
        "_phi_vals_z40":     phi_z40,
        "_z_profile":        z_profile,
        "_zs_G1":            zs_G1,
    }


# ── Output helpers ────────────────────────────────────────────────────────────
def write_z40_csv(path: Path, q: dict) -> None:
    """Write the z=40nm slice scatter data (x, y, phi) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = q["_dof_coords_z40"]
    vals   = q["_phi_vals_z40"]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "phi"])
        for (cx, cy, _cz), pv in zip(coords, vals):
            writer.writerow([cx, cy, pv])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="4-gate probe stability test across mesh strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick",      action="store_true",
                        help="Run cases A, B, D, E (default if neither flag set)")
    parser.add_argument("--full",       action="store_true",
                        help="Run cases A, B, C, D, E")
    parser.add_argument("--outdir",     default="tests/test_probe_stability/output")
    parser.add_argument("--write-xdmf", action="store_true",
                        help="Write XDMF for each solution + z=40nm slice CSV")
    parser.add_argument("--save-csv",   action="store_true",
                        help="Write probe_stability_results.csv and probe_stability_quantities.csv")
    parser.add_argument("--max-dofs",   type=int, default=1_500_000)
    args = parser.parse_args()

    if not args.quick and not args.full:
        args.quick = True

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD

    results: list[dict] = []

    for (case, h_coarse, h_fine_box, in_quick, in_full) in CASE_DEFS:
        if args.full and not in_full:
            continue
        if not args.full and not in_quick:
            continue

        h_coarse_nm = h_coarse / NM
        h_fine_nm   = (h_fine_box / NM) if h_fine_box is not None else None

        print(f"\n── Case {case}  h_coarse={h_coarse_nm:.0f}nm"
              + (f"  h_fine={h_fine_nm:.0f}nm" if h_fine_nm else "  (uniform)") + " ──")

        boxes = make_boxes(h_fine_box)
        mesh  = build_mesh(comm, h_coarse, boxes)
        V     = build_function_space(mesh)

        ndofs  = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        ncells = mesh.topology.index_map(mesh.topology.dim).size_global

        if ndofs > args.max_dofs:
            print(f"  SKIP: {ndofs:,} DOFs exceeds --max-dofs={args.max_dofs:,}")
            continue

        print(f"  cells={ncells:,}  dofs={ndofs:,}")

        bcs = build_bcs(mesh, V)

        t0 = time.perf_counter()
        phi = solve_poisson(mesh, V, bcs)
        solve_time = time.perf_counter() - t0
        print(f"  solve_time={solve_time:.2f}s")

        q = extract_quantities(mesh, phi, V)

        if args.write_xdmf:
            xdmf_path = outdir / f"probe_stability_{case}.xdmf"
            write_field_xdmf(str(xdmf_path), mesh, phi)
            print(f"  Wrote {xdmf_path}")
            z40_csv_path = outdir / f"z40nm_slice_{case}.csv"
            write_z40_csv(z40_csv_path, q)
            print(f"  Wrote z=40nm slice: {z40_csv_path}")

        results.append({
            "case":         case,
            "h_coarse_nm":  h_coarse_nm,
            "h_fine_nm":    h_fine_nm,
            "cells":        ncells,
            "dofs":         ndofs,
            "solve_time":   solve_time,
            **{k: v for k, v in q.items() if not k.startswith("_")},
        })

    if not results:
        print("\nNo cases were run.")
        return

    # ── Main table ────────────────────────────────────────────────────────────
    FLOAT_FMT = "{:+.5f}"

    col_keys = [
        "case", "dofs",
        "phi_min", "phi_max",
        "phi_G1", "phi_G2", "phi_G3", "phi_G4",
        "barrier_12", "barrier_13",
        "phi_center",
        "delta_phi_12", "delta_barrier",
        "solve_time",
    ]
    col_heads = [
        "case", "dofs",
        "phi_min", "phi_max",
        "phi_G1", "phi_G2", "phi_G3", "phi_G4",
        "barr_12", "barr_13",
        "phi_ctr",
        "d12(V)", "d_barr(V)",
        "time_s",
    ]

    print()
    print("Device quantities (all potentials in Volts unless noted)")
    print()

    # Build column widths
    widths = [max(len(h), 8) for h in col_heads]
    header_str = "  ".join(f"{h:>{w}}" for h, w in zip(col_heads, widths))
    sep = "─" * len(header_str)
    print(sep)
    print(header_str)
    print(sep)

    for r in results:
        row_parts = []
        for key, head, w in zip(col_keys, col_heads, widths):
            if key == "case":
                row_parts.append(f"{r['case']:>{w}}")
            elif key == "dofs":
                row_parts.append(f"{r['dofs']:>{w},}")
            elif key == "solve_time":
                row_parts.append(f"{r['solve_time']:>{w}.2f}")
            else:
                val = r.get(key, np.nan)
                if np.isnan(val):
                    row_parts.append(f"{'nan':>{w}}")
                else:
                    row_parts.append(f"{val:>+{w}.5f}")
        print("  ".join(row_parts))

    print(sep)

    # ── STABILITY section ─────────────────────────────────────────────────────
    # Reference: finest available (C > B > A)
    ref_case = None
    for preferred in ("C", "B", "A"):
        if any(r["case"] == preferred for r in results):
            ref_case = preferred
            break

    ref_row = next((r for r in results if r["case"] == ref_case), None)

    quantity_keys = [
        "phi_min", "phi_max",
        "phi_G1", "phi_G2", "phi_G3", "phi_G4",
        "barrier_12", "barrier_13",
        "phi_center", "phi_bottom_center",
        "z_profile_at40nm",
        "delta_phi_12", "delta_barrier",
    ]

    case_names = [r["case"] for r in results if r["case"] != ref_case]

    print()
    print(f"STABILITY  (differences vs reference case {ref_case}, values in mV)")
    print()

    stab_col_heads = ["quantity", f"ref({ref_case}) V"] + [f"case_{c} mV" for c in case_names]
    stab_widths    = [18, 14] + [12] * len(case_names)
    stab_header    = "  ".join(f"{h:>{w}}" for h, w in zip(stab_col_heads, stab_widths))
    stab_sep       = "─" * len(stab_header)

    print(stab_sep)
    print(stab_header)
    print(stab_sep)

    # Mode-specific thresholds
    if args.full:
        warn_mv = None            # no separate warning tier in full mode
        fail_mv = FAIL_MV_FULL
    else:
        warn_mv = WARN_MV_QUICK
        fail_mv = FAIL_MV_QUICK

    # severity → list of message strings
    warn_flags: list[str] = []
    fail_flags: list[str] = []

    for qk in quantity_keys:
        ref_val = ref_row.get(qk, np.nan) if ref_row else np.nan
        ref_str = f"{ref_val:>+.5f}" if not np.isnan(ref_val) else "    nan"

        diff_strs = []
        for c in case_names:
            case_row = next((r for r in results if r["case"] == c), None)
            if case_row is None:
                diff_strs.append("    n/a")
                continue
            cval = case_row.get(qk, np.nan)
            if np.isnan(ref_val) or np.isnan(cval):
                diff_strs.append("    nan")
            else:
                diff_mv = (cval - ref_val) * 1e3
                diff_strs.append(f"{diff_mv:>+.2f}")
                # Check instability for B vs D and B vs E
                if ref_case == "B" and c in ("D", "E"):
                    adiff = abs(diff_mv)
                    tag = f"{qk}: B={ref_val*1e3:.2f}mV  {c}={cval*1e3:.2f}mV  |diff|={adiff:.2f}mV"
                    if adiff > fail_mv:
                        fail_flags.append(f"  {tag} > {fail_mv:.0f}mV")
                    elif warn_mv is not None and adiff > warn_mv:
                        warn_flags.append(f"  {tag} > {warn_mv:.0f}mV")

        row_cells = (
            [f"{qk:>{stab_widths[0]}}", f"{ref_str:>{stab_widths[1]}}"]
            + [f"{ds:>{stab_widths[2+i]}}" for i, ds in enumerate(diff_strs)]
        )
        print("  ".join(row_cells))

    print(stab_sep)

    print()
    b_avail = any(r["case"] == "B" for r in results)
    checked = [c for c in ("D", "E") if any(r["case"] == c for r in results)]

    if not b_avail or not checked:
        print("INFO: Case B and at least one of D/E not present — stability check skipped")
    elif fail_flags:
        print(f"FAIL: {len(fail_flags)} quantity(ies) differ by >{fail_mv:.0f}mV "
              f"(B vs {'/'.join(checked)}):")
        for msg in fail_flags:
            print(msg)
        if warn_flags:
            print(f"WARNING: {len(warn_flags)} additional quantity(ies) differ by >{warn_mv:.0f}mV:")
            for msg in warn_flags:
                print(msg)
        sys.exit(1)
    elif warn_flags:
        print(f"WARNING: {len(warn_flags)} quantity(ies) differ by >{warn_mv:.0f}mV "
              f"(B vs {'/'.join(checked)}) — within FAIL threshold of {fail_mv:.0f}mV:")
        for msg in warn_flags:
            print(msg)
    else:
        print(f"STABLE: all quantities agree within {warn_mv or fail_mv:.0f}mV "
              f"(B vs {'/'.join(checked)})")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.save_csv:
        # Summary results
        res_keys = [
            "case", "h_coarse_nm", "h_fine_nm", "cells", "dofs", "solve_time",
        ] + quantity_keys
        res_path = outdir / "probe_stability_results.csv"
        with open(res_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=res_keys)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in res_keys})
        print(f"\nSaved {res_path}")

        # Per-quantity stability table
        qty_path = outdir / "probe_stability_quantities.csv"
        qty_fieldnames = ["quantity", f"ref_{ref_case}_V"] + [f"case_{c}_mV" for c in case_names]
        with open(qty_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=qty_fieldnames)
            writer.writeheader()
            for qk in quantity_keys:
                ref_val = ref_row.get(qk, np.nan) if ref_row else np.nan
                row = {"quantity": qk, f"ref_{ref_case}_V": ref_val}
                for c in case_names:
                    case_row = next((r for r in results if r["case"] == c), None)
                    if case_row is None:
                        row[f"case_{c}_mV"] = ""
                    else:
                        cval = case_row.get(qk, np.nan)
                        row[f"case_{c}_mV"] = (
                            (cval - ref_val) * 1e3
                            if not (np.isnan(ref_val) or np.isnan(cval))
                            else ""
                        )
                writer.writerow(row)
        print(f"Saved {qty_path}")


if __name__ == "__main__":
    main()
