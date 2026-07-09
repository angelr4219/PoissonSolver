"""
test_four_square_refinement_box.py

Demonstrate that local mesh refinement (RefinementBox) around the gate region
achieves near-fine-mesh accuracy with far fewer total DOFs.

Geometry (SI units internally, inputs in nm):
  Box:    Lx=300nm, Ly=300nm, Lz=150nm
  Domain: x in [-150,150]nm, y in [-150,150]nm, z in [0,150]nm
  z=0  : top surface / gate plane
  z=Lz : bottom contact (phi=0)

4 square gates on z=0, each 60x60nm with 30nm gaps:
  G1: centre (-45,-45)nm  V=+0.25V
  G2: centre (+45,-45)nm  V=+0.10V
  G3: centre (-45,+45)nm  V=-0.10V
  G4: centre (+45,+45)nm  V=+0.25V

Side faces and top face outside gates: Neumann (natural BC, do nothing).

Usage:
  python tests/test_four_square_refinement_box.py [--quick|--full]
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
NM = 1e-9   # metre per nanometre

Lx = 300 * NM
Ly = 300 * NM
Lz = 150 * NM

W   = 60 * NM  # gate side length
Z_QW = 40 * NM  # quantum-well proxy depth for probe lines

# Gate definitions: (x_centre, y_centre, voltage_V)
GATES = [
    (-45 * NM, -45 * NM, +0.25),  # G1
    (+45 * NM, -45 * NM, +0.10),  # G2
    (-45 * NM, +45 * NM, -0.10),  # G3
    (+45 * NM, +45 * NM, +0.25),  # G4
]

# RefinementBox parameters (shared by cases D/E/F)
REFBOX_CX = 0.0
REFBOX_CY = 0.0
REFBOX_CZ = Lz / 2       # midpoint in z
REFBOX_LX = 200 * NM
REFBOX_LY = 200 * NM
REFBOX_LZ =  80 * NM     # covers gates and QW depth below

TOL = 1e-12  # geometric tolerance for locate_dofs_geometrical

# ── Case definitions ─────────────────────────────────────────────────────────
# (name, h_coarse, h_fine_box_or_None, in_quick, in_full)
CASE_DEFS = [
    ("A", 20 * NM, None,      True,  True),
    ("B", 10 * NM, None,      True,  True),
    ("C",  5 * NM, None,      False, True),   # --full only (or if within max_dofs)
    ("D", 20 * NM, 10 * NM,   True,  True),
    ("E", 20 * NM,  5 * NM,   True,  True),
    ("F", 20 * NM,  2 * NM,   False, True),   # --full only, likely exceeds max_dofs
]


# ── Probe-line factories ─────────────────────────────────────────────────────
def make_probe_xline(n: int = 80) -> np.ndarray:
    """x from -120 to 120nm, y=0, z=40nm."""
    xs = np.linspace(-120 * NM, 120 * NM, n)
    return np.column_stack([xs, np.zeros(n), np.full(n, Z_QW)])


def make_probe_zline(n: int = 50) -> np.ndarray:
    """x=0, y=0, z from 5 to 140nm."""
    zs = np.linspace(5 * NM, 140 * NM, n)
    return np.column_stack([np.zeros(n), np.zeros(n), zs])


def make_probe_barrier12(n: int = 20) -> np.ndarray:
    """x from G1_centre to G2_centre (-45 to +45nm), y=-45nm, z=40nm."""
    xs = np.linspace(-45 * NM, +45 * NM, n)
    return np.column_stack([xs, np.full(n, -45 * NM), np.full(n, Z_QW)])


# ── Mesh / FEM helpers ───────────────────────────────────────────────────────
def make_boxes(h_fine_box) -> list[RefinementBox]:
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

    # Dirichlet BCs for each gate patch on z=0
    for (x0, y0, voltage) in GATES:
        def gate_marker(x, _x0=x0, _y0=y0):
            return (
                np.isclose(x[2], 0.0, atol=TOL) &
                (x[0] >= _x0 - W / 2 - TOL) & (x[0] <= _x0 + W / 2 + TOL) &
                (x[1] >= _y0 - W / 2 - TOL) & (x[1] <= _y0 + W / 2 + TOL)
            )
        dofs = locate_dofs_geometrical(V, gate_marker)
        bcs.append(fem.dirichletbc(np.float64(voltage), dofs, V))

    # Bottom contact: phi=0 at z=Lz
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
    f = fem.Constant(mesh, default_scalar_type(0.0))  # rho=0, pure Laplace
    L = f * v * ufl.dx

    problem = LinearProblem(
        a, L,
        petsc_options_prefix="fourgate_poisson_",
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
    """Evaluate phi at N points. Returns shape-(N,) array; NaN where point not found."""
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    pts = np.asarray(pts_array, dtype=np.float64)
    n = len(pts)
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, pts)

    results = np.full(n, np.nan)
    eval_pts_list, eval_cells_list, eval_idx_list = [], [], []

    for i in range(n):
        links = colliding_cells.links(i)
        if len(links) > 0:
            eval_pts_list.append(pts[i])
            eval_cells_list.append(links[0])
            eval_idx_list.append(i)

    if eval_pts_list:
        ep = np.array(eval_pts_list, dtype=np.float64)
        ec = np.array(eval_cells_list, dtype=np.int32)
        vals = phi.eval(ep, ec)
        for j, idx in enumerate(eval_idx_list):
            results[idx] = float(vals[j, 0])

    return results


def eval_probe_lines(mesh, phi):
    """Return (px, pz, pb): x-line, z-line, barrier-12 probe arrays."""
    px = eval_points(mesh, phi, make_probe_xline())
    pz = eval_points(mesh, phi, make_probe_zline())
    pb = eval_points(mesh, phi, make_probe_barrier12())
    return px, pz, pb


def compute_errors(probe_ref, probe_case):
    """
    Compute max / mean / rms absolute differences across all probe lines.
    NaN entries are excluded from statistics.
    Returns (max_diff, mean_diff, rms_diff).
    """
    diffs = []
    for r, c in zip(probe_ref, probe_case):
        mask = ~(np.isnan(r) | np.isnan(c))
        if mask.any():
            diffs.append(np.abs(r[mask] - c[mask]))
    if not diffs:
        return np.nan, np.nan, np.nan
    all_d = np.concatenate(diffs)
    return float(np.max(all_d)), float(np.mean(all_d)), float(np.sqrt(np.mean(all_d**2)))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="4-gate RefinementBox accuracy test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick",      action="store_true",
                        help="Run cases A, B, D, E (default if neither flag set)")
    parser.add_argument("--full",       action="store_true",
                        help="Run all cases A-F")
    parser.add_argument("--outdir",     default="tests/test_four_square_refinement_box/output",
                        help="Directory for XDMF and CSV output")
    parser.add_argument("--write-xdmf", action="store_true",
                        help="Write solution XDMF for each case")
    parser.add_argument("--save-csv",   action="store_true",
                        help="Write four_gate_results.csv to --outdir")
    parser.add_argument("--max-dofs",   type=int, default=1_500_000,
                        help="Skip cases whose DOF count exceeds this value")
    args = parser.parse_args()

    if not args.quick and not args.full:
        args.quick = True  # default behaviour

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD

    results: list[dict] = []
    probe_by_case: dict[str, tuple] = {}
    ndof_by_case: dict[str, int] = {}

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

        if args.write_xdmf:
            h_tag = f"{h_coarse_nm:.0f}nm"
            xdmf_path = outdir / f"four_gate_{case}_h{h_tag}.xdmf"
            write_field_xdmf(str(xdmf_path), mesh, phi)
            print(f"  Wrote {xdmf_path}")

        probes = eval_probe_lines(mesh, phi)
        probe_by_case[case] = probes
        ndof_by_case[case]  = ndofs

        results.append({
            "case":         case,
            "h_coarse_nm":  h_coarse_nm,
            "h_fine_nm":    h_fine_nm,
            "cells":        ncells,
            "dofs":         ndofs,
            "solve_time":   solve_time,
            "probes":       probes,
        })

    if not results:
        print("\nNo cases were run. Check --max-dofs or add --full/--quick flags.")
        return

    # Choose reference: case C if available, otherwise case B
    ref_case   = "C" if "C" in probe_by_case else ("B" if "B" in probe_by_case else None)
    ref_probes = probe_by_case.get(ref_case)
    ndof_ref   = ndof_by_case.get(ref_case, 1)

    # Compute per-case errors and compression ratios vs reference
    for r in results:
        if ref_probes is not None and r["case"] != ref_case:
            mx, mn, rms = compute_errors(ref_probes, r["probes"])
        else:
            mx = mn = rms = np.nan
        r["max_probe_diff"]   = mx
        r["mean_probe_diff"]  = mn
        r["rms_probe_diff"]   = rms
        r["compression_ratio"] = ndof_ref / r["dofs"] if r["dofs"] > 0 else np.nan

    # ── Print results table ──────────────────────────────────────────────────
    hdr = (f"{'case':>4}  {'h_coarse':>8}  {'h_fine':>6}  {'cells':>9}  "
           f"{'dofs':>9}  {'max_Δφ(V)':>10}  {'rms_Δφ(V)':>10}  "
           f"{'compress':>8}  {'time_s':>7}")
    sep = "─" * len(hdr)
    print(f"\nReference case for error metrics: {ref_case or 'none'}")
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        h_fine_str = f"{r['h_fine_nm']:.0f}" if r["h_fine_nm"] is not None else "—"
        mx_str  = f"{r['max_probe_diff']:.5f}"  if not np.isnan(r["max_probe_diff"])  else "  (ref)"
        rms_str = f"{r['rms_probe_diff']:.5f}"  if not np.isnan(r["rms_probe_diff"])  else "  (ref)"
        cr_str  = f"{r['compression_ratio']:.3f}" if not np.isnan(r["compression_ratio"]) else "  1.000"
        print(
            f"{r['case']:>4}  {r['h_coarse_nm']:>8.0f}  {h_fine_str:>6}  "
            f"{r['cells']:>9,}  {r['dofs']:>9,}  "
            f"{mx_str:>10}  {rms_str:>10}  "
            f"{cr_str:>8}  {r['solve_time']:>7.2f}"
        )
    print(sep)

    print()
    print("Goal: cases D/E should match case B probe values within 5%, with fewer DOFs")
    print("compression_ratio > 1 means fewer DOFs than uniform-fine reference")

    # ── PASS/FAIL for case E ─────────────────────────────────────────────────
    case_e = next((r for r in results if r["case"] == "E"), None)
    if case_e is not None and ref_probes is not None and not np.isnan(case_e["rms_probe_diff"]):
        all_ref_vals = np.concatenate([p[~np.isnan(p)] for p in ref_probes])
        ref_scale    = float(np.max(np.abs(all_ref_vals))) if len(all_ref_vals) > 0 else 1.0
        threshold    = 0.05 * ref_scale
        passed       = case_e["rms_probe_diff"] < threshold
        status       = "PASS" if passed else "FAIL"
        print(
            f"\n{status}: Case E rms_probe_diff={case_e['rms_probe_diff']:.6f} V  "
            f"threshold={threshold:.6f} V  (5 % of max|phi_ref|={ref_scale:.5f} V)"
        )
    else:
        print("\nINFO: Case E or reference not available — skipping PASS/FAIL check")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    if args.save_csv:
        fieldnames = [
            "case", "h_coarse_nm", "h_fine_nm", "cells", "dofs",
            "max_probe_diff", "mean_probe_diff", "rms_probe_diff",
            "compression_ratio", "solve_time",
        ]
        csv_path = outdir / "four_gate_results.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})
        print(f"\nSaved results CSV: {csv_path}")


if __name__ == "__main__":
    main()
