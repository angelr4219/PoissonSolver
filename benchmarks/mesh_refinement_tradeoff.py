#!/usr/bin/env python3
"""Mesh-refinement tradeoff study: 4-quadrant mixed refinement vs. uniform
coarse vs. uniform fine, all solving the SAME point-charge Poisson problem.

Three mesh strategies, same domain (cube, side L_nm), same point charge,
same Dirichlet (phi=0) outer BC, same probe points -- only the mesh differs:

  uniform_coarse : single h = H_COARSE_NM everywhere      (fast, expected inaccurate)
  uniform_fine   : single h = H_FINE_NM everywhere         (slow, expected accurate)
  quadrant_mixed : 4 quadrants in the x-y plane (full z height), each with
                   its own target h -- reusing the box-field refinement
                   machinery from src/poisson/refinement.py:
                       Q1 (x<0,y<0): h = 20 nm
                       Q2 (x>0,y<0): h = 10 nm
                       Q3 (x<0,y>0): h =  5 nm
                       Q4 (x>0,y>0): h =  1 nm   <- charge lives here

The point charge sits at the centre of Q4 (the finest quadrant), and probe
points are placed at increasing radii from the charge, inside Q4, so the
comparison directly tests whether local refinement near the source can
match the accuracy of a globally-fine mesh at a fraction of the cost.

Reference: analytic isolated point-charge Coulomb potential
(poisson.analytics.phi_point_charge), uniform dielectric eps_r.

NOTE: all three meshes share the exact same domain and Dirichlet BC, so any
domain-truncation error is identical across the three -- differences in
accuracy below are attributable to mesh resolution/placement only.

Output: results/mesh_refinement/  (report.txt + per-config CSV + XDMF fields)

Usage:
    ./run_dolfinx.sh benchmarks/mesh_refinement_tradeoff.py [--help]
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poisson.charges import gaussian_rho
from poisson.analytics import phi_point_charge, EPS0
from poisson.refinement import RefinementBox, build_refined_mesh_3d

COMM = MPI.COMM_WORLD
IS_ROOT = COMM.rank == 0


def _log(msg: str) -> None:
    if IS_ROOT:
        print(msg, flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Mesh builders
# ──────────────────────────────────────────────────────────────────────────

def build_uniform_mesh(L_m, h_m):
    """Uniform structured tet mesh over [-L/2,L/2] x [-L/2,L/2] x [0,L],
    matching the domain convention used by build_refined_mesh_3d below."""
    N = max(2, int(round(L_m / h_m)))
    p0 = np.array([-L_m / 2, -L_m / 2, 0.0])
    p1 = np.array([+L_m / 2, +L_m / 2, L_m])
    domain = dmesh.create_box(COMM, points=[p0, p1], n=[N, N, N],
                              cell_type=dmesh.CellType.tetrahedron)
    return domain


def build_quadrant_mesh(L_m, h_q1, h_q2, h_q3, h_q4):
    """4-quadrant mixed-refinement mesh via the gmsh Box-field machinery in
    src/poisson/refinement.py. Quadrant boxes tile [-L/2,L/2]^2 x [0,L]."""
    quarter = L_m / 2.0
    h_coarse = max(h_q1, h_q2, h_q3, h_q4)
    boxes = [
        RefinementBox(cx=-quarter / 2, cy=-quarter / 2, cz=L_m / 2,
                       lx=quarter, ly=quarter, lz=L_m, h_fine=h_q1),
        RefinementBox(cx=+quarter / 2, cy=-quarter / 2, cz=L_m / 2,
                       lx=quarter, ly=quarter, lz=L_m, h_fine=h_q2),
        RefinementBox(cx=-quarter / 2, cy=+quarter / 2, cz=L_m / 2,
                       lx=quarter, ly=quarter, lz=L_m, h_fine=h_q3),
        RefinementBox(cx=+quarter / 2, cy=+quarter / 2, cz=L_m / 2,
                       lx=quarter, ly=quarter, lz=L_m, h_fine=h_q4),
    ]
    domain = build_refined_mesh_3d(COMM, L_m, L_m, L_m, h_coarse, boxes)
    return domain


# ──────────────────────────────────────────────────────────────────────────
# Solve + evaluate
# ──────────────────────────────────────────────────────────────────────────

def solve_point_charge(domain, q, x0_m, sigma_m, eps_r):
    V = fem.functionspace(domain, ("Lagrange", 1))
    eps = fem.Constant(domain, PETSc.ScalarType(EPS0 * eps_r))
    rho = gaussian_rho(domain, x0_m, q, sigma_m)

    fdim = domain.topology.dim - 1
    facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc_dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx
    a_form, L_form = fem.form(a), fem.form(L)

    A = fem_petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = fem_petsc.assemble_vector(L_form)
    fem_petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc])

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()

    uh = fem.Function(V)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    uh.name = "phi"
    return uh, V


def eval_at_points(domain, uh, pts_m):
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(domain, domain.topology.dim)
    colls = compute_collisions_points(tree, pts_m)
    cells = compute_colliding_cells(domain, colls, pts_m)
    vals = []
    for i, pt in enumerate(pts_m):
        cell_list = cells.links(i)
        if len(cell_list) > 0:
            v = float(np.asarray(uh.eval(pt.reshape(1, -1), cell_list[:1])).reshape(-1)[0])
        else:
            v = float("nan")
        vals.append(v)
    return np.array(vals)


def n_cells_global(domain):
    tdim = domain.topology.dim
    return domain.topology.index_map(tdim).size_global


# ──────────────────────────────────────────────────────────────────────────
def run_config(name, domain, q, x0_m, sigma_m, eps_r, probe_pts_m, probe_r_m, outdir):
    t0 = time.perf_counter()
    n_cells = n_cells_global(domain)
    t_mesh = time.perf_counter() - t0  # mesh already built by caller; report 0 here, real cost logged separately

    t1 = time.perf_counter()
    uh, V = solve_point_charge(domain, q, x0_m, sigma_m, eps_r)
    t_solve = time.perf_counter() - t1

    n_dofs = V.dofmap.index_map.size_global
    phi_h = eval_at_points(domain, uh, probe_pts_m)
    phi_ex = phi_point_charge(probe_r_m, q, eps_r)
    rel_err = np.abs(phi_h - phi_ex) / np.abs(phi_ex)

    if IS_ROOT:
        xdmf_path = os.path.join(outdir, f"{name}.xdmf")
        with XDMFFile(COMM, xdmf_path, "w") as f:
            f.write_mesh(domain)
            f.write_function(uh)

    return dict(name=name, n_cells=n_cells, n_dofs=n_dofs,
                t_solve=t_solve, phi_h=phi_h, phi_ex=phi_ex, rel_err=rel_err,
                max_rel_err=float(np.max(rel_err)), mean_rel_err=float(np.mean(rel_err)))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L-nm", type=float, default=80.0, help="Cube side length [nm]")
    p.add_argument("--h-coarse-nm", type=float, default=20.0, help="Uniform-coarse mesh size [nm]")
    p.add_argument("--h-fine-nm", type=float, default=1.0, help="Uniform-fine mesh size [nm]")
    p.add_argument("--h-q1-nm", type=float, default=20.0, help="Quadrant 1 (x<0,y<0) mesh size [nm]")
    p.add_argument("--h-q2-nm", type=float, default=10.0, help="Quadrant 2 (x>0,y<0) mesh size [nm]")
    p.add_argument("--h-q3-nm", type=float, default=5.0, help="Quadrant 3 (x<0,y>0) mesh size [nm]")
    p.add_argument("--h-q4-nm", type=float, default=1.0, help="Quadrant 4 (x>0,y>0) mesh size [nm] -- charge lives here")
    p.add_argument("--q", type=float, default=1.602176634e-19, help="Charge [C] (default +e)")
    p.add_argument("--sigma-nm", type=float, default=2.0, help="Gaussian regularisation width [nm]")
    p.add_argument("--eps-r", type=float, default=1.0, help="Uniform relative permittivity")
    p.add_argument("--probe-r-nm", type=float, nargs="+", default=[5.0, 10.0, 15.0],
                   help="Probe radii from the charge [nm], along +x, inside Q4")
    p.add_argument("--outdir", default="results/mesh_refinement", help="Output directory")
    return p


def main():
    args = build_parser().parse_args()
    if IS_ROOT:
        os.makedirs(args.outdir, exist_ok=True)
        print("=" * 70)
        print("Mesh-refinement tradeoff: uniform-coarse vs uniform-fine vs 4-quadrant")
        print(f"  L = {args.L_nm} nm   charge q = {args.q:.6e} C   sigma = {args.sigma_nm} nm")
        print("=" * 70)

    L_m = args.L_nm * 1e-9
    q = args.q
    sigma_m = args.sigma_nm * 1e-9
    eps_r = args.eps_r

    # Charge at the centre of quadrant 4 (x>0, y>0), mid-height in z.
    quarter_nm = args.L_nm / 2.0
    x0_nm = np.array([quarter_nm / 2.0, quarter_nm / 2.0, args.L_nm / 2.0])
    x0_m = x0_nm * 1e-9

    probe_pts_m = np.array(
        [[x0_nm[0] + r, x0_nm[1], x0_nm[2]] for r in args.probe_r_nm], dtype=float
    ) * 1e-9
    probe_r_m = np.array(args.probe_r_nm, dtype=float) * 1e-9

    results = []

    _log("\n-- uniform_coarse --")
    t0 = time.perf_counter()
    dom_c = build_uniform_mesh(L_m, args.h_coarse_nm * 1e-9)
    t_mesh_c = time.perf_counter() - t0
    res_c = run_config("uniform_coarse", dom_c, q, x0_m, sigma_m, eps_r, probe_pts_m, probe_r_m, args.outdir)
    res_c["t_mesh"] = t_mesh_c
    res_c["h_label"] = f"{args.h_coarse_nm:.1f}nm uniform"
    results.append(res_c)
    _log(f"  n_cells={res_c['n_cells']:>10,}  n_dofs={res_c['n_dofs']:>10,}  "
         f"t_mesh={t_mesh_c:6.2f}s  t_solve={res_c['t_solve']:6.2f}s  "
         f"max_rel_err={res_c['max_rel_err']:.3e}")

    _log("\n-- uniform_fine --")
    t0 = time.perf_counter()
    dom_f = build_uniform_mesh(L_m, args.h_fine_nm * 1e-9)
    t_mesh_f = time.perf_counter() - t0
    res_f = run_config("uniform_fine", dom_f, q, x0_m, sigma_m, eps_r, probe_pts_m, probe_r_m, args.outdir)
    res_f["t_mesh"] = t_mesh_f
    res_f["h_label"] = f"{args.h_fine_nm:.1f}nm uniform"
    results.append(res_f)
    _log(f"  n_cells={res_f['n_cells']:>10,}  n_dofs={res_f['n_dofs']:>10,}  "
         f"t_mesh={t_mesh_f:6.2f}s  t_solve={res_f['t_solve']:6.2f}s  "
         f"max_rel_err={res_f['max_rel_err']:.3e}")

    _log("\n-- quadrant_mixed --")
    t0 = time.perf_counter()
    dom_q = build_quadrant_mesh(L_m, args.h_q1_nm * 1e-9, args.h_q2_nm * 1e-9,
                                  args.h_q3_nm * 1e-9, args.h_q4_nm * 1e-9)
    t_mesh_q = time.perf_counter() - t0
    res_q = run_config("quadrant_mixed", dom_q, q, x0_m, sigma_m, eps_r, probe_pts_m, probe_r_m, args.outdir)
    res_q["t_mesh"] = t_mesh_q
    res_q["h_label"] = f"Q1={args.h_q1_nm:.0f}/Q2={args.h_q2_nm:.0f}/Q3={args.h_q3_nm:.0f}/Q4={args.h_q4_nm:.0f}nm"
    results.append(res_q)
    _log(f"  n_cells={res_q['n_cells']:>10,}  n_dofs={res_q['n_dofs']:>10,}  "
         f"t_mesh={t_mesh_q:6.2f}s  t_solve={res_q['t_solve']:6.2f}s  "
         f"max_rel_err={res_q['max_rel_err']:.3e}")

    if not IS_ROOT:
        return

    csv_path = os.path.join(args.outdir, "tradeoff_results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["name", "h_label", "n_cells", "n_dofs", "t_mesh", "t_solve",
                      "t_total", "max_rel_err", "mean_rel_err"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            r["t_total"] = r["t_mesh"] + r["t_solve"]
            writer.writerow(r)

    report_path = os.path.join(args.outdir, "report.txt")
    lines = []
    lines.append("=" * 78)
    lines.append("MESH-REFINEMENT TRADEOFF -- uniform-coarse vs uniform-fine vs 4-quadrant")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Domain: cube L = {args.L_nm} nm, x,y in [-L/2,L/2], z in [0,L]")
    lines.append(f"Charge: q = {q:.6e} C at quadrant-4 centre "
                 f"({x0_nm[0]:.1f}, {x0_nm[1]:.1f}, {x0_nm[2]:.1f}) nm, "
                 f"Gaussian sigma = {args.sigma_nm} nm, eps_r = {eps_r}")
    lines.append(f"Probe points: r = {args.probe_r_nm} nm from the charge, along +x, inside Q4")
    lines.append("Reference: analytic isolated point-charge Coulomb potential "
                 "(poisson.analytics.phi_point_charge)")
    lines.append("Quadrant layout (x<0/x>0 split at x=0, y<0/y>0 split at y=0, full z height):")
    lines.append(f"  Q1 (x<0,y<0): h={args.h_q1_nm} nm    Q2 (x>0,y<0): h={args.h_q2_nm} nm")
    lines.append(f"  Q3 (x<0,y>0): h={args.h_q3_nm} nm    Q4 (x>0,y>0): h={args.h_q4_nm} nm  <- charge here")
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"{'config':<16}{'mesh':<24}{'n_cells':>10}{'n_dofs':>10}"
                 f"{'t_mesh[s]':>11}{'t_solve[s]':>12}{'max_relerr':>12}{'mean_relerr':>13}")
    lines.append("-" * 100)
    for r in results:
        lines.append(f"{r['name']:<16}{r['h_label']:<24}{r['n_cells']:>10,}{r['n_dofs']:>10,}"
                     f"{r['t_mesh']:>11.2f}{r['t_solve']:>12.2f}"
                     f"{r['max_rel_err']:>12.3e}{r['mean_rel_err']:>13.3e}")
    lines.append("")
    lines.append("  Raw phi at each probe point (FEM vs analytic), per config:")
    for r in results:
        lines.append(f"  [{r['name']}  ({r['h_label']})]")
        for rr, ph, pe, re in zip(args.probe_r_nm, r["phi_h"], r["phi_ex"], r["rel_err"]):
            lines.append(f"      r={rr:6.1f}nm   phi_FEM={ph: .6e} V   "
                         f"phi_exact={pe: .6e} V   rel_err={re:.3e}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("TRADEOFF SUMMARY")
    lines.append("-" * 78)
    c, fi, qd = results[0], results[1], results[2]
    speedup_vs_fine = (fi["t_mesh"] + fi["t_solve"]) / max(qd["t_mesh"] + qd["t_solve"], 1e-9)
    dof_ratio = fi["n_dofs"] / max(qd["n_dofs"], 1)
    lines.append(f"  uniform_fine total time   = {fi['t_mesh']+fi['t_solve']:.2f}s "
                 f"({fi['n_dofs']:,} dofs, max_rel_err={fi['max_rel_err']:.3e})")
    lines.append(f"  quadrant_mixed total time = {qd['t_mesh']+qd['t_solve']:.2f}s "
                 f"({qd['n_dofs']:,} dofs, max_rel_err={qd['max_rel_err']:.3e})")
    lines.append(f"  uniform_coarse total time = {c['t_mesh']+c['t_solve']:.2f}s "
                 f"({c['n_dofs']:,} dofs, max_rel_err={c['max_rel_err']:.3e})")
    lines.append(f"  -> quadrant_mixed uses {dof_ratio:.1f}x fewer DOFs than uniform_fine, "
                 f"runs {speedup_vs_fine:.1f}x {'faster' if speedup_vs_fine > 1 else 'slower'}, "
                 f"for {'comparable' if abs(qd['max_rel_err']-fi['max_rel_err']) < 0.1*fi['max_rel_err'] else 'different'} "
                 f"near-charge accuracy.")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print("\n" + report)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nConsolidated report written to {report_path}")


if __name__ == "__main__":
    main()
