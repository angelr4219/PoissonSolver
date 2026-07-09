"""
test_square_gate_h_convergence.py
==================================
Convergence test: FEM vs analytic half-space potential for a rectangular gate.

Analytic reference (Davies/Jackson half-space rectangular gate):
  phi_gate(x,y,z) = (V_g/(2*pi)) * sum_{i,j in {-1,+1}}
    i*j * arctan( (x-x0+i*Wx/2)*(y-y0+j*Wy/2)
                  / (z * sqrt((x-x0+i*Wx/2)^2 + (y-y0+j*Wy/2)^2 + z^2)) )

Exact for tox=0, uniform epsilon, isolated gate in a half-space.

Geometry: box x in [-150,150]nm, y in [-150,150]nm, z in [0,150]nm
  z=0 is the gate plane; z increases into the material.
Gate: Wx=Wy=80nm, centered at (0,0), V_g=1.0V applied on z=0 within gate area.
BCs: gate facet -> Dirichlet V_g; all other faces -> Dirichlet analytic formula.
Source: rho=0 (pure Laplace, no free charge).

Probe lines (away from gate edges to avoid singularity):
  1. x-line: y=0, z=40nm, x in [-100,100]nm (50 pts)
  2. z-line: x=0, y=0, z in [5,140]nm (40 pts)
  3. off-axis: y=30nm, z=40nm, x in [-80,80]nm (40 pts)

Run:
    ./run_dolfinx.sh tests/test_square_gate_h_convergence.py --quick
    ./run_dolfinx.sh tests/test_square_gate_h_convergence.py --full
"""
from __future__ import annotations

import sys
import time
import argparse
import csv
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
import ufl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

COMM = MPI.COMM_WORLD

# ─── Physical / geometry constants ───────────────────────────────────────────
X_MIN, X_MAX = -150e-9, 150e-9
Y_MIN, Y_MAX = -150e-9, 150e-9
Z_MIN, Z_MAX = 0.0, 150e-9

WX_SI = WY_SI = 80e-9   # gate width / height [m]
X0, Y0 = 0.0, 0.0       # gate centre [m]
V_G = 1.0                # gate voltage [V]

SINGULARITY_EXCL_NM = 5.0   # exclude probe points within this radius of gate corners


# ─── Analytic half-space formula ─────────────────────────────────────────────

def phi_halfspace_gate(
    xpts: np.ndarray,
    ypts: np.ndarray,
    zpts: np.ndarray,
    x0: float = X0,
    y0: float = Y0,
    wx: float = WX_SI,
    wy: float = WY_SI,
    vg: float = V_G,
) -> np.ndarray:
    """
    Davies/Jackson analytic half-space potential for a rectangular gate.

    Valid for z > 0 (inside the half-space).  At z -> 0+:
      inside gate area  -> vg
      outside gate area -> 0
    """
    phi = np.zeros(len(xpts), dtype=np.float64)
    for i in (-1, 1):
        for j in (-1, 1):
            u = xpts - x0 + i * wx / 2.0
            v = ypts - y0 + j * wy / 2.0
            r2 = u ** 2 + v ** 2 + zpts ** 2
            denom = zpts * np.sqrt(r2)
            with np.errstate(divide="ignore", invalid="ignore"):
                arg = np.where(np.abs(denom) > 0.0, u * v / denom,
                               np.sign(u * v) * np.inf)
            phi += i * j * np.arctan(arg)
    return (vg / (2.0 * np.pi)) * phi


def phi_bc_interp(x: np.ndarray) -> np.ndarray:
    """
    Interpolation callable for dolfinx: x has shape (3, N) in SI units.

    On z=0: gate footprint -> V_G, outside footprint -> 0.0
    Elsewhere: analytic half-space formula.
    """
    xs, ys, zs = x[0], x[1], x[2]
    phi = phi_halfspace_gate(xs, ys, zs)

    # z=0 face: override with exact boundary values
    on_z0 = np.abs(zs) < 1e-13
    in_gate = (
        (np.abs(xs - X0) <= WX_SI / 2 + 1e-13) &
        (np.abs(ys - Y0) <= WY_SI / 2 + 1e-13)
    )
    phi[on_z0 & in_gate] = V_G
    phi[on_z0 & ~in_gate] = 0.0
    return phi


# ─── Gate corners (singularity zone) ─────────────────────────────────────────

GATE_CORNERS = np.array([
    [X0 + WX_SI / 2, Y0 + WY_SI / 2, 0.0],
    [X0 + WX_SI / 2, Y0 - WY_SI / 2, 0.0],
    [X0 - WX_SI / 2, Y0 + WY_SI / 2, 0.0],
    [X0 - WX_SI / 2, Y0 - WY_SI / 2, 0.0],
])


def is_near_singularity(pts: np.ndarray, excl_nm: float = SINGULARITY_EXCL_NM) -> np.ndarray:
    """Boolean mask: True for pts within excl_nm of any gate corner."""
    excl = excl_nm * 1e-9
    near = np.zeros(len(pts), dtype=bool)
    for corner in GATE_CORNERS:
        d = np.linalg.norm(pts - corner, axis=1)
        near |= d < excl
    return near


# ─── Probe line definitions ───────────────────────────────────────────────────

def make_probe_lines() -> np.ndarray:
    """Return (N, 3) array of probe points in SI units."""
    z40 = 40e-9
    # 1. x-line
    pts_x = np.column_stack([
        np.linspace(-100e-9, 100e-9, 50),
        np.zeros(50),
        np.full(50, z40),
    ])
    # 2. z-line
    pts_z = np.column_stack([
        np.zeros(40),
        np.zeros(40),
        np.linspace(5e-9, 140e-9, 40),
    ])
    # 3. off-axis
    pts_off = np.column_stack([
        np.linspace(-80e-9, 80e-9, 40),
        np.full(40, 30e-9),
        np.full(40, z40),
    ])
    return np.vstack([pts_x, pts_z, pts_off])


# ─── Mesh generation ─────────────────────────────────────────────────────────

def build_mesh(comm: MPI.Comm, h_fine_nm: float, h_far_nm: float):
    """
    Build a structured tetrahedral box mesh for the square-gate half-space problem.

    Uses dolfinx.mesh.create_box + locate_entities_boundary to avoid the OCC
    addRectangle-on-face crash. Uniform mesh at h_fine_nm.

    Physical tags:
      facets 10    — gate surface (z=0, within gate footprint)
      facets  2    — all other boundary faces
      cells   1    — volume
    """
    from dolfinx import mesh as _mesh

    h = h_fine_nm * 1e-9
    nx = max(4, round((X_MAX - X_MIN) / h))
    ny = max(4, round((Y_MAX - Y_MIN) / h))
    nz = max(2, round((Z_MAX - Z_MIN) / h))

    msh = _mesh.create_box(
        comm,
        [np.array([X_MIN, Y_MIN, Z_MIN]),
         np.array([X_MAX, Y_MAX, Z_MAX])],
        [nx, ny, nz],
        cell_type=_mesh.CellType.tetrahedron,
    )
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    msh.name = "square_gate"

    fdim = msh.topology.dim - 1
    tol  = h * 0.6   # half-ish cell length

    half_x = WX_SI / 2
    half_y = WY_SI / 2

    def _on_gate(x):
        return (
            np.isclose(x[2], 0.0, atol=tol)
            & (x[0] >= X0 - half_x - tol) & (x[0] <= X0 + half_x + tol)
            & (x[1] >= Y0 - half_y - tol) & (x[1] <= Y0 + half_y + tol)
        )

    def _on_any_boundary(x):
        return (
            np.isclose(x[0], X_MIN, atol=tol) | np.isclose(x[0], X_MAX, atol=tol)
            | np.isclose(x[1], Y_MIN, atol=tol) | np.isclose(x[1], Y_MAX, atol=tol)
            | np.isclose(x[2], 0.0,   atol=tol) | np.isclose(x[2], Z_MAX, atol=tol)
        )

    gate_f  = _mesh.locate_entities_boundary(msh, fdim, _on_gate)
    all_f   = _mesh.locate_entities_boundary(msh, fdim, _on_any_boundary)
    gate_set = set(gate_f.tolist())
    other_f  = np.array([f for f in all_f if f not in gate_set], dtype=np.int32)

    fidx = np.concatenate([gate_f, other_f])
    fval = np.concatenate([
        np.full(len(gate_f),  10, dtype=np.int32),
        np.full(len(other_f),  2, dtype=np.int32),
    ])
    order = np.argsort(fidx)
    facet_tags = _mesh.meshtags(
        msh, fdim, fidx[order].astype(np.int32), fval[order]
    )

    n_cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_tags = _mesh.meshtags(
        msh, msh.topology.dim,
        np.arange(n_cells, dtype=np.int32),
        np.ones(n_cells, dtype=np.int32),
    )

    return msh, cell_tags, facet_tags


# ─── FEM solve for one h level ────────────────────────────────────────────────

def run_h(h_nm: float, args, prev_h: float | None, prev_rms: float | None) -> dict | None:
    h_far_nm = min(h_nm * 8.0, 60.0)

    t0 = time.perf_counter()
    msh, cell_tags, facet_tags = build_mesh(COMM, h_nm, h_far_nm)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    V = fem.functionspace(msh, ("Lagrange", 1))
    ndofs = V.dofmap.index_map.size_global
    ncells = msh.topology.index_map(msh.topology.dim).size_global

    if ndofs > args.max_dofs:
        if COMM.rank == 0:
            print(f"  [skip] h={h_nm}nm: {ndofs} DOFs > --max-dofs={args.max_dofs}",
                  flush=True)
        return None

    # ── Boundary condition ──────────────────────────────────────────────────
    phi_bc_fn = fem.Function(V, name="phi_bc")
    phi_bc_fn.interpolate(phi_bc_interp)

    # Collect all boundary facets from gmsh tags (gate=10, walls=2)
    gate_facets = facet_tags.find(10)
    wall_facets = facet_tags.find(2)
    all_facets = np.unique(np.concatenate([gate_facets, wall_facets]))

    if len(all_facets) == 0:
        # Fallback: locate all 6 box faces geometrically
        tol_geom = 1e-13
        def _bndry_marker(x):
            return (
                np.isclose(x[0], X_MIN, atol=tol_geom) |
                np.isclose(x[0], X_MAX, atol=tol_geom) |
                np.isclose(x[1], Y_MIN, atol=tol_geom) |
                np.isclose(x[1], Y_MAX, atol=tol_geom) |
                np.isclose(x[2], Z_MIN, atol=tol_geom) |
                np.isclose(x[2], Z_MAX, atol=tol_geom)
            )
        bndry_dofs = fem.locate_dofs_geometrical(V, _bndry_marker)
    else:
        bndry_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, all_facets)

    bc = fem.dirichletbc(phi_bc_fn, bndry_dofs)

    # ── Laplace variational problem (rho = 0) ────────────────────────────────
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    # Zero right-hand side (pure Laplace, no free charge)
    L_form = fem.form(
        fem.Constant(msh, default_scalar_type(0.0)) * v * ufl.dx
    )

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b_vec = assemble_vector(L_form)
    apply_lifting(b_vec, [a], bcs=[[bc]])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_vec, [bc])

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()
    ksp.setOperators(A)
    x_sol = A.createVecRight()
    ksp.solve(b_vec, x_sol)

    phi = fem.Function(V, name="phi")
    phi.x.petsc_vec.array[:] = x_sol.array
    phi.x.scatter_forward()

    t_solve = time.perf_counter() - t0

    # ── Probe-line error evaluation ───────────────────────────────────────────
    probe_pts = make_probe_lines()                          # (N, 3) SI
    phi_analytic_probe = phi_halfspace_gate(
        probe_pts[:, 0], probe_pts[:, 1], probe_pts[:, 2]
    )
    near_sing = is_near_singularity(probe_pts)

    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(msh, msh.topology.dim)

    # Collect valid probe indices and cell IDs (local rank only)
    local_errs: list[float] = []
    for idx in range(len(probe_pts)):
        if near_sing[idx]:
            continue
        pt = probe_pts[idx]
        collisions = compute_collisions_points(tree, pt.reshape(1, 3))
        colliding = compute_colliding_cells(msh, collisions, pt.reshape(1, 3))
        if len(colliding.links(0)) == 0:
            continue
        cell = colliding.links(0)[0]
        v_fem = phi.eval(pt.reshape(1, 3), np.array([cell], dtype=np.int32))
        local_errs.append(abs(float(v_fem.ravel()[0]) - phi_analytic_probe[idx]))

    # Gather errors across all MPI ranks
    all_errs_lists = COMM.gather(local_errs, root=0)
    if COMM.rank == 0:
        all_errs = []
        for lst in all_errs_lists:
            all_errs.extend(lst)
        if all_errs:
            arr = np.array(all_errs)
            max_err = float(np.max(arr))
            mean_err = float(np.mean(arr))
            rms_err = float(np.sqrt(np.mean(arr ** 2)))
        else:
            max_err = mean_err = rms_err = float("nan")
    else:
        max_err = mean_err = rms_err = float("nan")

    # Broadcast errors from rank 0 to all ranks for consistent return value
    max_err = COMM.bcast(max_err, root=0)
    mean_err = COMM.bcast(mean_err, root=0)
    rms_err = COMM.bcast(rms_err, root=0)

    # Observed convergence rate
    if (prev_h is not None and prev_rms is not None
            and rms_err > 0 and prev_rms > 0 and not np.isnan(rms_err)):
        rate = np.log(prev_rms / rms_err) / np.log(prev_h / h_nm)
    else:
        rate = float("nan")

    # ── Optional XDMF output ─────────────────────────────────────────────────
    if args.write_xdmf:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from dolfinx.io import XDMFFile

        # Error function for visualisation
        phi_ref_fn = fem.Function(V, name="phi_analytic")
        phi_ref_fn.interpolate(phi_bc_interp)  # uses same analytic formula
        err_fn = fem.Function(V, name="error")
        err_fn.x.array[:] = phi.x.array - phi_ref_fn.x.array

        fname = outdir / f"sqgate_h{h_nm:.1f}nm.xdmf"
        with XDMFFile(msh.comm, str(fname), "w") as xf:
            xf.write_mesh(msh)
            xf.write_function(phi)
            xf.write_function(err_fn)

    return {
        "h_nm": h_nm,
        "ndofs": ndofs,
        "ncells": ncells,
        "max_err": max_err,
        "mean_err": mean_err,
        "rms_err": rms_err,
        "rate": rate,
        "time_s": t_solve,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEM h-convergence vs analytic half-space gate potential"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="h=[20,10,5]nm (default)")
    mode.add_argument("--full", action="store_true",
                      help="h=[20,10,5,2.5]nm")
    parser.add_argument("--outdir", default="tests/test_square_gate_h_convergence/output",
                        help="Output directory for XDMF/CSV files")
    parser.add_argument("--write-xdmf", action="store_true",
                        help="Write solution and error to XDMF")
    parser.add_argument("--save-csv", action="store_true",
                        help="Write results table to CSV")
    parser.add_argument("--max-dofs", type=int, default=2_000_000,
                        help="Skip mesh if global DOFs exceed this (default 2e6)")
    args = parser.parse_args()

    if args.full:
        h_values = [20.0, 10.0, 5.0, 2.5]
    else:
        h_values = [20.0, 10.0, 5.0]

    rows: list[dict] = []
    prev_h: float | None = None
    prev_rms: float | None = None

    for h in h_values:
        if COMM.rank == 0:
            print(f"\nh={h}nm  (h_far={min(h*8, 60):.1f}nm) ...", flush=True)

        r = run_h(h, args, prev_h, prev_rms)
        if r is None:
            continue

        rows.append(r)
        prev_h = h
        prev_rms = r["rms_err"]

    if COMM.rank == 0:
        print()
        print("=== Square gate h-convergence: FEM vs analytic half-space [V] ===")
        print("Gate: Wx=Wy=80nm, V_g=1V, box 300x300x150nm, uniform epsilon")
        print("Probe lines: x-line (z=40nm), z-line (x=y=0), off-axis (y=30nm,z=40nm)")
        print(f"Singularity exclusion radius: {SINGULARITY_EXCL_NM}nm around gate corners")
        print()

        hdr = (
            f"{'h_nm':>6} {'dofs':>9} {'cells':>9} "
            f"{'max_probe_err_V':>16} {'mean_probe_err_V':>17} {'rms_probe_err_V':>16} "
            f"{'observed_rate':>14} {'time_s':>8}"
        )
        print(hdr)
        print("-" * len(hdr))

        for r in rows:
            rate_str = f"{r['rate']:>14.2f}" if not np.isnan(r["rate"]) else f"{'—':>14}"
            print(
                f"{r['h_nm']:>6.1f} {r['ndofs']:>9} {r['ncells']:>9} "
                f"{r['max_err']:>16.3e} {r['mean_err']:>17.3e} {r['rms_err']:>16.3e} "
                f"{rate_str} {r['time_s']:>8.2f}"
            )

        print()
        print("Expected: rate ~1.5-2.0 for P1 FEM with edge singularities")
        print("Note: avoid evaluating error within 5nm of gate corners (field singularity)")

        if args.save_csv:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            csv_path = outdir / "square_gate_results.csv"
            fieldnames = [
                "h_nm", "ndofs", "ncells",
                "max_probe_err_V", "mean_probe_err_V", "rms_probe_err_V",
                "observed_rate", "time_s",
            ]
            csv_rows = [
                {
                    "h_nm": r["h_nm"],
                    "ndofs": r["ndofs"],
                    "ncells": r["ncells"],
                    "max_probe_err_V": r["max_err"],
                    "mean_probe_err_V": r["mean_err"],
                    "rms_probe_err_V": r["rms_err"],
                    "observed_rate": r["rate"],
                    "time_s": r["time_s"],
                }
                for r in rows
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
