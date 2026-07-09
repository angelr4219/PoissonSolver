"""
test_fft_vs_fem.py
==================
Show when FFT is trustworthy (coarse/fast comparisons) and when FEM-CoulombBC
beats the FFT floor.  Exposes the Gaussian smearing error floor of FFT.

Physics: sphere of charge Q=1e, R=10nm, in a cubic box of half-side 100nm.
Reference: free-space Coulomb potential.

Four solver variants (run in order):
  A) FFT           — spectral, periodic, uniform grid; Gaussian shell smearing
  B) FEM-ZeroWall  — phi=0 on walls (domain truncation error, should NOT converge)
  C) FEM-CoulombBC — phi=Q/(4pieps0 r) on walls (correct far-field, SHOULD converge)
  D) Analytic      — Coulomb reference at probe points only (no solve)

FFT: use fft_poisson_3d + sphere_charge_shell_grid + phi_analytic_sphere
     from src/poisson/fft_solve.py.  Subtract DC before error computation
     (periodic BCs fix potential up to a constant).

FEM: use sphere_refined_box from src/poisson/geom_sphere.py.

Expected:
  FFT:           fast (<1s), error stalls at ~1.4e-3 V regardless of h
                 (Gaussian smearing floor -- cannot be reduced by grid refinement)
  FEM-ZeroWall:  error ~flat ~1e-2 V (boundary condition dominates)
  FEM-CoulombBC: error DECREASES with h -- correct solver for MaSQE comparison

PASS/FAIL: FEM-ZeroWall at finest h in [1e-3, 2e-1] V; FEM-CoulombBC at finest h < FEM-ZeroWall/5.

Run:
    ./run_dolfinx.sh tests/test_fft_vs_fem.py --quick
    ./run_dolfinx.sh tests/test_fft_vs_fem.py --full
    ./run_dolfinx.sh tests/test_fft_vs_fem.py --quick --skip-fem
    ./run_dolfinx.sh tests/test_fft_vs_fem.py --quick --skip-fft
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
from poisson.geom_sphere import sphere_refined_box
from poisson.fft_solve import fft_poisson_3d, sphere_charge_shell_grid, phi_analytic_sphere

COMM = MPI.COMM_WORLD

# ── Physical constants / geometry ─────────────────────────────────────────────
EPS0 = 8.8541878128e-12   # F/m
Q = 1.6e-19               # C  (one elementary charge)
R_NM = 10.0
BOX_NM = 100.0
R_SI = R_NM * 1e-9
BOX_SI = BOX_NM * 1e-9

PROBE_NEAR_R = 2.0 * R_SI   # 20 nm
PROBE_FAR_R = 50.0e-9        # 50 nm


# ── Analytic reference ────────────────────────────────────────────────────────

def phi_analytic_pts(pts: np.ndarray) -> np.ndarray:
    """Free-space Coulomb potential at pts (N, 3) [SI] -> (N,) [V]."""
    r = np.linalg.norm(pts, axis=1)
    return np.where(
        r >= R_SI,
        Q / (4.0 * np.pi * EPS0 * np.where(r > 0, r, 1.0)),
        Q / (4.0 * np.pi * EPS0 * R_SI),
    )


# ── Probe rings ───────────────────────────────────────────────────────────────

def make_probe_ring(radius: float, n: int = 8) -> np.ndarray:
    """Return (n, 3) ring of points in the x-y plane at the given radius."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros(n),
    ])


# ── Shared FEM helpers ────────────────────────────────────────────────────────

def _source_ufl(msh, h_nm: float):
    """Gaussian shell source term matching sphere_charge_shell_grid."""
    sigma = max(h_nm * 0.5, 0.1) * 1e-9
    x = ufl.SpatialCoordinate(msh)
    r2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    return (Q / (sigma ** 3 * (2.0 * np.pi) ** 1.5)) * ufl.exp(-0.5 * r2 / sigma ** 2)


def _ksp_solve(A, b, comm):
    """CG + hypre solver; returns solution PETSc Vec."""
    ksp = PETSc.KSP().create(comm)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()
    ksp.setOperators(A)
    x = A.createVecRight()
    ksp.solve(b, x)
    return x


def _rms_global(phi_h: fem.Function, V: fem.FunctionSpace) -> float:
    """Global RMS error vs free-space Coulomb over all DOFs (allreduce)."""
    coords = V.tabulate_dof_coordinates()
    vals = phi_h.x.array.real[: len(coords)]
    ref = phi_analytic_pts(coords)
    diff = vals - ref
    ss = COMM.allreduce(float(np.sum(diff ** 2)), op=MPI.SUM)
    n = COMM.allreduce(float(len(diff)), op=MPI.SUM)
    return float(np.sqrt(ss / n))


def _max_global(phi_h: fem.Function, V: fem.FunctionSpace) -> float:
    """Global max absolute error vs free-space Coulomb (allreduce)."""
    coords = V.tabulate_dof_coordinates()
    vals = phi_h.x.array.real[: len(coords)]
    ref = phi_analytic_pts(coords)
    return float(COMM.allreduce(float(np.max(np.abs(vals - ref))), op=MPI.MAX))


def _probe_error_fem(phi_h: fem.Function, msh, probe_pts: np.ndarray) -> float:
    """
    Average absolute error at probe_pts (N, 3) [SI] using BoundingBoxTree.

    MPI-safe: each rank evaluates local cells; results are allreduced so all
    ranks return the same value.
    """
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(msh, msh.topology.dim)
    refs = phi_analytic_pts(probe_pts)

    total_err = 0.0
    n_found = 0
    for idx, pt in enumerate(probe_pts):
        collisions = compute_collisions_points(tree, pt.reshape(1, 3))
        colliding = compute_colliding_cells(msh, collisions, pt.reshape(1, 3))
        if len(colliding.links(0)) > 0:
            v = phi_h.eval(
                pt.reshape(1, 3),
                np.array([colliding.links(0)[0]], dtype=np.int32),
            )
            local_err = abs(float(v.ravel()[0]) - refs[idx])
            local_found = 1
        else:
            local_err = 0.0
            local_found = 0

        # Gather across ranks so the result is identical on all processes
        global_err = COMM.allreduce(local_err, op=MPI.SUM)
        global_found = COMM.allreduce(local_found, op=MPI.SUM)
        if global_found > 0:
            total_err += global_err / global_found
            n_found += 1

    return total_err / n_found if n_found > 0 else float("nan")


# ── Solver A: FFT ─────────────────────────────────────────────────────────────

def run_fft(h_nm: float, args) -> dict:
    """
    Spectral Poisson solve on a uniform periodic grid.

    Runs on MPI rank 0 only; other ranks return {}.
    DC offset removed before error computation (periodic BCs fix phi up
    to an additive constant).
    """
    if COMM.rank != 0:
        return {}

    N = max(int(round(2.0 * BOX_NM / h_nm)), 4)
    t0 = time.perf_counter()

    rho, xyz = sphere_charge_shell_grid(N, BOX_SI, R_SI, Q, sigma_nm=0.5)
    dx = xyz[1] - xyz[0]
    phi_grid = fft_poisson_3d(rho, dx)

    # Analytic reference on the same uniform grid
    phi_ref_grid = phi_analytic_sphere(xyz, R_SI, Q)

    # DC removal: shift phi_grid so its mean matches the analytic mean
    dc = np.mean(phi_grid) - np.mean(phi_ref_grid)
    phi_dc = phi_grid - dc

    diff = phi_dc - phi_ref_grid
    rms_err = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))

    # Probe error via trilinear interpolation on uniform grid
    def _grid_interp(pts_si: np.ndarray) -> np.ndarray:
        L = BOX_SI
        out = np.empty(len(pts_si))
        for k, pt in enumerate(pts_si):
            ix = (pt[0] + L) / (2.0 * L) * N
            iy = (pt[1] + L) / (2.0 * L) * N
            iz = (pt[2] + L) / (2.0 * L) * N
            ix = float(np.clip(ix, 0, N - 1.001))
            iy = float(np.clip(iy, 0, N - 1.001))
            iz = float(np.clip(iz, 0, N - 1.001))
            x0i, y0i, z0i = int(ix), int(iy), int(iz)
            fx, fy, fz = ix - x0i, iy - y0i, iz - z0i
            x1i = min(x0i + 1, N - 1)
            y1i = min(y0i + 1, N - 1)
            z1i = min(z0i + 1, N - 1)
            out[k] = (
                phi_dc[x0i, y0i, z0i] * (1 - fx) * (1 - fy) * (1 - fz) +
                phi_dc[x1i, y0i, z0i] * fx * (1 - fy) * (1 - fz) +
                phi_dc[x0i, y1i, z0i] * (1 - fx) * fy * (1 - fz) +
                phi_dc[x0i, y0i, z1i] * (1 - fx) * (1 - fy) * fz +
                phi_dc[x1i, y1i, z0i] * fx * fy * (1 - fz) +
                phi_dc[x0i, y1i, z1i] * (1 - fx) * fy * fz +
                phi_dc[x1i, y0i, z1i] * fx * (1 - fy) * fz +
                phi_dc[x1i, y1i, z1i] * fx * fy * fz
            )
        return out

    near_pts = make_probe_ring(PROBE_NEAR_R)
    far_pts = make_probe_ring(PROBE_FAR_R)
    probe_near = float(np.mean(np.abs(_grid_interp(near_pts) - phi_analytic_pts(near_pts))))
    probe_far = float(np.mean(np.abs(_grid_interp(far_pts) - phi_analytic_pts(far_pts))))

    t_run = time.perf_counter() - t0

    if args.write_xdmf:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_xdmf_grid(
            phi_dc, xyz,
            outdir / f"fft_vs_fem_FFT_h{h_nm:.1f}nm.xdmf",
        )

    return {
        "solver": "FFT",
        "h_nm": h_nm,
        "ndofs": N ** 3,
        "rms_err": rms_err,
        "max_err": max_err,
        "probe_near": probe_near,
        "probe_far": probe_far,
        "time_s": t_run,
        "notes": f"N={N}, DC-removed",
    }


def _write_xdmf_grid(phi_grid: np.ndarray, xyz: np.ndarray, fname: Path) -> None:
    """Write a uniform 3-D grid to XDMF + HDF5 (3DCoRectMesh format)."""
    try:
        import h5py
    except ImportError:
        print(f"  [warn] h5py not available; skipping XDMF write for {fname.name}")
        return

    h5_path = fname.with_suffix(".h5")
    N = phi_grid.shape[0]
    dx = float(xyz[1] - xyz[0])
    origin = float(xyz[0])
    stem = fname.stem

    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("phi", data=phi_grid.astype(np.float64))

    xdmf_text = (
        '<?xml version="1.0"?>\n'
        '<Xdmf Version="3.0">\n'
        '  <Domain>\n'
        f'    <Grid Name="{stem}" GridType="Uniform">\n'
        f'      <Topology TopologyType="3DCoRectMesh" Dimensions="{N} {N} {N}"/>\n'
        '      <Geometry GeometryType="Origin_DxDyDz">\n'
        f'        <DataItem Format="XML" Dimensions="3">'
        f'{origin} {origin} {origin}</DataItem>\n'
        f'        <DataItem Format="XML" Dimensions="3">'
        f'{dx} {dx} {dx}</DataItem>\n'
        '      </Geometry>\n'
        '      <Attribute Name="phi" AttributeType="Scalar" Center="Node">\n'
        f'        <DataItem Format="HDF" Dimensions="{N} {N} {N}">'
        f'{h5_path.name}:/phi</DataItem>\n'
        '      </Attribute>\n'
        '    </Grid>\n'
        '  </Domain>\n'
        '</Xdmf>\n'
    )
    fname.write_text(xdmf_text)


# ── Solver B: FEM-ZeroWall ────────────────────────────────────────────────────

def run_fem_zerowall(h_nm: float, args) -> dict:
    """FEM solve with phi=0 on all box walls (domain truncation error)."""
    h_far_nm = min(h_nm * 10.0, BOX_NM / 2.0)
    t0 = time.perf_counter()

    msh, _, ftags = sphere_refined_box(
        COMM, BOX_NM, R_NM, h_near_nm=h_nm, h_far_nm=h_far_nm
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    ndofs = V.dofmap.index_map.size_global

    if ndofs > args.max_dofs:
        if COMM.rank == 0:
            print(f"  [skip] FEM-ZeroWall h={h_nm}nm: {ndofs} DOFs > --max-dofs",
                  flush=True)
        return {}

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    wall_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, ftags.find(1))
    bc = fem.dirichletbc(default_scalar_type(0.0), wall_dofs, V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_form = fem.form(_source_ufl(msh, h_nm) * v * ufl.dx)

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b_vec = assemble_vector(L_form)
    apply_lifting(b_vec, [a], bcs=[[bc]])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_vec, [bc])

    phi = fem.Function(V, name="phi_zerowall")
    phi.x.petsc_vec.array[:] = _ksp_solve(A, b_vec, msh.comm).array
    phi.x.scatter_forward()

    rms_err = _rms_global(phi, V)
    max_err = _max_global(phi, V)
    probe_near = _probe_error_fem(phi, msh, make_probe_ring(PROBE_NEAR_R))
    probe_far = _probe_error_fem(phi, msh, make_probe_ring(PROBE_FAR_R))
    t_run = time.perf_counter() - t0

    if args.write_xdmf:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from dolfinx.io import XDMFFile
        fname = outdir / f"fft_vs_fem_FEM_ZeroWall_h{h_nm:.1f}nm.xdmf"
        with XDMFFile(msh.comm, str(fname), "w") as xf:
            xf.write_mesh(msh)
            xf.write_function(phi)

    return {
        "solver": "FEM-ZeroWall",
        "h_nm": h_nm,
        "ndofs": ndofs,
        "rms_err": rms_err,
        "max_err": max_err,
        "probe_near": probe_near,
        "probe_far": probe_far,
        "time_s": t_run,
        "notes": "",
    }


# ── Solver C: FEM-CoulombBC ───────────────────────────────────────────────────

def run_fem_coulombbc(h_nm: float, args) -> dict:
    """FEM solve with phi=Q/(4pieps0 r) on all box walls (exact far-field BC)."""
    h_far_nm = min(h_nm * 10.0, BOX_NM / 2.0)
    t0 = time.perf_counter()

    msh, _, ftags = sphere_refined_box(
        COMM, BOX_NM, R_NM, h_near_nm=h_nm, h_far_nm=h_far_nm
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    ndofs = V.dofmap.index_map.size_global

    if ndofs > args.max_dofs:
        if COMM.rank == 0:
            print(f"  [skip] FEM-CoulombBC h={h_nm}nm: {ndofs} DOFs > --max-dofs",
                  flush=True)
        return {}

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    # Interpolate exact Coulomb on wall DOFs
    phi_exact_fn = fem.Function(V, name="phi_coulomb_bc")

    def _coulomb_interp(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        return np.where(
            r >= R_SI,
            Q / (4.0 * np.pi * EPS0 * np.where(r > 0, r, 1.0)),
            Q / (4.0 * np.pi * EPS0 * R_SI),
        )

    phi_exact_fn.interpolate(_coulomb_interp)

    wall_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, ftags.find(1))
    bc = fem.dirichletbc(phi_exact_fn, wall_dofs)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_form = fem.form(_source_ufl(msh, h_nm) * v * ufl.dx)

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b_vec = assemble_vector(L_form)
    apply_lifting(b_vec, [a], bcs=[[bc]])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_vec, [bc])

    phi = fem.Function(V, name="phi_coulombbc")
    phi.x.petsc_vec.array[:] = _ksp_solve(A, b_vec, msh.comm).array
    phi.x.scatter_forward()

    rms_err = _rms_global(phi, V)
    max_err = _max_global(phi, V)
    probe_near = _probe_error_fem(phi, msh, make_probe_ring(PROBE_NEAR_R))
    probe_far = _probe_error_fem(phi, msh, make_probe_ring(PROBE_FAR_R))
    t_run = time.perf_counter() - t0

    if args.write_xdmf:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from dolfinx.io import XDMFFile
        fname = outdir / f"fft_vs_fem_FEM_CoulombBC_h{h_nm:.1f}nm.xdmf"
        with XDMFFile(msh.comm, str(fname), "w") as xf:
            xf.write_mesh(msh)
            xf.write_function(phi)

    return {
        "solver": "FEM-CoulombBC",
        "h_nm": h_nm,
        "ndofs": ndofs,
        "rms_err": rms_err,
        "max_err": max_err,
        "probe_near": probe_near,
        "probe_far": probe_far,
        "time_s": t_run,
        "notes": "",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FFT vs FEM comparison: sphere charge Q=1e, R=10nm, box=100nm"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="h=[50,25,10]nm (default)")
    mode.add_argument("--full", action="store_true",
                      help="h=[50,25,10,5]nm")
    parser.add_argument("--outdir", default="tests/test_fft_vs_fem/output",
                        help="Output directory for XDMF/CSV files")
    parser.add_argument("--write-xdmf", action="store_true",
                        help="Write solution fields to XDMF/HDF5")
    parser.add_argument("--save-csv", action="store_true",
                        help="Write results table to CSV")
    parser.add_argument("--max-dofs", type=int, default=2_000_000,
                        help="Skip FEM mesh if DOFs exceed this (default 2e6)")
    parser.add_argument("--skip-fem", action="store_true",
                        help="Skip all FEM runs (FFT only)")
    parser.add_argument("--skip-fft", action="store_true",
                        help="Skip FFT runs (FEM only)")
    args = parser.parse_args()

    if args.full:
        h_values = [50.0, 25.0, 10.0, 5.0]
    else:
        h_values = [50.0, 25.0, 10.0]

    # rows collected on rank 0 for FFT; on all ranks for FEM (errors are allreduced)
    rows: list[dict] = []

    for h in h_values:
        if COMM.rank == 0:
            print(f"\n--- h={h:.1f}nm ---", flush=True)

        # A) FFT — rank 0 only
        if not args.skip_fft:
            fft_result = run_fft(h, args)
            if COMM.rank == 0 and fft_result:
                rows.append(fft_result)

        # B) FEM-ZeroWall
        if not args.skip_fem:
            r = run_fem_zerowall(h, args)
            if r:
                rows.append(r)   # same dict on all ranks (allreduce errors)

        # C) FEM-CoulombBC
        if not args.skip_fem:
            r = run_fem_coulombbc(h, args)
            if r:
                rows.append(r)

    # ── Print table (rank 0 only) ─────────────────────────────────────────────
    if COMM.rank == 0:
        print()
        print("=== FFT vs FEM: sphere Q=1e, R=10nm, box half-side=100nm ===")
        print("Errors vs free-space Coulomb analytic [V]   (FFT: DC-removed)")
        print()

        hdr = (
            f"{'solver':<16} {'h_nm':>6} {'dofs':>10} "
            f"{'rms_err_V':>11} {'max_err_V':>11} "
            f"{'probe_near_V':>13} {'probe_far_V':>12} "
            f"{'time_s':>8}  notes"
        )
        sep = "-" * (len(hdr) + 10)
        print(hdr)
        print(sep)

        for r in rows:
            print(
                f"{r['solver']:<16} {r['h_nm']:>6.1f} {r['ndofs']:>10} "
                f"{r['rms_err']:>11.3e} {r['max_err']:>11.3e} "
                f"{r['probe_near']:>13.3e} {r['probe_far']:>12.3e} "
                f"{r['time_s']:>8.2f}  {r['notes']}"
            )

        print()
        print("Expected:")
        print("  FFT:          fast (<1s), error stalls at ~1.4e-3V"
              " (Gaussian smearing floor regardless of h)")
        print("  FEM-ZeroWall: error ~flat ~1e-2V"
              " (domain truncation dominates, not discretisation)")
        print("  FEM-CoulombBC: error should DECREASE with h"
              " -- this is the correct solver for MaSQE comparison")
        print()

        # PASS/FAIL
        # 1. FEM-ZeroWall at finest h stays in domain-truncation regime (~1e-2 V)
        # 2. FEM-CoulombBC at finest h < FEM-ZeroWall at finest h / 5
        fem_zw_rows = sorted(
            [r for r in rows if r["solver"] == "FEM-ZeroWall" and not np.isnan(r["rms_err"])],
            key=lambda r: r["h_nm"],
        )
        fem_cbc_rows = sorted(
            [r for r in rows if r["solver"] == "FEM-CoulombBC" and not np.isnan(r["rms_err"])],
            key=lambda r: r["h_nm"],
        )

        failures: list[str] = []

        if fem_zw_rows and fem_cbc_rows:
            finest_zw  = fem_zw_rows[0]   # smallest h_nm = finest mesh
            finest_cbc = fem_cbc_rows[0]  # smallest h_nm = finest mesh

            # Check 1: FEM-ZeroWall at finest h in [1e-3, 2e-1] V
            zw_rms = finest_zw["rms_err"]
            if not (1e-3 <= zw_rms <= 2e-1):
                failures.append(
                    f"FEM-ZeroWall h={finest_zw['h_nm']:.0f}nm RMS={zw_rms:.3e} V "
                    f"not in expected range [1e-3, 2e-1] V"
                )

            # Check 2: FEM-CoulombBC finest < FEM-ZeroWall finest / 5
            cbc_rms   = finest_cbc["rms_err"]
            threshold = zw_rms / 5.0
            if cbc_rms >= threshold:
                failures.append(
                    f"FEM-CoulombBC h={finest_cbc['h_nm']:.0f}nm RMS={cbc_rms:.3e} V "
                    f">= FEM-ZeroWall/5 = {threshold:.3e} V"
                )
        else:
            failures.append("insufficient FEM data to evaluate pass/fail criteria")

        if failures:
            print("FAIL")
            for msg in failures:
                print(f"  {msg}")
            sys.exit(1)
        else:
            print("PASS")

        if args.save_csv:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            csv_path = outdir / "fft_vs_fem_results.csv"
            fieldnames = [
                "solver", "h_nm", "ndofs",
                "rms_err_V", "max_err_V",
                "probe_near_V", "probe_far_V",
                "time_s", "notes",
            ]
            csv_rows = [
                {
                    "solver": r["solver"],
                    "h_nm": r["h_nm"],
                    "ndofs": r["ndofs"],
                    "rms_err_V": r["rms_err"],
                    "max_err_V": r["max_err"],
                    "probe_near_V": r["probe_near"],
                    "probe_far_V": r["probe_far"],
                    "time_s": r["time_s"],
                    "notes": r["notes"],
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
