"""
test_bc_comparison_neutral_periodic.py
========================================
Compare periodic FEM (dolfinx_mpc) and FFT for a charge-neutral dipole source.

Periodic BC is only physically valid when the source has zero net charge.
This test proves that FEM-Periodic and FFT agree when both solve the same
problem, and shows how FEM-LargeBox (isolated box, zero-wall BC) compares.

Physics
-------
Gaussian dipole in a cubic box of half-side BOX = 100 nm:
  +Q Gaussian centred at (-d/2, 0, 0)
  -Q Gaussian centred at (+d/2, 0, 0)
  d = 30 nm,  σ = max(h*0.5, 1.0) nm for both.
  Net charge = 0  (required for periodic BC to be well-posed).

Variants
--------
A) FEM-Periodic : dolfinx_mpc MultiPointConstraint (all 3 axes), gauge
                  fixed by pinning φ = 0 at the mesh node nearest the
                  origin (where dipole potential is 0 by symmetry).
B) FFT-Periodic : fft_poisson_3d on a uniform N³ grid; DC removed.
C) FEM-LargeBox : dipole in a 300 nm half-side box, zero-wall BC.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import csv
import time
from typing import NamedTuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import (
    assemble_vector,
    apply_lifting,
    set_bc,
    LinearProblem,
)
from dolfinx.mesh import create_box, CellType, locate_entities_boundary
import ufl
import dolfinx_mpc

from poisson.geom_sphere import sphere_refined_box
from poisson.fft_solve import fft_poisson_3d

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EPS0   = 8.8541878128e-12   # F/m
Q      = 1.6e-19             # C (one elementary charge)
R_NM   = 10.0                # reference radius [nm] (unused in mesh here)
BOX_NM = 100.0               # half-side of periodic box [nm]
D_NM   = 30.0                # dipole separation [nm]
BOX_SI = BOX_NM * 1e-9       # [m]
D_SI   = D_NM  * 1e-9        # [m]


# ---------------------------------------------------------------------------
# Source term helpers
# ---------------------------------------------------------------------------

def _dipole_rho_fn(sigma_si: float):
    """Return a callable suitable for fem.Function.interpolate()."""
    norm = Q / (sigma_si**3 * (2.0 * np.pi) ** 1.5)

    def _rho(x):
        # +Q at (-D_SI/2, 0, 0)
        r2_pos = (x[0] + D_SI / 2)**2 + x[1]**2 + x[2]**2
        # -Q at (+D_SI/2, 0, 0)
        r2_neg = (x[0] - D_SI / 2)**2 + x[1]**2 + x[2]**2
        return norm * (np.exp(-0.5 * r2_pos / sigma_si**2)
                       - np.exp(-0.5 * r2_neg / sigma_si**2))

    return _rho


def _build_fft_rho(N: int, box_si: float, sigma_si: float) -> np.ndarray:
    """
    Build an N³ Gaussian-dipole charge density on a uniform periodic grid
    [-box_si, box_si)³  with grid spacing  dx = 2·box_si / N.
    """
    xyz = np.linspace(-box_si, box_si, N, endpoint=False)
    dx  = xyz[1] - xyz[0]
    X, Y, Z = np.meshgrid(xyz, xyz, xyz, indexing="ij")

    norm = Q / (sigma_si**3 * (2.0 * np.pi) ** 1.5)

    r2_pos = (X + D_SI / 2)**2 + Y**2 + Z**2
    r2_neg = (X - D_SI / 2)**2 + Y**2 + Z**2

    rho = norm * (np.exp(-0.5 * r2_pos / sigma_si**2)
                  - np.exp(-0.5 * r2_neg / sigma_si**2))
    return rho, xyz, dx


# ---------------------------------------------------------------------------
# Geometry evaluation helper
# ---------------------------------------------------------------------------

def _eval_at_points(phi_fn: fem.Function, points: np.ndarray) -> np.ndarray:
    """
    Evaluate a dolfinx Function at an array of 3-D points (shape [N, 3]).

    Points that fall outside the mesh are returned as NaN.
    Works in serial; in parallel only returns non-NaN for locally owned cells.
    """
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    msh       = phi_fn.function_space.mesh
    pts       = np.asarray(points, dtype=np.float64)
    tree      = bb_tree(msh, msh.topology.dim)
    cands     = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh, cands, pts)

    vals = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        cells = colliding.links(i)
        if len(cells) > 0:
            v = phi_fn.eval(pts[i : i + 1, :], cells[:1])
            vals[i] = float(v[0] if v.ndim == 1 else v[0, 0])
    return vals


# ---------------------------------------------------------------------------
# FEM-Periodic solver (dolfinx_mpc)
# ---------------------------------------------------------------------------

class FEMPeriodicResult(NamedTuple):
    phi:     fem.Function
    n_dofs:  int
    elapsed: float


def solve_fem_periodic(
    h_nm: float,
    comm: MPI.Comm,
    max_dofs: int,
) -> FEMPeriodicResult:
    """
    Solve the periodic Poisson problem with dolfinx_mpc.

    Uses a structured box mesh so periodic node pairs align exactly.
    Gauge is fixed by pinning φ = 0 at the node nearest the origin.
    """
    sigma_si = max(h_nm * 0.5, 1.0) * 1e-9
    N        = max(4, int(round(2.0 * BOX_NM / h_nm)))   # cells per side

    t0  = time.perf_counter()
    msh = create_box(
        comm,
        [np.array([-BOX_SI, -BOX_SI, -BOX_SI]),
         np.array([ BOX_SI,  BOX_SI,  BOX_SI])],
        [N, N, N],
        CellType.tetrahedron,
    )

    V      = fem.functionspace(msh, ("Lagrange", 1))
    n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if n_dofs > max_dofs:
        raise RuntimeError(
            f"FEM-Periodic DOFs {n_dofs} exceed --max-dofs {max_dofs}."
        )

    # ---------------------------------------------------------- Periodic MPC
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    tol = 1.0e-10

    # x-periodicity: slave = face x=+BOX_SI, master = face x=-BOX_SI
    def _slave_x(x):
        return np.isclose(x[0], BOX_SI, atol=tol)

    def _map_x(x):
        out = np.copy(x)
        out[0] -= 2.0 * BOX_SI
        return out

    # y-periodicity
    def _slave_y(x):
        return np.isclose(x[1], BOX_SI, atol=tol)

    def _map_y(x):
        out = np.copy(x)
        out[1] -= 2.0 * BOX_SI
        return out

    # z-periodicity
    def _slave_z(x):
        return np.isclose(x[2], BOX_SI, atol=tol)

    def _map_z(x):
        out = np.copy(x)
        out[2] -= 2.0 * BOX_SI
        return out

    mpc.create_periodic_constraint_geometrical(V, _slave_x, _map_x, bcs=[], scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, _slave_y, _map_y, bcs=[], scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, _slave_z, _map_z, bcs=[], scale=1.0)
    mpc.finalize()

    # ------------------------------------------------- Gauge fix: φ(0,0,0) = 0
    # Dipole is symmetric → φ = 0 exactly at origin.
    dof_coords  = V.tabulate_dof_coordinates()  # (n_local, 3)
    r_dofs      = np.sqrt(
        dof_coords[:, 0]**2 + dof_coords[:, 1]**2 + dof_coords[:, 2]**2
    )
    gauge_local = int(np.argmin(r_dofs))
    gauge_bc    = fem.dirichletbc(
        0.0,
        np.array([gauge_local], dtype=np.int32),
        V,
    )

    # ---------------------------------------------------------- Source & forms
    f = fem.Function(V)
    f.interpolate(_dipole_rho_fn(sigma_si))

    u  = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=msh)

    a_ufl  = EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_ufl  = f * v * dx
    a_form = fem.form(a_ufl)
    L_form = fem.form(L_ufl)

    # ---------------------------------------------------------- MPC assembly
    A = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=[gauge_bc])
    A.assemble()

    b = dolfinx_mpc.assemble_vector(L_form, mpc)
    apply_lifting(b, [a_form], bcs=[[gauge_bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [gauge_bc])

    # ------------------------------------------------------------------ KSP
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, max_it=50000)
    ksp.setFromOptions()

    phi     = fem.Function(V)
    phi.name = f"phi_FEM_Per_h{h_nm:.0f}nm"
    ksp.solve(b, phi.x.petsc_vec)
    phi.x.scatter_forward()

    # Back-substitute to fill slave DOF values from master DOF values
    mpc.backsubstitution(phi.x.petsc_vec)
    phi.x.scatter_forward()

    ksp.destroy()
    A.destroy()
    b.destroy()

    elapsed = time.perf_counter() - t0
    return FEMPeriodicResult(phi, n_dofs, elapsed)


# ---------------------------------------------------------------------------
# FFT solver
# ---------------------------------------------------------------------------

class FFTResult(NamedTuple):
    phi_grid: np.ndarray   # (N, N, N) [V]
    xyz:      np.ndarray   # (N,) coordinate array [m]
    dx:       float        # grid spacing [m]
    n_pts:    int          # N³
    elapsed:  float


def solve_fft(h_nm: float) -> FFTResult:
    """FFT Poisson solve on a uniform periodic N³ grid."""
    sigma_si = max(h_nm * 0.5, 1.0) * 1e-9
    t0       = time.perf_counter()

    rho, xyz, dx = _build_fft_rho(
        N      = int(round(2.0 * BOX_NM / h_nm)),
        box_si = BOX_SI,
        sigma_si = sigma_si,
    )

    phi_grid  = fft_poisson_3d(rho, dx)
    phi_grid -= phi_grid.mean()    # DC removal (neutral periodic → mean = 0)

    elapsed = time.perf_counter() - t0
    N       = rho.shape[0]
    return FFTResult(phi_grid, xyz, dx, N**3, elapsed)


# ---------------------------------------------------------------------------
# FEM-LargeBox solver (reference isolated solution)
# ---------------------------------------------------------------------------

class FEMLargeBoxResult(NamedTuple):
    phi:     fem.Function
    n_dofs:  int
    elapsed: float


def solve_fem_largebox(
    h_nm: float,
    comm: MPI.Comm,
    max_dofs: int,
) -> FEMLargeBoxResult:
    """
    Dipole in a 3× larger box (300 nm half-side) with zero-wall BC.
    Used as an isolated reference for line-probe comparisons.
    """
    sigma_si = max(h_nm * 0.5, 1.0) * 1e-9
    box_nm   = 3.0 * BOX_NM
    h_far_nm = min(h_nm * 10.0, 150.0)

    t0 = time.perf_counter()

    msh, _, facet_tags = sphere_refined_box(
        comm,
        box_nm=box_nm,
        sphere_r_nm=R_NM,
        h_near_nm=h_nm,
        h_far_nm=h_far_nm,
    )

    V      = fem.functionspace(msh, ("Lagrange", 1))
    n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if n_dofs > max_dofs:
        raise RuntimeError(
            f"FEM-LargeBox DOFs {n_dofs} exceed --max-dofs {max_dofs}."
        )

    wall_facets = facet_tags.find(1)
    wall_dofs   = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, wall_facets
    )
    bc  = fem.dirichletbc(0.0, wall_dofs, V)

    f = fem.Function(V)
    f.interpolate(_dipole_rho_fn(sigma_si))

    u  = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=msh)

    a = EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = f * v * dx

    problem  = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"nplb_h{h_nm:.0f}_",
        petsc_options={
            "ksp_type":   "cg",
            "pc_type":    "hypre",
            "ksp_rtol":   1e-10,
            "ksp_max_it": 50000,
        },
    )
    phi      = problem.solve()
    phi.name = f"phi_LargeBox_h{h_nm:.0f}nm"
    elapsed  = time.perf_counter() - t0

    return FEMLargeBoxResult(phi, n_dofs, elapsed)


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compare_fem_vs_fft(
    phi_fem: fem.Function,
    fft_res: FFTResult,
) -> float:
    """
    Interpolate FFT solution at every FEM DOF coordinate;
    return RMS difference (FEM − FFT).
    """
    from scipy.interpolate import RegularGridInterpolator

    dof_coords = phi_fem.function_space.tabulate_dof_coordinates()   # (N, 3)
    phi_fem_vals = phi_fem.x.array                                    # (N,)

    interp = RegularGridInterpolator(
        (fft_res.xyz, fft_res.xyz, fft_res.xyz),
        fft_res.phi_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    phi_fft_at_fem = interp(dof_coords)

    mask = np.isfinite(phi_fft_at_fem)
    diff = phi_fem_vals[mask] - phi_fft_at_fem[mask]
    return float(np.sqrt(np.mean(diff**2)))


def _line_probe_rms(
    phi_a_vals: np.ndarray,
    phi_b_vals: np.ndarray,
) -> float:
    mask = np.isfinite(phi_a_vals) & np.isfinite(phi_b_vals)
    if mask.sum() == 0:
        return float("nan")
    diff = phi_a_vals[mask] - phi_b_vals[mask]
    return float(np.sqrt(np.mean(diff**2)))


def compare_on_line_probe(
    phi_fem_per: fem.Function,
    fft_res: FFTResult,
    phi_lb: fem.Function,
    n_probe: int = 200,
) -> tuple[float, float, float]:
    """
    Compare all three solutions on a line probe:
      x ∈ [-BOX_SI, BOX_SI],  y = z = 0.

    Returns (fem_per_vs_lb_rms, fft_vs_lb_rms, fem_per_vs_fft_line_rms).
    """
    from scipy.interpolate import RegularGridInterpolator

    x_probe  = np.linspace(-BOX_SI, BOX_SI, n_probe)
    pts      = np.column_stack(
        [x_probe, np.zeros(n_probe), np.zeros(n_probe)]
    )

    # Evaluate FEM-Periodic on line probe
    vals_per = _eval_at_points(phi_fem_per, pts)

    # Evaluate FFT on line probe (1-D slice)
    interp_fft = RegularGridInterpolator(
        (fft_res.xyz, fft_res.xyz, fft_res.xyz),
        fft_res.phi_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    vals_fft = interp_fft(pts)

    # Evaluate FEM-LargeBox on line probe
    vals_lb  = _eval_at_points(phi_lb, pts)

    fem_per_vs_lb_rms      = _line_probe_rms(vals_per, vals_lb)
    fft_vs_lb_rms          = _line_probe_rms(vals_fft, vals_lb)
    fem_per_vs_fft_line_rms = _line_probe_rms(vals_per, vals_fft)

    return fem_per_vs_lb_rms, fft_vs_lb_rms, fem_per_vs_fft_line_rms


# ---------------------------------------------------------------------------
# XDMF / HDF5 output helpers
# ---------------------------------------------------------------------------

def _write_fem_xdmf(phi: fem.Function, path: Path, comm: MPI.Comm) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with dolfinx.io.XDMFFile(comm, str(path), "w") as xf:
        xf.write_mesh(phi.function_space.mesh)
        xf.write_function(phi)


def _write_fft_xdmf(
    fft_res: FFTResult,
    path_xdmf: Path,
    h_nm: float,
) -> None:
    """
    Write FFT grid to HDF5 + XDMF using the 3DCoRectMesh topology,
    consistent with the sphere_triple_comparison.py FFT writer.
    """
    import h5py

    path_xdmf.parent.mkdir(parents=True, exist_ok=True)
    h5_path = path_xdmf.with_suffix(".h5")
    N       = fft_res.phi_grid.shape[0]
    origin  = float(fft_res.xyz[0])
    dx      = float(fft_res.dx)

    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset("phi", data=fft_res.phi_grid, compression="gzip")

    # XDMF: 3DCoRectMesh (node-centred, Fortran/C ordering = Nz Ny Nx)
    xdmf_str = f"""<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="fft_phi_h{h_nm:.0f}nm" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{N} {N} {N}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Format="XML" Dimensions="3">{origin:.6e} {origin:.6e} {origin:.6e}</DataItem>
        <DataItem Format="XML" Dimensions="3">{dx:.6e} {dx:.6e} {dx:.6e}</DataItem>
      </Geometry>
      <Attribute Name="phi" AttributeType="Scalar" Center="Node">
        <DataItem Format="HDF" Dimensions="{N} {N} {N}">{h5_path.name}:/phi</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    path_xdmf.write_text(xdmf_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Periodic comparison: FEM-Periodic vs FFT vs FEM-LargeBox\n"
            "Source: Gaussian dipole (net charge = 0).  "
            "Periodic BC valid only for charge-neutral problems."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true", default=True,
        help="h = [25, 10] nm  (default)",
    )
    mode.add_argument(
        "--full", action="store_true",
        help="h = [25, 10, 5] nm",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("tests/test_bc_comparison_neutral_periodic/output"),
    )
    parser.add_argument("--write-xdmf", action="store_true")
    parser.add_argument("--save-csv",   action="store_true")
    parser.add_argument("--max-dofs",   type=int, default=1_000_000)
    args = parser.parse_args()

    h_values = [25.0, 10.0, 5.0] if args.full else [25.0, 10.0]
    comm     = MPI.COMM_WORLD

    # -------------------------------------------------------------- header
    col_w  = [6, 9, 9, 20, 24, 20, 10, 10]
    header = (
        f"{'h_nm':>{col_w[0]}} | {'FEM_dofs':>{col_w[1]}} | "
        f"{'FFT_pts':>{col_w[2]}} | {'FEM_Per_vs_FFT_rms':>{col_w[3]}} | "
        f"{'FEM_Per_vs_LargeBox_rms':>{col_w[4]}} | "
        f"{'FFT_vs_LargeBox_rms':>{col_w[5]}} | "
        f"{'t_fem_s':>{col_w[6]}} | {'t_fft_s':>{col_w[7]}}"
    )
    sep = "-" * len(header)

    if comm.rank == 0:
        print()
        print("=" * len(header))
        print("Periodic comparison: FEM-Periodic vs FFT vs FEM-LargeBox")
        print("Dipole source (+Q at -15nm, -Q at +15nm), box half-side=100nm")
        print("=" * len(header))
        print(header)
        print(sep)

    rows: list[dict] = []

    for h_nm in h_values:
        if comm.rank == 0:
            print(f"\nh = {h_nm} nm", flush=True)

        # ---- FEM-Periodic ----
        if comm.rank == 0:
            print("  Solving FEM-Periodic ...", flush=True)
        try:
            per_res = solve_fem_periodic(h_nm, comm, args.max_dofs)
        except RuntimeError as exc:
            if comm.rank == 0:
                print(f"  SKIPPED FEM-Periodic: {exc}", flush=True)
            continue

        # ---- FFT ----
        if comm.rank == 0:
            print("  Solving FFT ...", flush=True)
        fft_res = solve_fft(h_nm)

        # ---- FEM-LargeBox ----
        if comm.rank == 0:
            print("  Solving FEM-LargeBox ...", flush=True)
        try:
            lb_res = solve_fem_largebox(h_nm, comm, args.max_dofs)
        except RuntimeError as exc:
            if comm.rank == 0:
                print(f"  SKIPPED FEM-LargeBox: {exc}", flush=True)
            continue

        # ---- Comparisons ----
        if comm.rank == 0:
            print("  Computing comparisons ...", flush=True)

        # FEM-Per vs FFT: RMS at all FEM DOF coordinates
        fem_per_vs_fft_rms = compare_fem_vs_fft(per_res.phi, fft_res)

        # Line-probe comparisons: x ∈ [-BOX, BOX], y=z=0
        fem_per_vs_lb_rms, fft_vs_lb_rms, _ = compare_on_line_probe(
            per_res.phi, fft_res, lb_res.phi
        )

        # ---- XDMF output ----
        if args.write_xdmf and comm.rank == 0:
            args.outdir.mkdir(parents=True, exist_ok=True)
            _write_fem_xdmf(
                per_res.phi,
                args.outdir / f"fem_periodic_h{h_nm:.0f}nm.xdmf",
                comm,
            )
            _write_fem_xdmf(
                lb_res.phi,
                args.outdir / f"fem_largebox_h{h_nm:.0f}nm.xdmf",
                comm,
            )
            _write_fft_xdmf(
                fft_res,
                args.outdir / f"fft_periodic_h{h_nm:.0f}nm.xdmf",
                h_nm,
            )
            print(f"  XDMF files written to {args.outdir}", flush=True)

        # ---- Print row ----
        if comm.rank == 0:
            row_str = (
                f"{h_nm:>{col_w[0]}.1f} | {per_res.n_dofs:>{col_w[1]}d} | "
                f"{fft_res.n_pts:>{col_w[2]}d} | "
                f"{fem_per_vs_fft_rms:>{col_w[3]}.4e} | "
                f"{fem_per_vs_lb_rms:>{col_w[4]}.4e} | "
                f"{fft_vs_lb_rms:>{col_w[5]}.4e} | "
                f"{per_res.elapsed:>{col_w[6]}.2f} | "
                f"{fft_res.elapsed:>{col_w[7]}.2f}"
            )
            print(row_str, flush=True)
            rows.append({
                "h_nm":                    h_nm,
                "FEM_dofs":               per_res.n_dofs,
                "FFT_pts":                fft_res.n_pts,
                "FEM_Per_vs_FFT_rms":     fem_per_vs_fft_rms,
                "FEM_Per_vs_LargeBox_rms": fem_per_vs_lb_rms,
                "FFT_vs_LargeBox_rms":    fft_vs_lb_rms,
                "t_fem_s":                per_res.elapsed,
                "t_fft_s":                fft_res.elapsed,
            })

    if comm.rank == 0:
        print(sep)

    # ---------------------------------------------------------------- CSV
    if args.save_csv and comm.rank == 0 and rows:
        args.outdir.mkdir(parents=True, exist_ok=True)
        csv_path = args.outdir / "neutral_periodic_results.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved: {csv_path}", flush=True)

    # -------------------------------------------------------- expected summary
    if comm.rank == 0:
        print()
        print("Expected behaviour")
        print("------------------")
        print(
            "FEM-Periodic and FFT should agree closely "
            "(both solve same periodic neutral problem)"
        )
        print(
            "LargeBox shows what isolated dipole looks like — "
            "periodic replicates it only when box >> dipole separation"
        )
        print()

        # PASS/FAIL: FEM_Per_vs_FFT_rms < 5% of max(|phi|) at h<=10nm
        # Estimate max(|phi|) ≈ Q/(4πε₀) * 2/(D_SI) [rough dipole scale]
        phi_scale = Q / (4.0 * np.pi * EPS0 * (D_SI / 2.0))
        threshold = 0.05 * phi_scale

        fine_rows = [r for r in rows if r["h_nm"] <= 10.0]
        if fine_rows:
            for r in fine_rows:
                ratio  = r["FEM_Per_vs_FFT_rms"] / phi_scale * 100.0
                passed = r["FEM_Per_vs_FFT_rms"] < threshold
                status = "PASS" if passed else "FAIL"
                print(
                    f"PASS/FAIL h={r['h_nm']:.0f}nm: "
                    f"FEM_Per_vs_FFT_rms = {r['FEM_Per_vs_FFT_rms']:.3e} V "
                    f"({ratio:.2f}% of |φ|_max ≈ {phi_scale:.3e} V) "
                    f"< 5% threshold ({threshold:.3e} V)  → {status}"
                )
        else:
            print(
                "PASS/FAIL: no h<=10nm results available "
                "(need --full or h<=10 in h_values)."
            )


if __name__ == "__main__":
    main()
