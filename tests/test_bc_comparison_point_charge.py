"""
test_bc_comparison_point_charge.py
====================================
Diagnostic test: zero-wall Dirichlet BC (ZeroWall) causes domain-truncation
error that does NOT converge with h.  Exact Coulomb Dirichlet BC (CoulombBC)
on the walls DOES converge.  LargeDomain (3× box, zero-wall BC) shows the
domain-size effect.

Physics
-------
Gaussian point charge at origin (Q = 1e, σ = max(h*0.5, 0.1) nm) in a cubic
box of half-side BOX_NM = 100 nm.  Reference: free-space Coulomb potential
φ_ref = Q / (4πε₀ · max(r, R_SI)).
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

import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

from poisson.geom_sphere import sphere_refined_box

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EPS0   = 8.8541878128e-12   # F/m
Q      = 1.6e-19             # C (one elementary charge)
R_NM   = 10.0                # Gaussian-cap radius [nm]
BOX_NM = 100.0               # half-side of reference cubic box [nm]
R_SI   = R_NM * 1e-9         # [m]
BOX_SI = BOX_NM * 1e-9       # [m]


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

class ErrorMetrics(NamedTuple):
    rms: float         # RMS error vs analytic over all DOFs [V]
    max_abs: float     # max |error| over all DOFs [V]
    probe_near: float  # |error| at DOF nearest r ≈ 2·R_SI [V]
    probe_far: float   # |error| at DOF nearest r ≈ 50 nm [V]


def compute_errors(phi: fem.Function) -> ErrorMetrics:
    """Compare phi against the free-space Coulomb reference at every DOF."""
    coords = phi.function_space.tabulate_dof_coordinates()   # (N, 3)
    vals   = phi.x.array                                      # (N,)

    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)
    r_safe   = np.maximum(r, R_SI)
    phi_ref  = Q / (4.0 * np.pi * EPS0 * r_safe)
    err      = vals - phi_ref

    rms     = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))

    # Near-field probe: DOF closest to r = 2·R_SI = 20 nm
    target_near = 2.0 * R_SI
    idx_near    = int(np.argmin(np.abs(r - target_near)))
    probe_near  = float(np.abs(err[idx_near]))

    # Far-field probe: DOF closest to r = 50 nm
    target_far = 50.0e-9
    idx_far    = int(np.argmin(np.abs(r - target_far)))
    probe_far  = float(np.abs(err[idx_far]))

    return ErrorMetrics(rms, max_abs, probe_near, probe_far)


# ---------------------------------------------------------------------------
# Per-case solver
# ---------------------------------------------------------------------------

def solve_variant(
    variant: str,
    h_nm: float,
    comm: MPI.Comm,
    write_xdmf: bool,
    outdir: Path,
    max_dofs: int,
) -> tuple[int, ErrorMetrics, float]:
    """
    Build mesh, apply BC, solve, compute errors.

    Parameters
    ----------
    variant : "ZeroWall" | "CoulombBC" | "LargeDomain"
    h_nm    : target near-field element size [nm]

    Returns
    -------
    (n_dofs, metrics, elapsed_seconds)
    """
    sigma_nm = max(h_nm * 0.5, 0.1)
    sigma_si = sigma_nm * 1e-9

    # Box size and far-field mesh size
    if variant == "LargeDomain":
        box_nm    = 3.0 * BOX_NM                        # 300 nm half-side
        h_far_nm  = min(h_nm * 10.0, 150.0)
    else:
        box_nm    = BOX_NM
        h_far_nm  = min(h_nm * 10.0, BOX_NM / 2.0)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------ mesh
    msh, _, facet_tags = sphere_refined_box(
        comm,
        box_nm=box_nm,
        sphere_r_nm=R_NM,
        h_near_nm=h_nm,
        h_far_nm=h_far_nm,
    )

    V = fem.functionspace(msh, ("Lagrange", 1))

    n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if comm.rank == 0:
        print(f"  [{variant} h={h_nm}nm]  dofs = {n_dofs:,}", flush=True)
    if n_dofs > max_dofs:
        raise RuntimeError(
            f"DOF count {n_dofs} exceeds --max-dofs {max_dofs}; skipping."
        )

    # ------------------------------------------------------------ Dirichlet BC
    wall_facets = facet_tags.find(1)
    wall_dofs   = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, wall_facets
    )

    if variant in ("ZeroWall", "LargeDomain"):
        bc  = fem.dirichletbc(0.0, wall_dofs, V)
        bcs = [bc]
    else:  # CoulombBC
        phi_bc_fn = fem.Function(V)

        def _phi_coulomb(x):
            r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            r = np.maximum(r, R_SI)
            return Q / (4.0 * np.pi * EPS0 * r)

        phi_bc_fn.interpolate(_phi_coulomb)
        bc  = fem.dirichletbc(phi_bc_fn, wall_dofs)
        bcs = [bc]

    # ---------------------------------------------------------- Gaussian source
    f = fem.Function(V)

    def _rho_gaussian(x):
        r2   = x[0]**2 + x[1]**2 + x[2]**2
        norm = Q / (sigma_si**3 * (2.0 * np.pi) ** 1.5)
        return norm * np.exp(-0.5 * r2 / sigma_si**2)

    f.interpolate(_rho_gaussian)

    # ---------------------------------------------------------- Weak form
    u  = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=msh)

    # -∇·(ε₀∇φ) = ρ  →  ε₀ ∫ ∇φ·∇v dx = ∫ ρ v dx
    a = EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = f * v * dx

    # ------------------------------------------------------------------ solve
    _prefix = f"bcpt_{variant.lower()}_h{h_nm:.0f}_"
    problem = LinearProblem(
        a, L,
        petsc_options_prefix=_prefix,
        bcs=bcs,
        petsc_options={
            "ksp_type":    "cg",
            "pc_type":     "hypre",
            "ksp_rtol":    1e-10,
            "ksp_max_it":  10000,
        },
    )
    phi       = problem.solve()
    phi.name  = f"phi_{variant}_h{h_nm:.0f}nm"
    elapsed   = time.perf_counter() - t0

    # ------------------------------------------------------------------ error
    metrics = compute_errors(phi)

    # ------------------------------------------------------------------ XDMF
    if write_xdmf:
        outdir.mkdir(parents=True, exist_ok=True)
        fname = outdir / f"bc_comp_{variant}_h{h_nm:.0f}nm.xdmf"
        with dolfinx.io.XDMFFile(comm, str(fname), "w") as xf:
            xf.write_mesh(msh)
            xf.write_function(phi)
        if comm.rank == 0:
            print(f"  Written: {fname}", flush=True)

    return n_dofs, metrics, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "BC comparison: ZeroWall vs CoulombBC vs LargeDomain\n"
            "Proves that exact Coulomb wall BC yields h-convergence while\n"
            "zero-wall BC is contaminated by domain-truncation error."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true", default=True,
        help="h = [50, 25, 10] nm  (default)",
    )
    mode.add_argument(
        "--full", action="store_true",
        help="h = [50, 25, 10, 5] nm",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("tests/test_bc_comparison_point_charge/output"),
    )
    parser.add_argument("--write-xdmf", action="store_true")
    parser.add_argument("--save-csv",   action="store_true")
    parser.add_argument("--max-dofs",   type=int, default=2_000_000)
    args = parser.parse_args()

    h_values = [50.0, 25.0, 10.0, 5.0] if args.full else [50.0, 25.0, 10.0]
    comm     = MPI.COMM_WORLD
    variants = ["ZeroWall", "CoulombBC", "LargeDomain"]

    # -------------------------------------------------------------- header
    col_w  = [12, 6, 9, 12, 12, 14, 13, 8]
    header = (
        f"{'variant':>{col_w[0]}} | {'h_nm':>{col_w[1]}} | "
        f"{'dofs':>{col_w[2]}} | {'rms_err_V':>{col_w[3]}} | "
        f"{'max_err_V':>{col_w[4]}} | {'probe_near_V':>{col_w[5]}} | "
        f"{'probe_far_V':>{col_w[6]}} | {'time_s':>{col_w[7]}}"
    )
    sep = "-" * len(header)

    if comm.rank == 0:
        print()
        print("=" * len(header))
        print("BC Comparison: ZeroWall vs CoulombBC vs LargeDomain")
        print("=" * len(header))
        print(header)
        print(sep)

    rows: list[dict] = []

    for variant in variants:
        for h_nm in h_values:
            if comm.rank == 0:
                print(f"\nRunning {variant}, h = {h_nm} nm ...", flush=True)

            try:
                n_dofs, metrics, elapsed = solve_variant(
                    variant, h_nm, comm,
                    write_xdmf=args.write_xdmf,
                    outdir=args.outdir,
                    max_dofs=args.max_dofs,
                )
            except RuntimeError as exc:
                if comm.rank == 0:
                    print(f"  SKIPPED: {exc}", flush=True)
                continue

            if comm.rank == 0:
                row_str = (
                    f"{variant:>{col_w[0]}} | {h_nm:>{col_w[1]}.1f} | "
                    f"{n_dofs:>{col_w[2]}d} | "
                    f"{metrics.rms:>{col_w[3]}.4e} | "
                    f"{metrics.max_abs:>{col_w[4]}.4e} | "
                    f"{metrics.probe_near:>{col_w[5]}.4e} | "
                    f"{metrics.probe_far:>{col_w[6]}.4e} | "
                    f"{elapsed:>{col_w[7]}.2f}"
                )
                print(row_str, flush=True)
                rows.append({
                    "variant":      variant,
                    "h_nm":         h_nm,
                    "dofs":         n_dofs,
                    "rms_err_V":    metrics.rms,
                    "max_err_V":    metrics.max_abs,
                    "probe_near_V": metrics.probe_near,
                    "probe_far_V":  metrics.probe_far,
                    "time_s":       elapsed,
                })

    if comm.rank == 0:
        print(sep)

    # ---------------------------------------------------------------- CSV
    if args.save_csv and comm.rank == 0 and rows:
        args.outdir.mkdir(parents=True, exist_ok=True)
        csv_path = args.outdir / "bc_comparison_results.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved: {csv_path}", flush=True)

    # --------------------------------------------------------- expected summary
    if comm.rank == 0:
        print()
        print("Expected behaviour")
        print("------------------")
        print(
            "ZeroWall:    RMS error should stay ~flat across h "
            "(domain truncation dominates)"
        )
        print(
            "CoulombBC:   RMS error should DECREASE with h "
            "(proper convergence)"
        )
        print(
            "LargeDomain: RMS error should be lower than ZeroWall at same h"
        )
        print()

        # PASS/FAIL
        # 1. ZeroWall at finest h stays in domain-truncation regime (~1e-2 V)
        # 2. CoulombBC at finest h < ZeroWall at finest h / 5  (physics-meaningful threshold)
        # 3. CoulombBC at finest h < CoulombBC at coarsest h   (actually converging)
        coulomb_rows = sorted(
            [r for r in rows if r["variant"] == "CoulombBC"],
            key=lambda r: r["h_nm"],
        )
        zerowall_rows = sorted(
            [r for r in rows if r["variant"] == "ZeroWall"],
            key=lambda r: r["h_nm"],
        )

        failures: list[str] = []

        if coulomb_rows and zerowall_rows:
            finest_cbc    = coulomb_rows[0]    # smallest h_nm = finest mesh
            coarsest_cbc  = coulomb_rows[-1]   # largest h_nm = coarsest mesh
            finest_zw     = zerowall_rows[0]   # smallest h_nm = finest mesh

            # Check 1: ZeroWall at finest h is in [1e-3, 1e-1] V (domain-truncation floor)
            zw_rms = finest_zw["rms_err_V"]
            if not (1e-3 <= zw_rms <= 1e-1):
                failures.append(
                    f"ZeroWall h={finest_zw['h_nm']:.0f}nm RMS={zw_rms:.3e} V "
                    f"not in expected range [1e-3, 1e-1] V"
                )

            # Check 2: CoulombBC finest < ZeroWall finest / 5
            cbc_finest_rms = finest_cbc["rms_err_V"]
            threshold = zw_rms / 5.0
            if cbc_finest_rms >= threshold:
                failures.append(
                    f"CoulombBC h={finest_cbc['h_nm']:.0f}nm RMS={cbc_finest_rms:.3e} V "
                    f">= ZeroWall/5 = {threshold:.3e} V"
                )

            # Check 3: CoulombBC finest < CoulombBC coarsest (converging)
            if len(coulomb_rows) >= 2 and cbc_finest_rms >= coarsest_cbc["rms_err_V"]:
                failures.append(
                    f"CoulombBC h={finest_cbc['h_nm']:.0f}nm RMS ({cbc_finest_rms:.3e} V) "
                    f">= h={coarsest_cbc['h_nm']:.0f}nm RMS ({coarsest_cbc['rms_err_V']:.3e} V) "
                    f"-- not converging"
                )
        else:
            failures.append("insufficient data (need CoulombBC and ZeroWall rows)")

        if failures:
            print("FAIL")
            for msg in failures:
                print(f"  {msg}")
            sys.exit(1)
        else:
            print("PASS")


if __name__ == "__main__":
    main()
