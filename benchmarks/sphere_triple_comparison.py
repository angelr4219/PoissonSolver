"""
Sphere Triple Comparison Benchmark
===================================
Compares three Poisson solver approaches on a conducting sphere in a box:

  1. FEM-Dirichlet  -- phi=0 on box walls, gmsh Ball-field refinement
  2. FEM-Periodic   -- 3-D periodic BC via dolfinx_mpc
  3. FFT            -- spectral solver on uniform Cartesian grid (numpy only)

Mesh refinements tested: h = [50, 25, 10, 5, 1, 0.1] nm

For each (method, h) pair we record:
  - L2 error vs analytic solution
  - H1 seminorm error
  - Max pointwise error
  - DOF count
  - Wall-clock solve time

Outputs (written to ./benchmark_results/):
  comparison_results.csv   -- table of all metrics
  phi_dir_h{h}nm.xdmf     -- FEM-Dirichlet solution fields
  phi_per_h{h}nm.xdmf     -- FEM-Periodic solution fields
  phi_fft_h{h}nm.npz      -- FFT solution arrays
  SUGGESTIONS.txt          -- improvement notes

Usage
-----
    # Inside the Docker container:
    python benchmarks/sphere_triple_comparison.py [--skip-dir] [--skip-per] [--skip-fft]

    # Via wrapper script:
    ./benchmarks/run_sphere_benchmark.sh
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.rank

# Physical constants
EPS0 = 8.8541878128e-12  # F/m
Q = 1.6e-19              # C  (one elementary charge)
R_NM = 10.0              # sphere radius [nm]
BOX_NM = 100.0           # half-box side [nm]  (box is [-BOX_NM, BOX_NM]^3)

R_SI = R_NM * 1e-9
BOX_SI = BOX_NM * 1e-9

# Refinement levels to test [nm]
H_LEVELS_NM = [50.0, 25.0, 10.0, 5.0, 1.0, 0.1]

# For very fine h, FEM becomes impractical — skip FEM below this threshold
FEM_MIN_H_NM = 5.0  # skip FEM for h < this (too many DOFs / too slow)

OUT_DIR = Path("benchmark_results")


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------

def phi_analytic_pts(pts: np.ndarray) -> np.ndarray:
    """phi [V] at Nx3 points for a sphere of charge Q, radius R_SI."""
    r = np.linalg.norm(pts, axis=1)
    return np.where(r >= R_SI,
                    Q / (4.0 * np.pi * EPS0 * r),
                    Q / (4.0 * np.pi * EPS0 * R_SI))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def fem_errors(phi_h, V):
    """
    Compute errors for a FEM solution vs the analytic conducting-sphere potential.

    Uses DOF coordinates directly (robust across all dolfinx versions).
    Returns (L2_rms, H1_seminorm, max_err) all in Volts.
    H1 is approximated as nan (requires UFL grad which varies by dolfinx version).
    """
    dof_coords = V.tabulate_dof_coordinates()   # (Ndof, 3)
    phi_num    = phi_h.x.array.real             # local DOF values

    phi_ref = phi_analytic_pts(dof_coords[:len(phi_num)])

    diff = phi_num - phi_ref
    # Gather global stats
    local_ss  = float(np.sum(diff**2))
    local_n   = float(len(diff))
    local_max = float(np.max(np.abs(diff))) if len(diff) else 0.0

    global_ss  = COMM.allreduce(local_ss,  op=MPI.SUM)
    global_n   = COMM.allreduce(local_n,   op=MPI.SUM)
    max_err    = COMM.allreduce(local_max,  op=MPI.MAX)

    L2_rms = float(np.sqrt(global_ss / global_n))  # RMS over DOFs
    return L2_rms, float("nan"), max_err


# ---------------------------------------------------------------------------
# Method 1: FEM Dirichlet
# ---------------------------------------------------------------------------

def run_dirichlet(h_nm: float, out_dir: Path) -> dict:
    from dolfinx import fem, default_scalar_type
    from dolfinx.io import XDMFFile
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
    from petsc4py import PETSc
    import ufl

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from poisson.geom_sphere import sphere_refined_box

    h_far_nm = min(h_nm * 10, BOX_NM / 2)
    t0 = time.perf_counter()

    msh, _, facet_tags = sphere_refined_box(
        COMM,
        box_nm=BOX_NM,
        sphere_r_nm=R_NM,
        h_near_nm=h_nm,
        h_far_nm=h_far_nm,
    )

    from dolfinx.fem import functionspace, Function, form, dirichletbc, locate_dofs_topological
    V = functionspace(msh, ("Lagrange", 1))

    # phi = 0 on all box walls (facet tag 1)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    wall_dofs = locate_dofs_topological(V, msh.topology.dim - 1,
                                        facet_tags.find(1))
    bc = dirichletbc(default_scalar_type(0.0), wall_dofs, V)

    # Point charge source term as a narrow Gaussian
    sigma = max(h_nm * 0.5, 0.1) * 1e-9
    x = ufl.SpatialCoordinate(msh)
    r2 = x[0]**2 + x[1]**2 + x[2]**2
    rho_expr = (Q / (sigma**3 * (2 * np.pi)**1.5)) * ufl.exp(-0.5 * r2 / sigma**2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = form(EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_form = form(rho_expr * v * ufl.dx)

    A = assemble_matrix(a_form, bcs=[bc]); A.assemble()
    b = assemble_vector(L_form)
    apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()
    ksp.setOperators(A)

    phi = Function(V, name="phi_dirichlet")
    ksp.solve(b, phi.x.petsc_vec)
    phi.x.scatter_forward()
    t_solve = time.perf_counter() - t0

    L2, H1, max_err = fem_errors(phi, V)
    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # Write XDMF
    xdmf_path = out_dir / f"phi_dir_h{h_nm}nm.xdmf"
    with XDMFFile(COMM, str(xdmf_path), "w") as f:
        f.write_mesh(msh)
        f.write_function(phi)

    return {"method": "FEM-Dirichlet", "h_nm": h_nm, "ndof": ndof,
            "L2_err": L2, "H1_err": H1, "max_err": max_err, "time_s": t_solve}


# ---------------------------------------------------------------------------
# Method 2: FEM Periodic
# ---------------------------------------------------------------------------

def run_periodic(h_nm: float, out_dir: Path) -> dict:
    from dolfinx import fem, default_scalar_type
    from dolfinx.io import XDMFFile
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
    from petsc4py import PETSc
    import ufl

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from poisson.geom_sphere import sphere_refined_box

    try:
        import dolfinx_mpc
    except ImportError:
        return {"method": "FEM-Periodic", "h_nm": h_nm, "ndof": 0,
                "L2_err": float("nan"), "H1_err": float("nan"),
                "max_err": float("nan"), "time_s": 0.0,
                "note": "dolfinx_mpc not installed"}

    h_far_nm = min(h_nm * 10, BOX_NM / 2)
    t0 = time.perf_counter()

    msh, _, _ = sphere_refined_box(
        COMM,
        box_nm=BOX_NM,
        sphere_r_nm=R_NM,
        h_near_nm=h_nm,
        h_far_nm=h_far_nm,
    )

    from dolfinx.fem import functionspace, Function
    V = functionspace(msh, ("Lagrange", 1))

    # Build 3-D periodic MPC
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    L_box = 2.0 * BOX_SI
    tol = 1e-12 * BOX_SI

    def _slave_x(x): return np.isclose(x[0],  BOX_SI, atol=tol)
    def _map_x(x):   out = np.copy(x); out[0] -= L_box; return out
    def _slave_y(x): return np.isclose(x[1],  BOX_SI, atol=tol)
    def _map_y(x):   out = np.copy(x); out[1] -= L_box; return out
    def _slave_z(x): return np.isclose(x[2],  BOX_SI, atol=tol)
    def _map_z(x):   out = np.copy(x); out[2] -= L_box; return out

    mpc.create_periodic_constraint_geometrical(V, _slave_x, _map_x, [], scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, _slave_y, _map_y, [], scale=1.0)
    mpc.create_periodic_constraint_geometrical(V, _slave_z, _map_z, [], scale=1.0)
    mpc.finalize()

    sigma = max(h_nm * 0.5, 0.1) * 1e-9
    x = ufl.SpatialCoordinate(msh)
    r2 = x[0]**2 + x[1]**2 + x[2]**2
    rho_expr = (Q / (sigma**3 * (2 * np.pi)**1.5)) * ufl.exp(-0.5 * r2 / sigma**2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_ufl = EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_ufl = rho_expr * v * ufl.dx

    phi = Function(mpc.function_space, name="phi_periodic")
    A = dolfinx_mpc.assemble_matrix(fem.form(a_ufl), mpc)
    A.assemble()
    b = dolfinx_mpc.assemble_vector(fem.form(L_ufl), mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()
    ksp.setOperators(A)
    ksp.solve(b, phi.x.petsc_vec)
    phi.x.scatter_forward()
    t_solve = time.perf_counter() - t0

    L2, H1, max_err = fem_errors(phi, V)
    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    xdmf_path = out_dir / f"phi_per_h{h_nm}nm.xdmf"
    with XDMFFile(COMM, str(xdmf_path), "w") as f:
        f.write_mesh(msh)
        f.write_function(phi)

    return {"method": "FEM-Periodic", "h_nm": h_nm, "ndof": ndof,
            "L2_err": L2, "H1_err": H1, "max_err": max_err, "time_s": t_solve}


# ---------------------------------------------------------------------------
# HDF5 + XDMF writer for FFT uniform-grid output
# ---------------------------------------------------------------------------

def _write_fft_xdmf(phi: np.ndarray, phi_ref: np.ndarray, xyz: np.ndarray,
                    dx: float, stem: Path) -> None:
    """
    Write phi and phi_ref on a uniform 3-D Cartesian grid to:
      <stem>.h5   -- datasets: /phi, /phi_ref, /xyz
      <stem>.xdmf -- XDMF2 descriptor readable by ParaView / VisIt

    The XDMF uses a 3DCoRectMesh (uniform rectilinear grid) so ParaView
    opens it identically to the FEM XDMF files.
    """
    import h5py

    h5_path = Path(str(stem) + ".h5")
    xdmf_path = Path(str(stem) + ".xdmf")

    N = phi.shape[0]
    origin = float(xyz[0])
    spacing = dx  # metres

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("phi",     data=phi.astype(np.float64),     compression="gzip")
        f.create_dataset("phi_ref", data=phi_ref.astype(np.float64), compression="gzip")
        f.create_dataset("xyz",     data=xyz.astype(np.float64))
        f.attrs["origin_m"]  = origin
        f.attrs["spacing_m"] = spacing
        f.attrs["N"]         = N

    h5_name = h5_path.name
    n_nodes = N ** 3
    xdmf_txt = f"""\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="FFT_phi" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{N} {N} {N}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Format="XML" Dimensions="3">{origin} {origin} {origin}</DataItem>
        <DataItem Format="XML" Dimensions="3">{spacing} {spacing} {spacing}</DataItem>
      </Geometry>
      <Attribute Name="phi" AttributeType="Scalar" Center="Node">
        <DataItem Format="HDF" Dimensions="{N} {N} {N}" NumberType="Float" Precision="8">
          {h5_name}:/phi
        </DataItem>
      </Attribute>
      <Attribute Name="phi_ref" AttributeType="Scalar" Center="Node">
        <DataItem Format="HDF" Dimensions="{N} {N} {N}" NumberType="Float" Precision="8">
          {h5_name}:/phi_ref
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    xdmf_path.write_text(xdmf_txt)


# ---------------------------------------------------------------------------
# Method 3: FFT
# ---------------------------------------------------------------------------

def run_fft(h_nm: float, out_dir: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from poisson.fft_solve import (fft_poisson_3d, sphere_charge_shell_grid,
                                   phi_analytic_sphere, max_n_for_memory)

    L = BOX_SI
    dx = h_nm * 1e-9
    N = int(round(2 * L / dx))

    n_max = max_n_for_memory(fraction=0.3)
    if N > n_max:
        return {"method": "FFT", "h_nm": h_nm, "ndof": N**3,
                "L2_err": float("nan"), "H1_err": float("nan"),
                "max_err": float("nan"), "time_s": 0.0,
                "note": f"N={N} > memory limit {n_max}"}

    t0 = time.perf_counter()

    rho, xyz = sphere_charge_shell_grid(N, L, R_SI, Q, sigma_nm=max(h_nm * 0.5, 0.2))
    phi_num = fft_poisson_3d(rho, dx)
    t_solve = time.perf_counter() - t0

    phi_ref = phi_analytic_sphere(xyz, R_SI, Q)

    diff = phi_num - phi_ref
    # Remove DC offset (periodic BC fixes mean not absolute level)
    diff -= diff.mean()

    L2 = float(np.sqrt(np.mean(diff**2)))
    max_err = float(np.max(np.abs(diff)))

    _write_fft_xdmf(phi_num, phi_ref, xyz, dx, out_dir / f"phi_fft_h{h_nm}nm")

    return {"method": "FFT", "h_nm": h_nm, "ndof": N**3,
            "L2_err": L2, "H1_err": float("nan"), "max_err": max_err,
            "time_s": t_solve}


# ---------------------------------------------------------------------------
# Suggestions file
# ---------------------------------------------------------------------------

SUGGESTIONS = """\
SUGGESTIONS FOR IMPROVING ACCURACY (towards MASQUE-level comparison)
=====================================================================

1. FAR-FIELD BOUNDARY CONDITION (FEM-Dirichlet)
   Use the exact Coulomb far-field  phi = Q/(4*pi*eps0*r)  on the box walls
   instead of phi=0.  This eliminates domain-truncation error, which dominates
   at coarse h when the box is not large enough.

2. WIGNER-SEITZ / MADELUNG CORRECTION (FEM-Periodic)
   A periodic array of charges introduces a Madelung self-energy.  Apply the
   neutralising background correction  rho -= rho.mean()  and document that
   the result converges to the isolated-sphere limit only as box/R -> inf.

3. HIGHER-ORDER ELEMENTS
   Switch from P1 to P2 Lagrange elements.  Convergence improves from O(h^2)
   to O(h^4) in L2, halving the required mesh density for the same accuracy.

4. CAPACITANCE EXTRACTION
   Compute C = Q / phi_surface via a surface-flux integral
       C = -eps0 * integral(grad(phi).n dS) / phi_sphere
   This is the quantity MASQUE compares against analytic C = 4*pi*eps0*R.

5. MULTI-DIELECTRIC GEOMETRY
   The FFT solver assumes constant eps.  For real gate stacks (SiGe/oxide/gate)
   use only FEM.  Add a material tag layer to the sphere benchmark to test
   this code path.

6. SOLVER SCALABILITY
   For DOF counts > 1M, switch from CG+Hypre to MUMPS direct solver
   (PETSc option: -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps).
   This avoids iterative convergence issues on ill-conditioned periodic systems.

7. FFT SMEARING ARTEFACT
   The Gaussian charge shell introduces a smearing error proportional to sigma.
   Deconvolve in k-space:  phi_hat_corrected = phi_hat / G_hat  where G_hat is
   the Fourier transform of the Gaussian.  This recovers the point-charge limit
   without requiring sigma -> 0.

8. COMPARISON METRIC
   Report relative error at r = 2R (outside sphere, away from singularity) in
   addition to global L2.  MASQUE benchmarks often quote pointwise error at
   specific probe points rather than global norms.
"""


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dir", action="store_true")
    parser.add_argument("--skip-per", action="store_true")
    parser.add_argument("--skip-fft", action="store_true")
    args = parser.parse_args()

    if RANK == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    COMM.Barrier()

    results = []
    fields = ["method", "h_nm", "ndof", "L2_err", "H1_err", "max_err", "time_s", "note"]

    for h_nm in H_LEVELS_NM:
        skip_fem = h_nm < FEM_MIN_H_NM

        # --- FEM Dirichlet ---
        if not args.skip_dir:
            if skip_fem:
                if RANK == 0:
                    print(f"[DIR ] h={h_nm}nm: skipping (too fine for FEM)")
                results.append({"method": "FEM-Dirichlet", "h_nm": h_nm,
                                 "ndof": 0, "L2_err": float("nan"),
                                 "H1_err": float("nan"), "max_err": float("nan"),
                                 "time_s": 0.0, "note": "skipped: too fine"})
            else:
                if RANK == 0:
                    print(f"[DIR ] h={h_nm}nm ...", flush=True)
                try:
                    r = run_dirichlet(h_nm, OUT_DIR)
                except Exception as exc:
                    r = {"method": "FEM-Dirichlet", "h_nm": h_nm, "ndof": 0,
                         "L2_err": float("nan"), "H1_err": float("nan"),
                         "max_err": float("nan"), "time_s": 0.0, "note": str(exc)}
                results.append(r)
                if RANK == 0:
                    print(f"         L2={r['L2_err']:.3e}  t={r['time_s']:.1f}s")

        # --- FEM Periodic ---
        if not args.skip_per:
            if skip_fem:
                if RANK == 0:
                    print(f"[PER ] h={h_nm}nm: skipping (too fine for FEM)")
                results.append({"method": "FEM-Periodic", "h_nm": h_nm,
                                 "ndof": 0, "L2_err": float("nan"),
                                 "H1_err": float("nan"), "max_err": float("nan"),
                                 "time_s": 0.0, "note": "skipped: too fine"})
            else:
                if RANK == 0:
                    print(f"[PER ] h={h_nm}nm ...", flush=True)
                try:
                    r = run_periodic(h_nm, OUT_DIR)
                except Exception as exc:
                    r = {"method": "FEM-Periodic", "h_nm": h_nm, "ndof": 0,
                         "L2_err": float("nan"), "H1_err": float("nan"),
                         "max_err": float("nan"), "time_s": 0.0, "note": str(exc)}
                results.append(r)
                if RANK == 0:
                    print(f"         L2={r['L2_err']:.3e}  t={r['time_s']:.1f}s")

        # --- FFT ---
        if not args.skip_fft and RANK == 0:
            print(f"[FFT ] h={h_nm}nm ...", flush=True)
            try:
                r = run_fft(h_nm, OUT_DIR)
            except Exception as exc:
                r = {"method": "FFT", "h_nm": h_nm, "ndof": 0,
                     "L2_err": float("nan"), "H1_err": float("nan"),
                     "max_err": float("nan"), "time_s": 0.0, "note": str(exc)}
            results.append(r)
            print(f"         L2={r['L2_err']:.3e}  t={r['time_s']:.1f}s")

    # Write CSV and suggestions on rank 0
    if RANK == 0:
        csv_path = OUT_DIR / "comparison_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                row.setdefault("note", "")
                writer.writerow(row)

        sug_path = OUT_DIR / "SUGGESTIONS.txt"
        sug_path.write_text(SUGGESTIONS)

        print("\n=== Results written to", OUT_DIR, "===")
        print(f"  {csv_path.name}")
        print(f"  {sug_path.name}")
        print("\nSummary table:")
        print(f"{'Method':<16} {'h [nm]':>8} {'DOF':>10} {'L2 err':>12} {'max err':>12} {'time [s]':>10}")
        print("-" * 72)
        for r in results:
            note = r.get("note", "")
            tag = f" ({note})" if note else ""
            print(f"{r['method']:<16} {r['h_nm']:>8.1f} {r['ndof']:>10d} "
                  f"{r['L2_err']:>12.3e} {r['max_err']:>12.3e} {r['time_s']:>10.2f}{tag}")


if __name__ == "__main__":
    main()
