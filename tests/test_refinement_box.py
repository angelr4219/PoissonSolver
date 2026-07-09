"""
Test 4: Local refinement box vs uniform mesh
=============================================
Solves a point-charge Poisson problem where the charge is localised near
the origin. Compares:

  A) Uniform mesh at coarse h
  B) Uniform mesh at fine h (DOF-matched to C)
  C) Coarse mesh + local refinement box around the charge (same DOFs as B)

Shows that a refinement box gives near-fine-mesh accuracy near the charge
at the same DOF cost as a fine uniform mesh, but much cheaper away from it.

Analytic reference: Coulomb potential, φ = Q/(4πε₀r).

Run:
    ./run_dolfinx.sh tests/test_refinement_box.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
import ufl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from poisson.refinement import RefinementBox, build_refined_mesh_3d

COMM = MPI.COMM_WORLD
EPS0 = 8.8541878128e-12
Q    = 1.6e-19
BOX  = 100e-9    # half-box side [m]
# Probe region around the charge — error measured here
PROBE_R = 20e-9  # m


def phi_analytic(pts: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(pts, axis=1)
    r = np.where(r == 0, 1e-30, r)
    return Q / (4 * np.pi * EPS0 * r)


def _solve_on_mesh(msh, h_source_nm: float, label: str):
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Zero BC on all walls
    def walls(x):
        return (
            np.isclose(np.abs(x[0]), BOX, atol=1e-14) |
            np.isclose(np.abs(x[1]), BOX, atol=1e-14) |
            np.isclose(np.abs(x[2]), BOX, atol=1e-14)
        )
    wall_dofs = fem.locate_dofs_geometrical(V, walls)
    bc = fem.dirichletbc(default_scalar_type(0.0), wall_dofs, V)

    sigma = max(h_source_nm * 0.5, 0.5) * 1e-9
    x = ufl.SpatialCoordinate(msh)
    r2 = x[0]**2 + x[1]**2 + x[2]**2
    rho = (Q / (sigma**3 * (2*np.pi)**1.5)) * ufl.exp(-0.5 * r2 / sigma**2)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(rho * v * ufl.dx)
    A = assemble_matrix(a, bcs=[bc]); A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]]); b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg"); ksp.getPC().setType("hypre"); ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions(); ksp.setOperators(A)
    phi = fem.Function(V, name=f"phi_{label}")
    ksp.solve(b, phi.x.petsc_vec); phi.x.scatter_forward()

    # Error within probe sphere only (where field is non-trivial)
    coords = V.tabulate_dof_coordinates()
    num    = phi.x.array.real[:len(coords)]
    r      = np.linalg.norm(coords, axis=1)
    mask   = (r > 2e-9) & (r < PROBE_R)  # exclude r≈0 singularity
    if mask.sum() == 0:
        return float("nan"), V.dofmap.index_map.size_global

    diff      = num[mask] - phi_analytic(coords[mask])
    local_ss  = float(np.sum(diff**2))
    local_n   = float(mask.sum())
    global_ss = COMM.allreduce(local_ss, op=MPI.SUM)
    global_n  = COMM.allreduce(local_n,  op=MPI.SUM)
    rms = float(np.sqrt(global_ss / global_n)) if global_n > 0 else float("nan")
    return rms, V.dofmap.index_map.size_global


if __name__ == "__main__":
    # --- A: coarse uniform mesh (h=20nm) ------------------------------------
    if COMM.rank == 0: print("Building mesh A: uniform h=20nm ...", flush=True)
    t0 = time.perf_counter()
    msh_A = build_refined_mesh_3d(COMM, BOX, BOX, BOX*2, h_coarse=20e-9, boxes=[])
    err_A, ndof_A = _solve_on_mesh(msh_A, h_source_nm=20.0, label="A")
    t_A = time.perf_counter() - t0

    # --- B: fine uniform mesh (h=5nm) ---------------------------------------
    if COMM.rank == 0: print("Building mesh B: uniform h=5nm ...", flush=True)
    t0 = time.perf_counter()
    msh_B = build_refined_mesh_3d(COMM, BOX, BOX, BOX*2, h_coarse=5e-9, boxes=[])
    err_B, ndof_B = _solve_on_mesh(msh_B, h_source_nm=5.0, label="B")
    t_B = time.perf_counter() - t0

    # --- C: coarse mesh + refinement box around origin ----------------------
    if COMM.rank == 0: print("Building mesh C: coarse + refinement box ...", flush=True)
    t0 = time.perf_counter()
    refbox = RefinementBox(cx=0.0, cy=0.0, cz=BOX,  # cz=BOX because domain z in [0, 2*BOX]
                           lx=40e-9, ly=40e-9, lz=40e-9,
                           h_fine=5e-9)
    msh_C = build_refined_mesh_3d(COMM, BOX, BOX, BOX*2, h_coarse=20e-9, boxes=[refbox])
    err_C, ndof_C = _solve_on_mesh(msh_C, h_source_nm=5.0, label="C")
    t_C = time.perf_counter() - t0

    if COMM.rank == 0:
        print(f"\n=== Local Refinement Box Comparison ===")
        print(f"Domain: cube ±{BOX*1e9:.0f}nm, point charge at origin")
        print(f"Error: RMS vs analytic Coulomb within r < {PROBE_R*1e9:.0f}nm probe sphere")
        print()
        print(f"{'Mesh':>26} {'DOFs':>8} {'RMS err [V]':>12} {'Time':>8}")
        print("-" * 60)
        print(f"{'A: uniform h=20nm (coarse)':>26} {ndof_A:>8} {err_A:>12.3e} {t_A:>7.1f}s")
        print(f"{'B: uniform h=5nm  (fine)':>26} {ndof_B:>8} {err_B:>12.3e} {t_B:>7.1f}s")
        print(f"{'C: h=20nm + refbox h=5nm':>26} {ndof_C:>8} {err_C:>12.3e} {t_C:>7.1f}s")
        print()
        print("Goal: C should match B accuracy near the charge, at fewer DOFs than B.")
        print(f"Probe region: r in [2nm, {PROBE_R*1e9:.0f}nm]")
