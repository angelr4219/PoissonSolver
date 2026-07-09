"""
Test 3: BC comparison — zero-wall vs exact-Coulomb vs periodic
===============================================================
Sphere of charge Q, radius R=10nm, in a cubic box of half-side 100nm.
Analytic reference: free-space Coulomb potential.

Compares three boundary treatments at each mesh size h:

  A) FEM-ZeroWall   — φ=0 on box walls  (current default, has domain-truncation error)
  B) FEM-CoulombBC  — φ=Q/(4πε₀r) on box walls  (exact far-field)
  C) FEM-Periodic   — periodic BC via dolfinx_mpc  (Madelung artifact)

Expected:
  A) error stays flat ~1e-2 regardless of h  (BC dominates)
  B) error drops with h, eventually limited by mesh   (proper convergence)
  C) error may worsen at fine h due to Madelung artifact

Run:
    ./run_dolfinx.sh tests/test_bc_comparison.py
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
from poisson.geom_sphere import sphere_refined_box

COMM = MPI.COMM_WORLD
EPS0 = 8.8541878128e-12
Q    = 1.6e-19
R_NM = 10.0
BOX_NM = 100.0
R_SI  = R_NM  * 1e-9
BOX_SI = BOX_NM * 1e-9

H_LEVELS = [50.0, 25.0, 10.0, 5.0]   # nm; add finer if you have time


def phi_analytic(pts: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(pts, axis=1)
    return np.where(r >= R_SI,
                    Q / (4 * np.pi * EPS0 * np.where(r > 0, r, 1)),
                    Q / (4 * np.pi * EPS0 * R_SI))


def _rms_error(phi_h, V):
    coords = V.tabulate_dof_coordinates()
    num = phi_h.x.array.real[:len(coords)]
    ref = phi_analytic(coords)
    diff = num - ref
    ss  = COMM.allreduce(float(np.sum(diff**2)), op=MPI.SUM)
    n   = COMM.allreduce(float(len(diff)),       op=MPI.SUM)
    return float(np.sqrt(ss / n))


def _ksp_solve(A, b, msh):
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg"); ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions(); ksp.setOperators(A)
    x = A.createVecRight(); ksp.solve(b, x)
    return x


def _source(msh, h_nm):
    sigma = max(h_nm * 0.5, 0.1) * 1e-9
    x = ufl.SpatialCoordinate(msh)
    r2 = x[0]**2 + x[1]**2 + x[2]**2
    return (Q / (sigma**3 * (2*np.pi)**1.5)) * ufl.exp(-0.5 * r2 / sigma**2)


# ---- A: zero-wall BC -------------------------------------------------------

def run_zero_wall(h_nm):
    msh, _, ftags = sphere_refined_box(COMM, BOX_NM, R_NM,
                                       h_near_nm=h_nm,
                                       h_far_nm=min(h_nm*10, BOX_NM/2))
    V = fem.functionspace(msh, ("Lagrange", 1))
    wall_dofs = fem.locate_dofs_topological(V, msh.topology.dim-1, ftags.find(1))
    bc = fem.dirichletbc(default_scalar_type(0.0), wall_dofs, V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(_source(msh, h_nm) * v * ufl.dx)
    A = assemble_matrix(a, bcs=[bc]); A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    phi = fem.Function(V, name="phi_zero_wall")
    phi.x.petsc_vec.array[:] = _ksp_solve(A, b, msh).array
    phi.x.scatter_forward()
    return _rms_error(phi, V), V.dofmap.index_map.size_global


# ---- B: exact Coulomb BC ---------------------------------------------------

def run_coulomb_bc(h_nm):
    msh, _, ftags = sphere_refined_box(COMM, BOX_NM, R_NM,
                                       h_near_nm=h_nm,
                                       h_far_nm=min(h_nm*10, BOX_NM/2))
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Interpolate exact Coulomb on all wall DOFs
    phi_exact_fn = fem.Function(V)
    def _coulomb_interp(x):
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        out = np.where(r >= R_SI,
                       Q / (4*np.pi*EPS0 * np.where(r > 0, r, 1.0)),
                       Q / (4*np.pi*EPS0 * R_SI))
        return out
    phi_exact_fn.interpolate(_coulomb_interp)

    msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    wall_dofs = fem.locate_dofs_topological(V, msh.topology.dim-1, ftags.find(1))
    bc = fem.dirichletbc(phi_exact_fn, wall_dofs)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(_source(msh, h_nm) * v * ufl.dx)
    A = assemble_matrix(a, bcs=[bc]); A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    phi = fem.Function(V, name="phi_coulomb_bc")
    phi.x.petsc_vec.array[:] = _ksp_solve(A, b, msh).array
    phi.x.scatter_forward()
    return _rms_error(phi, V), V.dofmap.index_map.size_global


# ---- C: periodic BC --------------------------------------------------------

def run_periodic(h_nm):
    try:
        import dolfinx_mpc
    except ImportError:
        return float("nan"), 0

    msh, _, _ = sphere_refined_box(COMM, BOX_NM, R_NM,
                                   h_near_nm=h_nm,
                                   h_far_nm=min(h_nm*10, BOX_NM/2))
    V = fem.functionspace(msh, ("Lagrange", 1))

    L_box = 2.0 * BOX_SI
    tol = 1e-12 * BOX_SI
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    for axis, shift in enumerate([L_box, L_box, L_box]):
        def slave(x, a=axis): return np.isclose(x[a], BOX_SI, atol=tol)
        def mapping(x, a=axis, s=shift): out = np.copy(x); out[a] -= s; return out
        mpc.create_periodic_constraint_geometrical(V, slave, mapping, [], scale=1.0)
    mpc.finalize()

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_ufl = EPS0 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_ufl = _source(msh, h_nm) * v * ufl.dx

    phi = fem.Function(mpc.function_space, name="phi_periodic")
    A = dolfinx_mpc.assemble_matrix(fem.form(a_ufl), mpc); A.assemble()
    b = dolfinx_mpc.assemble_vector(fem.form(L_ufl), mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg"); ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10)
    ksp.setOperators(A); ksp.solve(b, phi.x.petsc_vec)
    phi.x.scatter_forward()
    return _rms_error(phi, V), V.dofmap.index_map.size_global


# ---- main ------------------------------------------------------------------

if __name__ == "__main__":
    rows = []
    for h in H_LEVELS:
        if COMM.rank == 0:
            print(f"h={h}nm ...", flush=True)

        t0 = time.perf_counter()
        err_a, ndof_a = run_zero_wall(h)
        t_a = time.perf_counter() - t0

        t0 = time.perf_counter()
        err_b, ndof_b = run_coulomb_bc(h)
        t_b = time.perf_counter() - t0

        t0 = time.perf_counter()
        err_c, ndof_c = run_periodic(h)
        t_c = time.perf_counter() - t0

        rows.append((h, ndof_a, err_a, t_a, err_b, t_b, err_c, t_c))

    if COMM.rank == 0:
        print("\n=== BC Comparison: sphere Q=1e, R=10nm, box=100nm ===")
        print("Metric: RMS error vs free-space Coulomb analytic [V]")
        print()
        print(f"{'h(nm)':>6} {'DOFs':>7} | {'ZeroWall':>10} {'t':>6} | {'CoulombBC':>10} {'t':>6} | {'Periodic':>10} {'t':>6}")
        print("-" * 72)
        for h, ndof, ea, ta, eb, tb, ec, tc in rows:
            print(f"{h:>6.1f} {ndof:>7} | {ea:>10.3e} {ta:>5.1f}s | {eb:>10.3e} {tb:>5.1f}s | {ec:>10.3e} {tc:>5.1f}s")
        print()
        print("ZeroWall:  φ=0 on box walls  — domain-truncation error, should NOT converge")
        print("CoulombBC: φ=Q/4πε₀r on walls — correct far-field, SHOULD converge with h")
        print("Periodic:  dolfinx_mpc periodic — Madelung artifact, may worsen at fine h")
