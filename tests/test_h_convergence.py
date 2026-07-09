"""
Test 1: h-refinement convergence (P1 FEM)
==========================================
Solves -Δu = f on [0,1]^2 with exact Dirichlet BCs.
Exact solution: u = sin(πx)sin(πy), f = 2π²sin(πx)sin(πy).

Expected: L2 error halves each time n doubles (rate ≈ 2.0 for P1).

Run:
    ./run_dolfinx.sh tests/test_h_convergence.py
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD


def solve(nx: int):
    domain = mesh.create_rectangle(
        COMM,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        n=(nx, nx),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)

    u_exact_ufl = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    u_D = fem.Function(V)
    u_D.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    facets = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda X: np.full(X.shape[1], True)
    )
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
    bc = fem.dirichletbc(u_D, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    problem = LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        f * v * ufl.dx,
        petsc_options_prefix=f"h{nx}_",
        bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
    )
    uh = problem.solve()

    ue = fem.Function(V)
    ue.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    e = uh - ue
    L2 = float(np.sqrt(fem.assemble_scalar(fem.form(e**2 * ufl.dx))))
    L2_ref = float(np.sqrt(fem.assemble_scalar(fem.form(ue**2 * ufl.dx))))
    H1s = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))))
    H1s_ref = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(ue), ufl.grad(ue)) * ufl.dx))))
    ndof = V.dofmap.index_map.size_global
    return L2 / L2_ref, H1s / H1s_ref, ndof


if __name__ == "__main__":
    ns = [8, 16, 32, 64, 128]
    results = [solve(n) for n in ns]

    if COMM.rank == 0:
        print("\n=== h-refinement: P1 FEM on unit square ===")
        print(f"{'n':>5} {'DOFs':>8} {'rel L2':>12} {'rel H1semi':>12} {'L2 rate':>9} {'H1 rate':>9}")
        print("-" * 62)
        for i, (n, (L2, H1, ndof)) in enumerate(zip(ns, results)):
            if i == 0:
                print(f"{n:>5} {ndof:>8} {L2:>12.3e} {H1:>12.3e} {'—':>9} {'—':>9}")
            else:
                prev_L2, prev_H1, _ = results[i - 1]
                hr = ns[i - 1] / n
                L2r = np.log(prev_L2 / L2) / np.log(hr)
                H1r = np.log(prev_H1 / H1) / np.log(hr)
                print(f"{n:>5} {ndof:>8} {L2:>12.3e} {H1:>12.3e} {L2r:>9.2f} {H1r:>9.2f}")
        print("\nExpected: L2 rate ≈ 2.0, H1 rate ≈ 1.0 for P1")
