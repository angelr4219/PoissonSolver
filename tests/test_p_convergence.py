"""
Test 2: p-refinement convergence (fixed mesh, varying polynomial degree)
=========================================================================
Solves -Δu = f on [0,1]^2 with exact Dirichlet BCs.
Exact solution: u = sin(πx)sin(πy), f = 2π²sin(πx)sin(πy).

Uses a coarse mesh (n=6) so the discretisation error dominates at all p levels
and the exponential p-convergence is visible across the full range p=1..5.

Expected: L2 error drops super-algebraically with p (spectral convergence).
Each degree increase should shrink the error by a large factor.

Run:
    ./run_dolfinx.sh tests/test_p_convergence.py
"""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD

# Coarse enough that discretisation error dominates at p=5
N_MESH = 6


def solve(deg: int):
    domain = mesh.create_rectangle(
        COMM,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        n=(N_MESH, N_MESH),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", deg))
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
        petsc_options_prefix=f"p{deg}_",
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    ue = fem.Function(V)
    ue.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    e = uh - ue
    L2 = float(np.sqrt(fem.assemble_scalar(fem.form(e**2 * ufl.dx))))
    L2_ref = float(np.sqrt(fem.assemble_scalar(fem.form(ue**2 * ufl.dx))))
    H1s = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))))
    ndof = V.dofmap.index_map.size_global
    return L2 / L2_ref, H1s, ndof


if __name__ == "__main__":
    degrees = [1, 2, 3, 4, 5]
    results = [solve(d) for d in degrees]

    if COMM.rank == 0:
        print(f"\n=== p-refinement: mesh n={N_MESH}, varying polynomial degree ===")
        print(f"{'p':>4} {'DOFs':>8} {'rel L2':>12} {'H1semi':>12} {'L2 reduction':>14}")
        print("-" * 56)
        for i, (deg, (L2, H1, ndof)) in enumerate(zip(degrees, results)):
            if i == 0:
                print(f"{deg:>4} {ndof:>8} {L2:>12.3e} {H1:>12.3e} {'—':>14}")
            else:
                prev_L2 = results[i - 1][0]
                reduction = prev_L2 / L2
                print(f"{deg:>4} {ndof:>8} {L2:>12.3e} {H1:>12.3e} {reduction:>14.1f}×")
        print("\nExpected: L2 drops by a large factor each p step (spectral convergence).")
        print("If errors plateau, the mesh is too fine — reduce N_MESH.")
