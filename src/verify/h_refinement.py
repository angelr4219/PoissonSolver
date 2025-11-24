#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

def solve(nx, ny):
    """
    Solve -Δu = f on [0,1]^2 with u = u_exact on the boundary,
    where u_exact = sin(pi x) sin(pi y), f = 2 pi^2 sin(pi x) sin(pi y).

    Returns (rel_L2_error, rel_H1_semi_error).
    """
    # Unit square mesh and P1 space
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        n=(nx, ny),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)

    # Exact solution and f
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Constant eps = 1
    eps = fem.Constant(domain, 1.0)

    # Dirichlet BC on all sides
    u_D = fem.Function(V)
    u_D.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain,
        fdim,
        lambda X: np.full(X.shape[1], True),
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_D, dofs)

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
        },
    )
    uh = problem.solve()

    # ---- Absolute error norms ----
    ue = fem.Function(V)
    ue.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    e = uh - ue

    abs_L2 = np.sqrt(fem.assemble_scalar(fem.form(e**2 * ufl.dx)))
    abs_H1s = np.sqrt(
        fem.assemble_scalar(
            fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx)
        )
    )

    # ---- Exact solution norms (for relative error) ----
    ue_L2 = np.sqrt(fem.assemble_scalar(fem.form(ue**2 * ufl.dx)))
    grad_ue = ufl.grad(ue)
    ue_H1s = np.sqrt(
        fem.assemble_scalar(
            fem.form(ufl.inner(grad_ue, grad_ue) * ufl.dx)
        )
    )

    # Avoid division by zero (for this MMS, ue norms are definitely >0)
    rel_L2 = abs_L2 / ue_L2
    rel_H1s = abs_H1s / ue_H1s

    return rel_L2, rel_H1s


if __name__ == "__main__":
    ns = [16, 32, 64, 128]
    errors = []
    for n in ns:
        rel_L2, rel_H1s = solve(n, n)
        if MPI.COMM_WORLD.rank == 0:
            print(f"n={n:3d}  rel L2={rel_L2:.3e}  rel H1semi={rel_H1s:.3e}")
        errors.append((rel_L2, rel_H1s))

    if MPI.COMM_WORLD.rank == 0:
        # Compute convergence rates (still use relative errors for slope)
        for i in range(1, len(ns)):
            h_ratio = ns[i - 1] / ns[i]
            L2_rate = np.log(errors[i - 1][0] / errors[i][0]) / np.log(h_ratio)
            H1_rate = np.log(errors[i - 1][1] / errors[i][1]) / np.log(h_ratio)
            print(
                f"h refinement between n={ns[i-1]} and n={ns[i]}: "
                f"rel L2 rate ≈ {L2_rate:.2f}, rel H1 rate ≈ {H1_rate:.2f}"
            )
