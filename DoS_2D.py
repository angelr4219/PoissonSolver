# poisson_2d_dos.py
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io, nls
from dolfinx.fem.petsc import NonlinearProblem

def poisson_2d_dos(Nx: int = 96, Ny: int = 96,
                   eps_r: float = 3.9, rho_vol: float = 0.0,
                   phi_left: float = 0.0, phi_right: float = 0.2,
                   mu_eV: float = 0.10, Ec0_eV: float = 0.0,
                   g_s: float = 2.0, g_v: float = 1.0,
                   m_eff_over_m0: float = 0.19, T: float = 300.0):
    """
    Nonlinear BC at y=1: ε ∂φ/∂y = σ(φ) with σ from 2D DOS model.
    Returns FE solution phi (dolfinx.fem.Function).
    """
    comm = MPI.COMM_WORLD
    # Physical constants
    eps0 = 8.8541878128e-12
    q  = 1.602176634e-19
    kB = 1.380649e-23
    hbar = 1.054571817e-34
    m0 = 9.1093837015e-31

    eps = eps0 * eps_r
    mu  = mu_eV  * q
    Ec0 = Ec0_eV * q
    m_eff = m_eff_over_m0 * m0
    kT = kB * T
    g2D = (g_s * g_v * m_eff) / (2.0 * np.pi * hbar * hbar)

    # Mesh/space
    domain = mesh.create_unit_square(comm, Nx, Ny, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    phi = fem.Function(V)
    v = ufl.TestFunction(V)

    # Mark boundaries for ds (1=L,2=R,3=B,4=T)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    def on_left(X):   return np.isclose(X[0], 0.0)
    def on_right(X):  return np.isclose(X[0], 1.0)
    def on_bottom(X): return np.isclose(X[1], 0.0)
    def on_top(X):    return np.isclose(X[1], 1.0)

    facets_L = mesh.locate_entities_boundary(domain, fdim, on_left)
    facets_R = mesh.locate_entities_boundary(domain, fdim, on_right)
    facets_B = mesh.locate_entities_boundary(domain, fdim, on_bottom)
    facets_T = mesh.locate_entities_boundary(domain, fdim, on_top)

    tags = np.hstack([np.full_like(facets_L, 1), np.full_like(facets_R, 2),
                      np.full_like(facets_B, 3), np.full_like(facets_T, 4)])
    from dolfinx.mesh import meshtags
    facet_mt = meshtags(domain, fdim,
                        np.hstack([facets_L, facets_R, facets_B, facets_T]),
                        tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_mt)

    # Dirichlet (x=0,1)
    phi_L_fun = fem.Function(V); phi_L_fun.x.array[:] = phi_left
    phi_R_fun = fem.Function(V); phi_R_fun.x.array[:] = phi_right
    dofs_L = fem.locate_dofs_topological(V, fdim, facets_L)
    dofs_R = fem.locate_dofs_topological(V, fdim, facets_R)
    bcs = [fem.dirichletbc(phi_L_fun, dofs_L),
           fem.dirichletbc(phi_R_fun, dofs_R)]

    # Constants
    eps_c = fem.Constant(domain, eps)
    rho_c = fem.Constant(domain, rho_vol)
    q_c   = fem.Constant(domain, q)
    kT_c  = fem.Constant(domain, kT)
    g2D_c = fem.Constant(domain, g2D)
    mu_c  = fem.Constant(domain, mu)
    Ec0_c = fem.Constant(domain, Ec0)

    # Initial guess: linear in x
    x = ufl.SpatialCoordinate(domain)
    phi.interpolate(fem.Expression(phi_left + (phi_right - phi_left)*x[0],
                                   V.element.interpolation_points()))

    # 2D DOS surface charge and derivative
    eta = (mu_c - (Ec0_c - q_c*phi)) / kT_c
    softplus = ufl.ln(1 + ufl.exp(eta))            # log(1+exp(η))
    sigma = -q_c * g2D_c * kT_c * softplus         # σ(φ)
    logistic = ufl.exp(eta) / (1 + ufl.exp(eta))   # exp(η)/(1+exp(η))
    # dsigma_dphi used implicitly by dolfinx via J = derivative(F, phi, w)

    # Residual and Jacobian
    F = eps_c * ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx \
        - rho_c * v * ufl.dx \
        - sigma * v * ds(4)
    w = ufl.TrialFunction(V)
    J = ufl.derivative(F, phi, w)

    problem = NonlinearProblem(F, phi, bcs=bcs, J=J)
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    n_it, converged = solver.solve(phi)

    with io.XDMFFile(MPI.COMM_WORLD, "poisson_dos_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain); xdmf.write_function(phi)

    if MPI.COMM_WORLD.rank == 0:
        arr = phi.x.array
        print(f"[2D-DOS] Newton iters={n_it}, converged={converged} | "
              f"phi: min={arr.min():.3e} V, max={arr.max():.3e} V")
        np.save("poisson_dos_solution.npy", arr)

    return phi

if __name__ == "__main__":
    poisson_2d_dos()
