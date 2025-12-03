#!/usr/bin/env python3
"""
DOLFINx point-charge verifier (single Gaussian-smoothed charge in a box).

Solves:
    -∇·(ε ∇φ) = ρ

with:
    - uniform ε = eps_r_bg * EPS0
    - Gaussian ρ centered at (x0, y0, z0)
    - Dirichlet φ = 0 on the outer boundary

Then:
    - checks total charge ∫ρ dV ≈ q_point
    - verifies φ ~ 1/r and |E| ~ 1/r^2 along a vertical line through the charge
    - dumps line-probe data to CSV for plotting using verify_pointcharge.py.
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

from verify_pointcharge import (
    verify_total_charge,
    verify_point_charge_profile,
)

# ---------- Physical constants ----------
EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m


# ---------- Problem parameters ----------
def default_params():
    """
    Return a dict of default geometry / solver / charge parameters.
    """
    params = {}

    # Box dimensions (meters)
    params["Lx"] = 200e-9
    params["Ly"] = 200e-9
    params["H"] = 200e-9   # height in z

    # Target cell size (meters) -> determines mesh resolution
    params["h"] = 10e-9

    # FEM degree
    params["deg"] = 1

    # Background relative permittivity
    params["eps_r_bg"] = 1.0  # vacuum; set to e.g. 11.7 for Si

    # Point charge
    params["q_point"] = 1.602176634e-19  # 1e charge, Coulombs
    params["x0"] = 0.0
    params["y0"] = 0.0
    params["z0"] = 0.0

    # Gaussian smearing (standard deviation), meters
    params["sigma"] = 10e-9

    # Output dir
    params["run_root"] = Path("results") / "point_charge_box"

    return params


# ---------- Mesh & spaces ----------
def build_box_mesh(Lx, Ly, H, h):
    """
    Build a tetrahedral box mesh:
        x ∈ [-Lx/2, Lx/2], y ∈ [-Ly/2, Ly/2], z ∈ [-H/2, H/2]
    with cell size ~ h.
    """
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))

    p0 = np.array([-Lx/2, -Ly/2, -H/2], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H/2], dtype=np.double)

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [p0, p1],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain


def dirichlet_boundary_all_sides(domain, V_phi):
    """
    φ = 0 on the entire outer boundary of the box.
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)

    coords = domain.geometry.x
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()

    Lx = float(x_max - x_min)
    Ly = float(y_max - y_min)
    H  = float(z_max - z_min)
    tol = 1e-12 * max(Lx, Ly, H)

    def on_boundary(x):
        return np.logical_or.reduce((
            np.isclose(x[0], x_min, atol=tol),
            np.isclose(x[0], x_max, atol=tol),
            np.isclose(x[1], y_min, atol=tol),
            np.isclose(x[1], y_max, atol=tol),
            np.isclose(x[2], z_min, atol=tol),
            np.isclose(x[2], z_max, atol=tol),
        ))

    boundary_dofs = fem.locate_dofs_geometrical(V_phi, on_boundary)
    bc_value = fem.Constant(domain, 0.0)
    bc = fem.dirichletbc(bc_value, boundary_dofs, V_phi)
    return [bc]


# ---------- Gaussian charge density ----------
def build_gaussian_rho(domain, V_phi, q_point, x0, y0, z0, sigma):
    """
    Build a fem.Function rho on V_phi representing a normalized 3D Gaussian
    with total charge q_point centered at (x0, y0, z0).
    """
    V_rho = V_phi  # same space is fine
    rho = fem.Function(V_rho, name="rho")

    x = V_rho.tabulate_dof_coordinates()
    gdim = domain.geometry.dim
    x = x.reshape((-1, gdim))

    # 3D normalized Gaussian
    norm = q_point / ((sigma * np.sqrt(2.0 * np.pi)) ** 3)

    dx2 = (x[:, 0] - x0) ** 2
    dy2 = (x[:, 1] - y0) ** 2
    dz2 = (x[:, 2] - z0) ** 2
    r2 = dx2 + dy2 + dz2

    rho_vals = norm * np.exp(-r2 / (2.0 * sigma**2))

    rho.x.array[:] = rho_vals

    return rho


# ---------- Solve Poisson + verification ----------
def solve_poisson_point_charge(params):
    """
    Full pipeline:
      - build mesh
      - set up spaces, permittivity, Gaussian rho
      - solve for phi
      - project E = -grad(phi)
      - run verification and write CSV
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    Lx = params["Lx"]
    Ly = params["Ly"]
    H  = params["H"]
    h  = params["h"]
    deg = params["deg"]
    eps_r_bg = params["eps_r_bg"]
    q_point = params["q_point"]
    x0 = params["x0"]
    y0 = params["y0"]
    z0 = params["z0"]
    sigma = params["sigma"]
    run_root = params["run_root"]

    if rank == 0:
        run_root.mkdir(parents=True, exist_ok=True)
        print("=== Point charge in box ===")
        print(f"Lx, Ly, H = {Lx:.3e}, {Ly:.3e}, {H:.3e} m")
        print(f"h ~ {h:.3e} m, deg = {deg}")
        print(f"epsr = {eps_r_bg}, q = {q_point:.6e} C")
        print(f"x0 = {x0},{y0},{z0}, sigma = {sigma:.3e} m")
        print(f"run_root = {run_root}")

    # -- Mesh
    domain = build_box_mesh(Lx, Ly, H, h)
    gdim = domain.geometry.dim

    # -- Function space for phi
    V_phi = fem.functionspace(domain, ("Lagrange", deg))
    phi = fem.Function(V_phi, name="phi")

    # -- Background permittivity (uniform)
    eps_r = fem.Constant(domain, eps_r_bg)
    eps = eps_r * EPS0

    # -- Charge density rho (Gaussian)
    rho = build_gaussian_rho(domain, V_phi, q_point, x0, y0, z0, sigma)

    # -- Weak form: a(u, v) = ∫ ε ∇u·∇v dx,   L(v) = ∫ ρ v dx
    u = ufl.TrialFunction(V_phi)
    v = ufl.TestFunction(V_phi)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = rho * v * ufl.dx

    bcs = dirichlet_boundary_all_sides(domain, V_phi)

    # -- Solve Poisson
    problem = LinearProblem(
        a,
        L_form,
        bcs=bcs,
        u=phi,
        petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
        petsc_options_prefix="pc_",  # required by this DOLFINx build
    )
    phi = problem.solve()

    if rank == 0:
        print("[solve] Poisson solve for point charge completed.")

    # -- Project E = -grad(phi) to a vector space
    V_E = fem.functionspace(domain, ("Lagrange", deg, (gdim,)))
    E = fem.Function(V_E, name="E")

    u_vec = ufl.TrialFunction(V_E)
    v_vec = ufl.TestFunction(V_E)
    a_E = ufl.inner(u_vec, v_vec) * ufl.dx
    L_E = ufl.inner(-ufl.grad(phi), v_vec) * ufl.dx

    problem_E = LinearProblem(
        a_E,
        L_E,
        u=E,
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
        petsc_options_prefix="E_",  # separate prefix label
    )
    E = problem_E.solve()

    if rank == 0:
        print("[solve] Projection of E = -grad(phi) completed.")

    # ---------- Verification ----------
    # 1) Total charge check
    verify_total_charge(rho, q_target=q_point, label=": vacuum box")

    # 2) Radial falloff check along z through the charge, + CSV dump
    z_min = -H / 2.0
    z_max =  H / 2.0

    verify_point_charge_profile(
        phi, E,
        q=q_point,
        xq=x0, yq=y0, zq=z0,
        z_min=z_min, z_max=z_max,
        npts=400,
        eps_r_bg=eps_r_bg,
        r_min_factor=3.0,
        outdir=run_root,
        basename="lineprobe_z",
    )

    return domain, V_phi, phi, rho, V_E, E


# ---------- main ----------
def main():
    params = default_params()
    solve_poisson_point_charge(params)


if __name__ == "__main__":
    main()
