#!/usr/bin/env python3
"""
DOLFINx: Jackson dielectric interface test with a single point charge.

Geometry:
  - 3D rectangular box with interface at z = 0.
  - Top region (z > 0):  epsilon_1 = eps_r_top * eps0
  - Bottom region (z < 0): epsilon_2 = eps_r_bot * eps0
  - One point charge q located at (x0, y0, z0) with z0 > 0 (in top region).

We solve
    -∇·(ε(x) ∇φ) = ρ
with
    ε(x) = eps_r_top * eps0  for z >= 0
         = eps_r_bot * eps0  for z <  0
and ρ a smoothed Gaussian approximation to a point charge.

All boundaries use Dirichlet φ = 0 (grounded box).

Verification:
  * Evaluate analytic Jackson image-charge potential along a 1D line
    (z-line at fixed x,y).
  * Compare numerical φ from FEniCS to analytic φ_Jackson.
  * Write CSV with columns:
        x, y, z, phi_num, phi_jackson, abs_error, rel_error
  * Print summary error metrics.

Example (charge at z0 = 10 nm above interface):

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable bash -lc '
    export PETSC_OPTIONS="-ksp_type cg -pc_type gamg"
    /dolfinx-env/bin/python3 PC/jackson_interface_point_charge_box.py \
      --Lx 8e-8 --Ly 8e-8 --H 8e-8 --h 5e-9 \
      --epsr-top 3.9 --epsr-bot 11.7 \
      --q "[1.602176634e-19]" \
      --x0 "[0.0]" --y0 "[0.0]" --z0 "[1.0e-8]" \
      --sigma 5e-9 \
      --deg 1 \
      --x-probe 0.0 --y-probe 0.0 \
      --z-min -4e-8 --z-max 4e-8 --npts 401 \
      --run-root results/jackson_interface_d10nm \
      --basename jackson_interface
  '

"""

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
RANK = COMM.rank
EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]


# ---------- Utilities ----------

def parse_list(s, n_expected=None, name=""):
    """
    Parse JSON-like list string, e.g. "[1.0, 2.0, 3.0]".
    Optionally check length.
    """
    try:
        vals = json.loads(s)
    except Exception as exc:
        raise ValueError(f"Could not parse {name} list from '{s}': {exc}")
    if not isinstance(vals, (list, tuple)):
        raise ValueError(f"{name} must be a list; got {type(vals)}")
    vals = [float(v) for v in vals]
    if n_expected is not None and len(vals) != n_expected:
        raise ValueError(
            f"{name}: expected {n_expected} entries, got {len(vals)}"
        )
    return np.array(vals, dtype=float)


def ensure_dir(path: Path):
    if RANK == 0:
        path.mkdir(parents=True, exist_ok=True)
    COMM.barrier()


# ---------- Mesh builder ----------

def build_box(Lx, Ly, H, h):
    """
    Structured BoxMesh with approximate spacing h.
    Domain: x ∈ [-Lx/2, Lx/2], y ∈ [-Ly/2, Ly/2], z ∈ [-H/2, H/2].
    """
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))

    p0 = np.array([-Lx / 2, -Ly / 2, -H / 2], dtype=np.double)
    p1 = np.array([ Lx / 2,  Ly / 2,  H / 2], dtype=np.double)

    domain = mesh.create_box(
        COMM,
        [p0, p1],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron,
    )
    return domain


# ---------- Charge & RHS ----------

def gaussian_rho_multi(V, xq, yq, zq, q, sigma):
    """
    Smooth multiple point charges with normalized Gaussians.

    ρ(r) = ∑_i q_i * G_i(r)
    with ∫ G_i dV = 1, so ∫ ρ dV = ∑ q_i.

    Implemented in physical units (C/m^3).
    """
    coords = V.tabulate_dof_coordinates()
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ndof = x.shape[0]
    n_q = len(q)

    rho_vals = np.zeros(ndof, dtype=np.double)

    norm = 1.0 / ((np.sqrt(2.0 * np.pi) * sigma) ** 3)

    for i in range(n_q):
        dx = x - xq[i]
        dy = y - yq[i]
        dz = z - zq[i]
        r2 = dx * dx + dy * dy + dz * dz
        rho_vals += q[i] * norm * np.exp(-0.5 * r2 / (sigma ** 2))

    # Wrap as FEniCSx Function
    rho = fem.Function(V)
    # Newer dolfinx: use rho.x.array instead of rho.vector.localForm()
    rho.x.array[:] = rho_vals
    rho.x.scatter_forward()

    # Diagnostic: total charge (integral of rho).
    dx = ufl.dx(domain=V.mesh)
    total_q = fem.assemble_scalar(fem.form(rho * dx))
    total_q = COMM.allreduce(total_q, op=MPI.SUM)

    if RANK == 0:
        print(f"[rho] total integrated charge ≈ {total_q:.6e} C "
              f"(target {np.sum(q):.6e} C)")

    return rho


# ---------- Analytic Jackson potential ----------

def analytic_phi_jackson_interface(
    x, y, z,
    xq, yq, zq, q,
    eps_r_top, eps_r_bot,
    r_min_factor=0.0,
):
    """
    Analytic potential for a single point charge near a planar dielectric interface.

    Geometry:
      - Interface at z = 0.
      - Top region (z > 0):  epsilon_1 = eps_r_top * EPS0
      - Bottom region (z < 0): epsilon_2 = eps_r_bot * EPS0
      - One real charge q located at (xq, yq, zq) with zq > 0 (in top region).

    Jackson image-charge result:
      Region 1 (z > 0):
        phi_1(r) = 1/(4π ε1) [ q/|r - r0| + q'/|r - r0'| ],
        r0  = (xq, yq,  zq),
        r0' = (xq, yq, -zq),
        q'  = q * (ε1 - ε2)/(ε1 + ε2).

      Region 2 (z < 0):
        phi_2(r) = 1/(4π ε2) [ q''/|r - r0| ],
        q'' = q * 2ε2/(ε1 + ε2).

    We optionally mask points closer than r_min_factor * |zq| to avoid the singularity.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # For now, assume exactly one charge
    assert len(q) == 1, "Jackson analytic: only one charge supported"
    q_phys = q[0]
    x0, y0, z0 = xq[0], yq[0], zq[0]
    if z0 <= 0.0:
        raise ValueError(
            "analytic_phi_jackson_interface assumes the charge is in the top region (z0 > 0)."
        )

    eps1 = eps_r_top * EPS0
    eps2 = eps_r_bot * EPS0

    # Distances to real and image charges
    dx = x - x0
    dy = y - y0
    dz_real = z - z0
    dz_im   = z + z0  # image at (x0, y0, -z0)

    r_real = np.sqrt(dx*dx + dy*dy + dz_real*dz_real)
    r_im   = np.sqrt(dx*dx + dy*dy + dz_im*dz_im)

    # Image charges
    q_image  = q_phys * (eps1 - eps2) / (eps1 + eps2)    # q'
    q_double = q_phys * (2.0 * eps2) / (eps1 + eps2)     # q''

    phi = np.full_like(r_real, np.nan, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Region 1: z >= 0
        mask_top = (z >= 0.0)
        if np.any(mask_top):
            phi_top = (q_phys / r_real[mask_top] +
                       q_image / r_im[mask_top])
            phi[mask_top] = (1.0 / (4.0 * np.pi * eps1)) * phi_top

        # Region 2: z < 0
        mask_bot = ~mask_top
        if np.any(mask_bot):
            phi_bot = q_double / r_real[mask_bot]
            phi[mask_bot] = (1.0 / (4.0 * np.pi * eps2)) * phi_bot

    # Optional masking very near the real charge
    if r_min_factor > 0.0:
        r_cut = r_min_factor * abs(z0)
        near = (r_real < r_cut)
        phi[near] = np.nan

    return phi


# ---------- Boundary conditions ----------

def grounded_boundary(domain, V):
    """
    φ = 0 on all boundaries (Dirichlet BC).
    """
    def on_boundary(x):
        return np.full(x.shape[1], True, dtype=bool)

    boundary_facets = mesh.locate_entities_boundary(
        domain,
        domain.topology.dim - 1,
        on_boundary,
    )

    boundary_dofs = fem.locate_dofs_topological(
        V, domain.topology.dim - 1, boundary_facets
    )

    bc_val = fem.Function(V)
    bc_val.x.array[:] = 0.0

    bc = fem.dirichletbc(bc_val, boundary_dofs)
    return [bc]


# ---------- 1D line probe + error check (Jackson) ----------

def lineprobe_z_and_error_jackson(domain, phi, params):
    """
    Evaluate numerical and analytic (Jackson image-charge) φ along a z-line
    at fixed (x_probe, y_probe). Write CSV and print error metrics.

    Requires params:
      - x_probe, y_probe, z_min, z_max, npts, r_min_factor
      - eps_r_top, eps_r_bot
      - xq, yq, zq, q
      - run_root, basename
    """
    x_probe = params["x_probe"]
    y_probe = params["y_probe"]
    z_min   = params["z_min"]
    z_max   = params["z_max"]
    npts    = params["npts"]
    r_min_factor = params["r_min_factor"]
    eps_r_top = params["eps_r_top"]
    eps_r_bot = params["eps_r_bot"]
    xq = params["xq"]
    yq = params["yq"]
    zq = params["zq"]
    q  = params["q"]
    run_root = Path(params["run_root"])
    basename = params.get("basename", "jackson_interface")

    # If z-range not specified, use mesh extents
    if z_min is None or z_max is None:
        coords = domain.geometry.x
        z_min_mesh = np.min(coords[:, 2])
        z_max_mesh = np.max(coords[:, 2])
        if z_min is None:
            z_min = z_min_mesh
        if z_max is None:
            z_max = z_max_mesh

    if RANK == 0:
        print(f"[probe] Jackson line along z at (x,y)=({x_probe:g},{y_probe:g}), "
              f"z ∈ [{z_min:g}, {z_max:g}], npts={npts}")

    # Build line points
    z_vals = np.linspace(z_min, z_max, npts)
    x_vals = np.full_like(z_vals, x_probe)
    y_vals = np.full_like(z_vals, y_probe)
    points = np.vstack([x_vals, y_vals, z_vals]).T

    # Bounding box tree to evaluate numerical φ
    bb = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb, points)
    colliding_cells = geometry.compute_colliding_cells(domain, candidates, points)

    points_on_proc = []
    cells = []
    idx_on_proc = []
    for i in range(points.shape[0]):
        links_i = colliding_cells.links(i)
        if len(links_i) > 0:
            idx_on_proc.append(i)
            points_on_proc.append(points[i])
            cells.append(links_i[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
    cells = np.array(cells, dtype=np.int32)

    phi_num = np.full(npts, np.nan, dtype=np.double)

    if points_on_proc.size > 0:
        values = phi.eval(points_on_proc, cells)
        values = values[:, 0]
        for local_i, global_i in enumerate(idx_on_proc):
            phi_num[global_i] = values[local_i]

    # Analytic Jackson potential
    phi_analytic = analytic_phi_jackson_interface(
        x_vals, y_vals, z_vals,
        xq, yq, zq, q,
        eps_r_top=eps_r_top,
        eps_r_bot=eps_r_bot,
        r_min_factor=r_min_factor,
    )

    # Errors
    abs_error = np.full(npts, np.nan, dtype=np.double)
    rel_error = np.full(npts, np.nan, dtype=np.double)

    mask = (
        np.isfinite(phi_num)
        & np.isfinite(phi_analytic)
        & (np.abs(phi_analytic) > 0)
    )
    abs_error[mask] = np.abs(phi_num[mask] - phi_analytic[mask])
    rel_error[mask] = abs_error[mask] / np.abs(phi_analytic[mask])

    if RANK == 0 and np.any(mask):
        max_rel = np.nanmax(rel_error[mask])
        rms_rel = np.sqrt(np.nanmean(rel_error[mask] ** 2))
        print(f"[error] Jackson max rel error (slice) ≈ {max_rel:.3e}")
        print(f"[error] Jackson RMS rel error (slice) ≈ {rms_rel:.3e}")
        print(f"[error] points used in error: {np.count_nonzero(mask)} / {npts}")

    if RANK == 0:
        csv_path = run_root / f"{basename}_jackson_lineprobe_z.csv"
        data = np.column_stack(
            [x_vals, y_vals, z_vals, phi_num, phi_analytic, abs_error, rel_error]
        )
        header = "x,y,z,phi_num,phi_jackson,abs_error,rel_error"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        print(f"[io] wrote Jackson line-probe CSV to {csv_path}")


# ---------- Solver ----------

def solve_poisson_jackson(params):
    Lx = params["Lx"]
    Ly = params["Ly"]
    H  = params["H"]
    h  = params["h"]
    eps_r_top = params["eps_r_top"]
    eps_r_bot = params["eps_r_bot"]
    deg = params["deg"]
    xq = params["xq"]
    yq = params["yq"]
    zq = params["zq"]
    q  = params["q"]
    sigma = params["sigma"]
    run_root = Path(params["run_root"])
    basename = params.get("basename", "jackson_interface")

    ensure_dir(run_root)

    if RANK == 0:
        print("=== Jackson dielectric interface: single point charge ===")
        print(f"Lx={Lx:g}, Ly={Ly:g}, H={H:g}, h≈{h:g}")
        print(f"eps_r_top={eps_r_top}, eps_r_bot={eps_r_bot}, deg={deg}, sigma={sigma:g}")
        print(f"charge(s): q={q}, z0={zq}")

    domain = build_box(Lx, Ly, H, h)
    V_phi = fem.functionspace(domain, ("CG", deg))

    # Variable permittivity: eps(x) piecewise based on z
    x = ufl.SpatialCoordinate(domain)
    eps_r_expr = ufl.conditional(
        ufl.ge(x[2], 0.0),
        eps_r_top,
        eps_r_bot,
    )
    eps = EPS0 * eps_r_expr

    # Build eps_r as a DG0 Function for visualization (top vs bottom permittivity)
    V_eps = fem.functionspace(domain, ("DG", 0))
    eps_r_fun = fem.Function(V_eps)
    eps_r_fun.name = "eps_r"

    def _eps_r_eval(x_arr):
        # x_arr has shape (3, n); return shape (1, n)
        return np.where(x_arr[2] >= 0.0, eps_r_top, eps_r_bot)

    eps_r_fun.interpolate(_eps_r_eval)

    # Build RHS ρ
    rho = gaussian_rho_multi(V_phi, xq, yq, zq, q, sigma)

    # Variational problem: -∇·(ε∇φ) = ρ  =>  ∫ ε ∇φ·∇v dx = ∫ ρ v dx
    phi = ufl.TrialFunction(V_phi)
    v   = ufl.TestFunction(V_phi)

    a = eps * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    bcs = grounded_boundary(domain, V_phi)

    problem = LinearProblem(
        a,
        L,
        petsc_options_prefix="jackson_phi_",
        bcs=bcs,
        u=None,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        },
    )

    phi_sol = problem.solve()

    # Electric field E = -∇φ, projected into a vector-valued CG space
    V_E = fem.functionspace(domain, ("CG", deg, (domain.geometry.dim,)))
    E_expr = -ufl.grad(phi_sol)

    # L2 projection: find E in V_E such that (E, w) = (-∇φ, w) for all w
    E_trial = ufl.TrialFunction(V_E)
    w = ufl.TestFunction(V_E)
    aE = ufl.inner(E_trial, w) * ufl.dx
    LE = ufl.inner(E_expr, w) * ufl.dx

    prob_E = LinearProblem(
        aE,
        LE,
        petsc_options_prefix="jackson_E_",
        bcs=[],
        u=None,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        },
    )
    E = prob_E.solve()
    E.name = "E"

    # Output XDMF
    xdmf_path = run_root / f"{basename}.xdmf"
    with io.XDMFFile(COMM, xdmf_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        phi_sol.name = "phi"
        rho.name = "rho"
        xdmf.write_function(phi_sol)
        xdmf.write_function(rho)
        xdmf.write_function(E)
        xdmf.write_function(eps_r_fun)

    if RANK == 0:
        print(f"[io] wrote XDMF to {xdmf_path}")

    # 1D slice analytic vs numeric Jackson check
    lineprobe_z_and_error_jackson(domain, phi_sol, params)

    return domain, V_phi, phi_sol, rho, V_E, E


# ---------- CLI ----------

def default_params():
    return {
        "Lx": 8e-8,
        "Ly": 8e-8,
        "H":  8e-8,
        "h":  5e-9,
        "eps_r_top": 3.9,
        "eps_r_bot": 11.7,
        "deg": 1,
        "q":  np.array([1.602176634e-19]),
        "xq": np.array([0.0]),
        "yq": np.array([0.0]),
        "zq": np.array([1.0e-8]),  # 10 nm above interface
        "sigma": 5e-9,
        "run_root": "results/jackson_interface_d10nm",
        "basename": "jackson_interface",
        # Probe / verification defaults
        "x_probe": 0.0,
        "y_probe": 0.0,
        "z_min": None,
        "z_max": None,
        "npts": 401,
        "r_min_factor": 0.0,  # set >0 to mask near-singularity points
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Jackson dielectric interface: Poisson solve with variable ε(z) and "
            "single smoothed point charge, plus 1D analytic vs numerical check "
            "along a z-line using the image-charge solution."
        )
    )
    parser.add_argument("--Lx", type=float, default=8e-8)
    parser.add_argument("--Ly", type=float, default=8e-8)
    parser.add_argument("--H", type=float, default=8e-8)
    parser.add_argument("--h", type=float, default=5e-9)
    parser.add_argument("--epsr-top", type=float, default=3.9,
                        help="relative permittivity in top region (z >= 0)")
    parser.add_argument("--epsr-bot", type=float, default=11.7,
                        help="relative permittivity in bottom region (z < 0)")
    parser.add_argument("--deg", type=int, default=1)
    parser.add_argument("--q", type=str,
                        default="[1.602176634e-19]",
                        help="JSON list of charges in Coulombs (use length 1)")
    parser.add_argument("--x0", type=str,
                        default="[0.0]",
                        help="JSON list of x positions (m)")
    parser.add_argument("--y0", type=str,
                        default="[0.0]",
                        help="JSON list of y positions (m)")
    parser.add_argument("--z0", type=str,
                        default="[1.0e-8]",
                        help="JSON list of z positions (m), default 10 nm")
    parser.add_argument("--sigma", type=float, default=5e-9,
                        help="Gaussian width (m)")
    parser.add_argument("--run-root", type=str,
                        default="results/jackson_interface_d10nm")
    parser.add_argument("--basename", type=str,
                        default="jackson_interface")

    # Line-probe options
    parser.add_argument("--x-probe", type=float, default=0.0,
                        help="x coordinate of z-line (m)")
    parser.add_argument("--y-probe", type=float, default=0.0,
                        help="y coordinate of z-line (m)")
    parser.add_argument("--z-min", type=float, default=None,
                        help="z min for probe line (m); default = bottom of box")
    parser.add_argument("--z-max", type=float, default=None,
                        help="z max for probe line (m); default = top of box")
    parser.add_argument("--npts", type=int, default=401,
                        help="number of probe points along z")
    parser.add_argument("--r-min-factor", type=float, default=0.0,
                        help="skip analytic eval where r < r_min_factor * |z0| (0 = off)")

    args = parser.parse_args()

    params = default_params()
    params["Lx"] = args.Lx
    params["Ly"] = args.Ly
    params["H"]  = args.H
    params["h"]  = args.h
    params["eps_r_top"] = args.epsr_top
    params["eps_r_bot"] = args.epsr_bot
    params["deg"] = args.deg
    params["q"]   = parse_list(args.q, name="q")
    n_q = len(params["q"])
    params["xq"]  = parse_list(args.x0, n_expected=n_q, name="x0")
    params["yq"]  = parse_list(args.y0, n_expected=n_q, name="y0")
    params["zq"]  = parse_list(args.z0, n_expected=n_q, name="z0")
    params["sigma"] = args.sigma
    params["run_root"] = args.run_root
    params["basename"] = args.basename

    # Probe params
    params["x_probe"] = args.x_probe
    params["y_probe"] = args.y_probe
    params["z_min"] = args.z_min
    params["z_max"] = args.z_max
    params["npts"] = args.npts
    params["r_min_factor"] = args.r_min_factor

    # For analytic Jackson function we currently only support a single charge.
    if len(params["q"]) != 1:
        raise ValueError("This Jackson verification script expects exactly one charge (len(q) == 1).")

    solve_poisson_jackson(params)


if __name__ == "__main__":
    main()
