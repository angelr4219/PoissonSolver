#!/usr/bin/env python3
"""
DOLFINx: Poisson in a 3D box with multiple point charges + 1D analytic check.

Solve
    -∇·(ε ∇φ) = ρ
with ε = ε_r * ε0 and ρ a sum of smoothed point charges in a rectangular box.

All boundaries use Dirichlet φ = 0 (grounded box).

Extras:
  * Evaluate analytic Coulomb potential from the same point charges
    along a 1D line (z-slice at fixed x,y).
  * Evaluate numerical φ along the same line.
  * Write CSV with columns:
        x, y, z, phi_num, phi_analytic, abs_error, rel_error
  * Print summary error metrics.

Example:

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable bash -lc '
    export PETSC_OPTIONS="-ksp_type cg -pc_type gamg"
    /dolfinx-env/bin/python3 PC/multi_point_charge_box.py \
      --Lx 1e-7 --Ly 1e-7 --H 1e-7 --h 5e-9 \
      --epsr 11.7 \
      --q "[1.602176634e-19,-1.602176634e-19]" \
      --x0 "[0.0, 2.5e-8]" \
      --y0 "[0.0, 0.0]" \
      --z0 "[0.0, 0.0]" \
      --sigma 5e-9 \
      --deg 1 \
      --x-probe 0.0 --y-probe 0.0 \
      --z-min -5e-8 --z-max 5e-8 --npts 401 \
      --run-root results/multi_point_demo
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

    We implement in physical units (C/m^3).
    """
    x = V.tabulate_dof_coordinates()[:, 0]
    y = V.tabulate_dof_coordinates()[:, 1]
    z = V.tabulate_dof_coordinates()[:, 2]

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
    with rho.vector.localForm() as loc:
        loc.setArray(rho_vals)
    rho.x.scatter_forward()

    # Diagnostic: total charge (integral of rho).
    dx = ufl.dx(domain=V.mesh)
    total_q = fem.assemble_scalar(fem.form(rho * dx))
    total_q = COMM.allreduce(total_q, op=MPI.SUM)

    if RANK == 0:
        print(f"[rho] total integrated charge ≈ {total_q:.6e} C "
              f"(target {np.sum(q):.6e} C)")

    return rho


def analytic_phi_multi(x, y, z, xq, yq, zq, q, eps_r, r_min_factor, sigma):
    """
    Analytic Coulomb potential of multiple point charges in homogeneous medium:

        φ(r) = 1/(4π ε_r ε0) ∑ q_i / |r - r_i|.

    We skip points closer than r_min_factor * sigma to *any* charge
    (set φ_analytic = NaN there) to avoid the singularity.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    npts = x.size
    n_q = len(q)

    phi = np.zeros(npts, dtype=np.double)
    eps = eps_r * EPS0
    const = 1.0 / (4.0 * np.pi * eps)
    r_cut = r_min_factor * sigma

    for i in range(n_q):
        dx = x - xq[i]
        dy = y - yq[i]
        dz = z - zq[i]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        # Avoid r = 0 singularity
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_i = const * q[i] / r
        # Where r == 0, set to NaN and handle later
        phi_i[~np.isfinite(phi_i)] = np.nan
        phi += phi_i

    # Mask out very near points (inside r_min_factor * sigma)
    for i in range(n_q):
        dx = x - xq[i]
        dy = y - yq[i]
        dz = z - zq[i]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        phi[r < r_cut] = np.nan

    return phi


# ---------- Boundary conditions ----------

def grounded_boundary(domain, V):
    """
    φ = 0 on all boundaries (Dirichlet BC).
    """
    def on_boundary(x):
        # All boundary facets: use mesh.locate_entities_boundary
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


# ---------- 1D line probe + error check ----------

def lineprobe_z_and_error(domain, phi, params):
    """
    Evaluate numerical and analytic φ along a z-line at fixed (x_probe, y_probe).

    Writes CSV:
        basename_lineprobe_z.csv
    with columns:
        x, y, z, phi_num, phi_analytic, abs_error, rel_error

    Also prints summary error metrics.
    """
    x_probe = params["x_probe"]
    y_probe = params["y_probe"]
    z_min = params["z_min"]
    z_max = params["z_max"]
    npts = params["npts"]
    eps_r = params["eps_r"]
    xq = params["xq"]
    yq = params["yq"]
    zq = params["zq"]
    q = params["q"]
    sigma = params["sigma"]
    r_min_factor = params["r_min_factor"]
    run_root = Path(params["run_root"])
    basename = params.get("basename", "multi_point_box")

    # If z range not specified, use full box extents from mesh
    if z_min is None or z_max is None:
        # Domain extents from mesh coordinates
        coords = domain.geometry.x
        z_min_mesh = np.min(coords[:, 2])
        z_max_mesh = np.max(coords[:, 2])
        if z_min is None:
            z_min = z_min_mesh
        if z_max is None:
            z_max = z_max_mesh

    if RANK == 0:
        print(f"[probe] line along z at (x,y)=({x_probe:g},{y_probe:g}), "
              f"z ∈ [{z_min:g}, {z_max:g}], npts={npts}")

    # Build line points
    z_vals = np.linspace(z_min, z_max, npts)
    x_vals = np.full_like(z_vals, x_probe)
    y_vals = np.full_like(z_vals, y_probe)
    points = np.vstack([x_vals, y_vals, z_vals]).T

    # Use bounding box tree to find cells and evaluate numerical φ
    bb = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb, points)
    colliding_cells = geometry.compute_colliding_cells(domain, candidates, points)

    # Collect points and cells that are actually in this process' part of the mesh
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
        # values has shape (n_local, 1)
        values = values[:, 0]
        for local_i, global_i in enumerate(idx_on_proc):
            phi_num[global_i] = values[local_i]

    # Analytic potential (Coulomb sum)
    phi_analytic = analytic_phi_multi(
        x_vals, y_vals, z_vals, xq, yq, zq, q, eps_r, r_min_factor, sigma
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

    # Summary metrics (on rank 0, assuming serial run)
    if RANK == 0 and np.any(mask):
        max_rel = np.nanmax(rel_error[mask])
        rms_rel = np.sqrt(np.nanmean(rel_error[mask] ** 2))
        print(f"[error] max rel error (slice) ≈ {max_rel:.3e}")
        print(f"[error] RMS rel error (slice) ≈ {rms_rel:.3e}")
        # Count how many points used
        print(f"[error] points used in error: {np.count_nonzero(mask)} / {npts}")

    # Write CSV from rank 0
    if RANK == 0:
        csv_path = run_root / f"{basename}_lineprobe_z.csv"
        data = np.column_stack(
            [x_vals, y_vals, z_vals, phi_num, phi_analytic, abs_error, rel_error]
        )
        header = "x,y,z,phi_num,phi_analytic,abs_error,rel_error"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        print(f"[io] wrote line-probe CSV to {csv_path}")


# ---------- Solver ----------

def solve_poisson_multi_point(params):
    Lx = params["Lx"]
    Ly = params["Ly"]
    H  = params["H"]
    h  = params["h"]
    eps_r = params["eps_r"]
    deg = params["deg"]
    xq = params["xq"]
    yq = params["yq"]
    zq = params["zq"]
    q  = params["q"]
    sigma = params["sigma"]
    run_root = Path(params["run_root"])
    basename = params.get("basename", "multi_point_box")

    ensure_dir(run_root)

    if RANK == 0:
        print("=== multi_point_charge_box ===")
        print(f"Lx={Lx:g}, Ly={Ly:g}, H={H:g}, h≈{h:g}")
        print(f"eps_r={eps_r}, deg={deg}, sigma={sigma:g}")
        print(f"{len(q)} charges, q={q}")

    domain = build_box(Lx, Ly, H, h)
    V_phi = fem.FunctionSpace(domain, ("CG", deg))

    # Permittivity (homogeneous)
    eps_val = eps_r * EPS0
    eps = fem.Constant(domain, PETSC.ScalarType(eps_val))

    # Build RHS ρ
    rho = gaussian_rho_multi(V_phi, xq, yq, zq, q, sigma)

    # Variational problem: -∇·(ε∇φ) = ρ  =>  ε ∇φ·∇v dx = ρ v dx
    phi = ufl.TrialFunction(V_phi)
    v   = ufl.TestFunction(V_phi)

    a = eps * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    bcs = grounded_boundary(domain, V_phi)

    problem = LinearProblem(
        a, L, bcs=bcs, u=None,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        },
    )

    phi_sol = problem.solve()

    # Electric field E = -∇φ, projected into vector space
    V_E = fem.VectorFunctionSpace(domain, ("CG", deg))
    E_expr = -ufl.grad(phi_sol)
    E = fem.Function(V_E)
    E.name = "E"

    # L2 projection: (E, w) = (-∇φ, w)
    w = ufl.TestFunction(V_E)
    aE = ufl.inner(E, w) * ufl.dx
    LE = ufl.inner(E_expr, w) * ufl.dx
    prob_E = LinearProblem(
        aE, LE, bcs=[],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        },
    )
    E = prob_E.solve()

    # Output XDMF (numerical fields only)
    xdmf_path = run_root / f"{basename}.xdmf"
    with io.XDMFFile(COMM, xdmf_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(domain)
        phi_sol.name = "phi"
        rho.name = "rho"
        xdmf.write_function(phi_sol)
        xdmf.write_function(rho)
        xdmf.write_function(E)

    if RANK == 0:
        print(f"[io] wrote XDMF to {xdmf_path}")

    # 1D slice analytic vs numeric check
    lineprobe_z_and_error(domain, phi_sol, params)

    return domain, V_phi, phi_sol, rho, V_E, E


# ---------- CLI ----------

def default_params():
    return {
        "Lx": 1e-7,
        "Ly": 1e-7,
        "H":  1e-7,
        "h":  5e-9,
        "eps_r": 11.7,
        "deg": 1,
        "q": np.array([1.602176634e-19]),
        "xq": np.array([0.0]),
        "yq": np.array([0.0]),
        "zq": np.array([0.0]),
        "sigma": 5e-9,
        "run_root": "results/multi_point_charge_box",
        "basename": "multi_point_box",
        # Probe / verification defaults
        "x_probe": 0.0,
        "y_probe": 0.0,
        "z_min": None,
        "z_max": None,
        "npts": 401,
        "r_min_factor": 3.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Poisson solve in 3D box with multiple smoothed point charges, "
            "plus 1D analytic vs numerical check along a z-line."
        )
    )
    parser.add_argument("--Lx", type=float, default=1e-7)
    parser.add_argument("--Ly", type=float, default=1e-7)
    parser.add_argument("--H", type=float, default=1e-7)
    parser.add_argument("--h", type=float, default=5e-9)
    parser.add_argument("--epsr", type=float, default=11.7)
    parser.add_argument("--deg", type=int, default=1)
    parser.add_argument("--q", type=str,
                        default="[1.602176634e-19]",
                        help="JSON list of charges in Coulombs")
    parser.add_argument("--x0", type=str,
                        default="[0.0]",
                        help="JSON list of x positions (m)")
    parser.add_argument("--y0", type=str,
                        default="[0.0]",
                        help="JSON list of y positions (m)")
    parser.add_argument("--z0", type=str,
                        default="[0.0]",
                        help="JSON list of z positions (m)")
    parser.add_argument("--sigma", type=float, default=5e-9,
                        help="Gaussian width (m)")
    parser.add_argument("--run-root", type=str,
                        default="results/multi_point_charge_box")
    parser.add_argument("--basename", type=str,
                        default="multi_point_box")

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
    parser.add_argument("--r-min-factor", type=float, default=3.0,
                        help="skip analytic eval where r < r_min_factor * sigma")

    args = parser.parse_args()

    params = default_params()
    params["Lx"] = args.Lx
    params["Ly"] = args.Ly
    params["H"]  = args.H
    params["h"]  = args.h
    params["eps_r"] = args.epsr
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

    solve_poisson_multi_point(params)


if __name__ == "__main__":
    main()

