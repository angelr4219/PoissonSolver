import math
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import geometry, fem
import ufl

COMM = MPI.COMM_WORLD
RANK = COMM.rank

EPS0 = 8.8541878128e-12


# ---------- Analytic point-charge formulas ----------

def phi_point_charge(r: np.ndarray, q: float, eps_r: float = 1.0) -> np.ndarray:
    """Analytic potential of a point charge in infinite homogeneous medium."""
    r = np.asarray(r)
    r_safe = np.where(r > 0, r, np.finfo(float).eps)
    return q / (4.0 * math.pi * EPS0 * eps_r * r_safe)


def E_point_charge(r: np.ndarray, q: float, eps_r: float = 1.0) -> np.ndarray:
    """Analytic |E| of a point charge in infinite homogeneous medium."""
    r = np.asarray(r)
    r_safe = np.where(r > 0, r, np.finfo(float).eps)
    return q / (4.0 * math.pi * EPS0 * eps_r * r_safe**2)


# ---------- Total charge check ----------

def verify_total_charge(rho: fem.Function, q_target: float, label: str = ""):
    """Check that ∫ ρ dV matches the intended total charge."""
    mesh_obj = rho.function_space.mesh
    dx = ufl.dx(domain=mesh_obj)
    Q_num = fem.assemble_scalar(fem.form(rho * dx))
    rel_err = abs(Q_num - q_target) / abs(q_target) if q_target != 0 else abs(Q_num)
    if RANK == 0:
        print(
            f"[verify{label}] Total charge ∫ρ dV = {Q_num:.4e} C "
            f"(target {q_target:.4e}, rel.err = {rel_err:.3e})"
        )


# ---------- Line probe using new geometry API ----------

def line_probe_z(
    phi: fem.Function,
    E: fem.Function | None,
    x0: float,
    y0: float,
    z_min: float,
    z_max: float,
    npts: int = 200,
):
    """
    Sample φ (and optionally |E|) along the vertical line (x0, y0, z).
    Returns: z (npts,), phi_vals (npts,), E_mag_vals (npts or None).
    """
    mesh_obj = phi.function_space.mesh
    gdim = mesh_obj.geometry.dim

    # 1) Build list of points: shape (npts, 3)
    zs = np.linspace(z_min, z_max, npts)
    points = np.zeros((npts, 3), dtype=mesh_obj.geometry.x.dtype)
    points[:, 0] = x0
    points[:, 1] = y0
    if gdim == 3:
        points[:, 2] = zs
    elif gdim == 2:
        # In case of 2D mesh (x,z) with y=0
        points[:, 1] = zs
    else:
        raise RuntimeError(f"Unsupported geometry dim gdim={gdim} for line_probe_z")

    # 2) Build bounding box tree (new API)
    tree = geometry.bb_tree(mesh_obj, mesh_obj.topology.dim)

    # 3) Candidate cells and actual colliding cells for all points
    candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(mesh_obj, candidates, points)

    # 4) Pick one cell per point (or -1 if none)
    cells = np.full(npts, -1, dtype=np.int32)
    for i in range(npts):
        links = colliding_cells.links(i)
        if len(links) > 0:
            cells[i] = links[0]

    # 5) Evaluate phi and E on those cells
    phi_vals = np.full(npts, np.nan)
    E_mag_vals = None if E is None else np.full(npts, np.nan)

    # Only evaluate where we found a valid cell
    mask = cells >= 0
    if np.any(mask):
        x_eval = points[mask]
        c_eval = cells[mask]

        # phi: scalar
        vals_phi = phi.eval(x_eval, c_eval)
        phi_vals[mask] = np.asarray(vals_phi).reshape(-1)

        # E: vector → take norm
        if E_mag_vals is not None:
            vals_E = E.eval(x_eval, c_eval)
            vals_E = np.asarray(vals_E)
            if vals_E.ndim == 1:
                # Single point case
                E_mag_vals[mask] = np.linalg.norm(vals_E)
            else:
                E_mag_vals[mask] = np.linalg.norm(vals_E, axis=1)

    # For 3D, zs is the actual z; for 2D case where we used (x,z),
    # we already stored z in zs above.
    return zs, phi_vals, E_mag_vals


# ---------- Point-charge radial profile + CSV ----------

def verify_point_charge_profile(
    phi: fem.Function,
    E: fem.Function | None,
    q: float,
    xq: float,
    yq: float,
    zq: float,
    z_min: float,
    z_max: float,
    npts: int = 200,
    eps_r_bg: float = 1.0,
    r_min_factor: float = 3.0,
    outdir: str | Path | None = None,
    basename: str = "pointcharge_zline",
):
    """
    Verify that φ and |E| along z through the charge behave like 1/r and 1/r^2,
    and optionally dump a CSV ready to plot.

    Columns in CSV (if written):
        z, r, phi_num, phi_ref, E_num, E_ref
    """
    zs, phi_num, E_mag_num = line_probe_z(phi, E, xq, yq, z_min, z_max, npts)

    # Radial distance from the charge along this line
    r = np.abs(zs - zq)

    # Analytic reference (for the background ε)
    phi_ref = phi_point_charge(r, q, eps_r=eps_r_bg)
    E_ref = E_point_charge(r, q, eps_r=eps_r_bg)

    # Only use points far enough from the charge (throw away near-field where Gaussian shape matters)
    dz = (z_max - z_min) / max(npts - 1, 1)
    r_min = r_min_factor * dz
    mask = (r > r_min) & np.isfinite(phi_num) & (np.abs(phi_ref) > 0)

    if not np.any(mask):
        if RANK == 0:
            print("[verify] No valid points for radial falloff check.")
        return

    r_sel = r[mask]
    phi_num_sel = phi_num[mask]
    phi_ref_sel = phi_ref[mask]

    # ---- numeric checks for φ ----
    rel_err_phi = np.linalg.norm(phi_num_sel - phi_ref_sel) / np.linalg.norm(phi_ref_sel)

    log_r = np.log10(r_sel)
    log_phi = np.log10(np.abs(phi_num_sel))
    slope_phi, intercept_phi = np.polyfit(log_r, log_phi, 1)

    if RANK == 0:
        print(f"[verify] φ radial profile L2 rel.err (r > {r_min:.3e} m): {rel_err_phi:.3e}")
        print(f"[verify] log10|φ| vs log10 r slope ≈ {slope_phi:.3f} (expected ≈ -1)")

    # ---- numeric checks for |E| ----
    rel_err_E = None
    slope_E = None
    if E_mag_num is not None:
        E_num_sel = E_mag_num[mask]
        E_ref_sel = E_ref[mask]
        rel_err_E = np.linalg.norm(E_num_sel - E_ref_sel) / np.linalg.norm(E_ref_sel)

        log_E = np.log10(np.abs(E_num_sel))
        slope_E, intercept_E = np.polyfit(log_r, log_E, 1)

        if RANK == 0:
            print(f"[verify] |E| radial profile L2 rel.err: {rel_err_E:.3e}")
            print(f"[verify] log10|E| vs log10 r slope ≈ {slope_E:.3f} (expected ≈ -2)")

    # ---- CSV dump for plotting ----
    if outdir is not None and RANK == 0:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        if E_mag_num is None:
            E_num_col = np.full_like(zs, np.nan, dtype=float)
            E_ref_col = np.full_like(zs, np.nan, dtype=float)
        else:
            E_num_col = E_mag_num
            E_ref_col = E_ref

        data = np.column_stack([zs, r, phi_num, phi_ref, E_num_col, E_ref_col])
        header = "z, r, phi_num, phi_ref, E_num, E_ref"
        csv_path = outdir / f"{basename}.csv"
        np.savetxt(csv_path, data, delimiter=",", header=header)
        print(f"[verify] wrote line probe CSV to {csv_path}")

    # Optionally return data if caller wants to use it directly
    return zs, r, phi_num, phi_ref, E_mag_num, E_ref
