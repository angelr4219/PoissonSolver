#!/usr/bin/env python3
"""
Jackson point-charge benchmark at a planar dielectric interface (DOLFINx).

Key outputs:
  - Pointwise relative error field `err_rel` (heatmap) written to XDMF
  - Single console line:
        RESULT h=... deg=... relerr_max=...

ParaView/XDMF note:
  - For p>1, CG(p) fields are not vertex-only. ParaView’s XDMF reader can misread them.
  - So: when writing XDMF, we ALWAYS interpolate phi/phi_exact/err_rel down to CG1 (PointData).
  - DG0 fields (rho, eps) remain CellData (correct).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem


COMM = MPI.COMM_WORLD
RANK = COMM.rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Jackson point-charge benchmark at a planar dielectric interface (DOLFINx)."
    )

    # --- mesh / geometry ---
    p.add_argument("--h", type=float, default=8e-9, help="Target mesh size (approx).")
    p.add_argument("--Lx", type=float, default=1e-7, help="Domain size in x (m).")
    p.add_argument("--Ly", type=float, default=1e-7, help="Domain size in y (m).")
    p.add_argument("--Lz", type=float, default=1e-7, help="Domain size in z (m).")

    # --- FEM ---
    p.add_argument("--deg", type=int, default=1, help="CG polynomial degree for phi.")

    # --- materials ---
    p.add_argument("--eps1r", type=float, default=11.7, help="Relative permittivity for z>0.")
    p.add_argument("--eps2r", type=float, default=3.9, help="Relative permittivity for z<0.")

    # --- source ---
    p.add_argument("--q", type=float, default=1.602176634e-19, help="Total charge (C).")
    p.add_argument("--z0", type=float, default=1e-8, help="Charge location z0 (m), z0>0.")
    p.add_argument("--sigma", type=float, default=5e-9, help="Gaussian width sigma (m).")

    # --- output ---
    p.add_argument("--outdir", type=str, default="results/jackson_pointcharge",
                   help="Output directory.")
    p.add_argument("--basename", type=str, default="jackson_pc",
                   help="Basename for output files (e.g., XDMF).")

    # --- sweep / output controls ---
    p.add_argument("--no-xdmf", action="store_true",
                   help="Skip writing XDMF output (faster for sweeps).")
    p.add_argument("--csv", type=str, default="",
                   help="If set, append one results row to this CSV (rank 0 only).")
    p.add_argument("--tag", type=str, default="",
                   help="Optional label stored in CSV (e.g., 'h-study' or 'p-study').")

    return p.parse_args()


def _make_mesh(args: argparse.Namespace) -> mesh.Mesh:
    Lx, Ly, Lz = args.Lx, args.Ly, args.Lz
    h = args.h

    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))

    p0 = np.array([-Lx / 2.0, -Ly / 2.0, -Lz / 2.0], dtype=np.float64)
    p1 = np.array([+Lx / 2.0, +Ly / 2.0, +Lz / 2.0], dtype=np.float64)

    return mesh.create_box(
        COMM, [p0, p1], [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )


def _compute_cell_midpoints(domain: mesh.Mesh) -> np.ndarray:
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)  # important for midpoints
    num_local_cells = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_local_cells, dtype=np.int32)
    return mesh.compute_midpoints(domain, tdim, cells)


def _cellwise_dg0_function(domain: mesh.Mesh, values_per_cell: np.ndarray, name: str) -> fem.Function:
    V0 = fem.functionspace(domain, ("DG", 0))
    f0 = fem.Function(V0, name=name)
    if f0.x.array.shape[0] != values_per_cell.shape[0]:
        raise RuntimeError(
            f"DG0 dof mismatch: dofs={f0.x.array.shape[0]} vs values={values_per_cell.shape[0]}"
        )
    f0.x.array[:] = values_per_cell
    return f0


def _build_eps_and_rho(domain: mesh.Mesh, args: argparse.Namespace) -> tuple[fem.Function, fem.Function, float, float]:
    eps0 = 8.8541878128e-12  # F/m
    eps1_abs = args.eps1r * eps0
    eps2_abs = args.eps2r * eps0

    mp = _compute_cell_midpoints(domain)  # (Nc, 3)
    zc = mp[:, 2]

    eps_vals = np.where(zc >= 0.0, eps1_abs, eps2_abs).astype(np.float64)

    # Gaussian charge density at cell centers
    x0 = np.array([0.0, 0.0, args.z0], dtype=np.float64)
    dx = mp - x0[None, :]
    r2 = np.einsum("ij,ij->i", dx, dx)

    sigma = float(args.sigma)
    norm = (sigma * np.sqrt(2.0 * np.pi)) ** 3
    rho_vals = (args.q / norm) * np.exp(-r2 / (2.0 * sigma * sigma))
    rho_vals = rho_vals.astype(np.float64)

    eps_fun = _cellwise_dg0_function(domain, eps_vals, name="eps")
    rho_fun = _cellwise_dg0_function(domain, rho_vals, name="rho")
    return eps_fun, rho_fun, eps1_abs, eps2_abs


def _phi_exact_jackson_callable(q: float, z0: float, eps1_abs: float, eps2_abs: float):
    """
    Callable for fem.Function.interpolate.
    x has shape (3, N). Return shape (N,).
    """
    q_im = q * (eps1_abs - eps2_abs) / (eps1_abs + eps2_abs)          # image in medium 1
    q_tr = q * (2.0 * eps2_abs) / (eps1_abs + eps2_abs)               # transmitted-equivalent charge

    r0 = np.array([0.0, 0.0, z0], dtype=np.float64)
    r0_im = np.array([0.0, 0.0, -z0], dtype=np.float64)

    fourpi = 4.0 * np.pi

    def eval_phi(x: np.ndarray) -> np.ndarray:
        X = x.T  # (N,3)

        d1 = X - r0[None, :]
        R1 = np.sqrt(np.einsum("ij,ij->i", d1, d1))

        dim = X - r0_im[None, :]
        Rim = np.sqrt(np.einsum("ij,ij->i", dim, dim))

        tiny = 1e-30
        R1 = np.maximum(R1, tiny)
        Rim = np.maximum(Rim, tiny)

        z = X[:, 2]
        phi = np.empty_like(z, dtype=np.float64)

        mask1 = z >= 0.0
        if np.any(mask1):
            phi[mask1] = (q / (fourpi * eps1_abs * R1[mask1])) + (q_im / (fourpi * eps1_abs * Rim[mask1]))

        mask2 = ~mask1
        if np.any(mask2):
            phi[mask2] = (q_tr / (fourpi * eps2_abs * R1[mask2]))

        return phi

    return eval_phi


def main() -> None:
    args = parse_args()

    domain = _make_mesh(args)
    V = fem.functionspace(domain, ("CG", args.deg))

    eps_fun, rho_fun, eps1_abs, eps2_abs = _build_eps_and_rho(domain, args)

    # Dirichlet data from analytic phi_exact on the boundary
    phi_exact_eval = _phi_exact_jackson_callable(args.q, args.z0, eps1_abs, eps2_abs)
    phi_D = fem.Function(V, name="phi_D")
    phi_D.interpolate(phi_exact_eval)

    # Boundary facets (ensure connectivity exists)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(phi_D, bc_dofs)

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_fun * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho_fun * v * ufl.dx

    # --- Solve (dolfinx:stable requires petsc_options_prefix) ---
    phi = fem.Function(V, name="phi")

    petsc_prefix = "jackson_pc_"
    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
        "ksp_max_it": 2000,
        "ksp_error_if_not_converged": True,
    }

    problem = LinearProblem(
        a, L,
        bcs=[bc],
        u=phi,
        petsc_options=petsc_opts,
        petsc_options_prefix=petsc_prefix,
    )

    phi = problem.solve()
    phi.name = "phi"

    # Interpolate analytic solution for error computation
    phi_exact = fem.Function(V, name="phi_exact")
    phi_exact.interpolate(phi_exact_eval)

    # -----------------------------
    # Pointwise relative error field (for heatmap)
    # err_rel(x_i) = |phi - phi_exact| / max(|phi_exact|, floor)
    # -----------------------------
    err_rel = fem.Function(V, name="err_rel")

    abs_exact_local = np.abs(phi_exact.x.array)
    max_exact_local = float(abs_exact_local.max()) if abs_exact_local.size else 0.0
    max_exact_global = COMM.allreduce(max_exact_local, op=MPI.MAX)

    floor = max(max_exact_global * 1e-12, 1e-300)
    denom = np.maximum(abs_exact_local, floor)

    err_rel.x.array[:] = np.abs(phi.x.array - phi_exact.x.array) / denom

    max_err_local = float(err_rel.x.array.max()) if err_rel.x.array.size else 0.0
    max_err_global = COMM.allreduce(max_err_local, op=MPI.MAX)

    if RANK == 0:
        print(f"RESULT h={args.h:.6e} deg={args.deg} relerr_max={max_err_global:.12e}", flush=True)

        if args.csv:
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)

            with open(csv_path, "a") as f:
                if write_header:
                    f.write("tag,h,deg,eps1r,eps2r,q,z0,sigma,Lx,Ly,Lz,relerr_max\n")
                f.write(
                    f"{args.tag},{args.h:.12e},{args.deg},{args.eps1r},{args.eps2r},"
                    f"{args.q:.12e},{args.z0:.12e},{args.sigma:.12e},"
                    f"{args.Lx:.12e},{args.Ly:.12e},{args.Lz:.12e},"
                    f"{max_err_global:.12e}\n"
                )

    # XDMF output (optional) — FORCE CG1 for PointData so ParaView behaves for p>1
    if not args.no_xdmf:
        outdir = Path(args.outdir)
        if RANK == 0:
            outdir.mkdir(parents=True, exist_ok=True)

        V1 = fem.functionspace(domain, ("CG", 1))

        def to_CG1(f: fem.Function, name: str) -> fem.Function:
            g = fem.Function(V1, name=name)
            g.interpolate(f)
            return g

        phi1 = to_CG1(phi, "phi")
        phi_exact1 = to_CG1(phi_exact, "phi_exact")
        err_rel1 = to_CG1(err_rel, "err_rel")

        xdmf_path = outdir / f"{args.basename}.xdmf"
        with io.XDMFFile(COMM, xdmf_path.as_posix(), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(phi1)
            xdmf.write_function(phi_exact1)
            xdmf.write_function(err_rel1)

            # DG0 -> CellData (correct)
            xdmf.write_function(rho_fun)
            xdmf.write_function(eps_fun)


if __name__ == "__main__":
    main()
