#!/usr/bin/env python3
"""
Make a percent-error XDMF by comparing a FEniCSx solution to the analytic
rectangular-gate potential on the *same finite-element DOFs*.

Writes:
  <case>_percent_error.xdmf   (scalar CGk field with % error at nodes)

Notes:
- By default uses each DOF's mesh z-coordinate (so it's a full 3D error field).
- You can override with a constant z-plane via --z.
- Avoids meshio; uses dolfinx.XDMFFile (requires the paired .h5 next to .xdmf).
"""

from __future__ import annotations
import argparse, os
from typing import List, Sequence
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
import ufl

# ---------------- Analytic rectangular-gate potential ----------------
def _g(u, v, z):
    # g(u,v,z) = (1/2π) atan2(u v, z sqrt(u^2+v^2+z^2))
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def phi0_rect(x, y, z, a: float, xs: List[float], Vs: List[float]) -> np.ndarray:
    # minus sign per the paper (Eq. 10–11)
    z = np.where(z == 0.0, np.finfo(float).eps, z)
    s = np.zeros_like(x, dtype=float)
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            _g(a - xi + x, a + y, z) +
            _g(a - xi + x, a - y, z) +
            _g(a + xi - x, a + y, z) +
            _g(a + xi - x, a - y, z)
        )
    return -s

# ---------------- CLI helpers ----------------
def parse_xs(xs_str: str, a: float) -> List[float]:
    out: List[float] = []
    for t in xs_str.split(","):
        t = t.strip()
        if not t: continue
        if "a" in t:
            s = t.replace("a", "").strip()
            coef = 1.0 if s in ("", "+") else (-1.0 if s == "-" else float(s))
            out.append(coef * a)
        else:
            out.append(float(t))
    return out

def parse_csv_floats(s: str) -> List[float]:
    return [float(t.strip()) for t in s.split(",") if t.strip()]

# ---------------- Main ----------------
def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Write % error XDMF versus analytic rectangular-gate potential",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("phi_xdmf", help="XDMF written by dolfinx (must have paired .h5 next to it)")
    p.add_argument("--a", type=float, required=True, help="Gate half-size a (m)")
    p.add_argument("--xs", type=str, required=True, help='Comma list of gate centers (m); allows tokens with "a", e.g., "-2a,0,2a"')
    p.add_argument("--Vs", type=str, required=True, help="Comma list of gate voltages (V), same length as xs")
    p.add_argument("--use-mesh-z", action="store_true", help="Use the DOF z-coordinates (3D error field)")
    p.add_argument("--z", type=float, default=None, help="Constant z (m) if not using mesh z")
    p.add_argument("--field", type=str, default="phi", help="Field name inside XDMF (the FE solution)")
    p.add_argument("--tau", type=float, default=1e-9, help="Floor for |phi_ref| to avoid blow-ups")
    args = p.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    xs = parse_xs(args.xs, args.a)
    Vs = parse_csv_floats(args.Vs)
    if len(xs) != len(Vs):
        if rank == 0:
            raise ValueError(f"xs length {len(xs)} != Vs length {len(Vs)}")
        return

    # Load FE function
    with io.XDMFFile(comm, args.phi_xdmf, "r") as xf:
        try:
            uh = xf.read_function(name=args.field)
        except Exception:
            uh = xf.read_function()  # fallback to first function
    V = uh.function_space
    mesh = V.mesh
    dim = mesh.geometry.dim

    # Pull DOF coordinates
    Xd = V.tabulate_dof_coordinates().reshape((-1, dim))
    x = Xd[:, 0]
    y = Xd[:, 1]
    if args.use_mesh_z:
        z = Xd[:, 2]
    else:
        if args.z is None:
            if rank == 0:
                raise ValueError("Provide --z or pass --use-mesh-z")
            return
        z = np.full_like(x, args.z)

    # Analytic on same DOFs
    phi_ref_vals = phi0_rect(x, y, z, args.a, xs, Vs)

    # Sanitize inputs
    uh_arr = uh.x.array
    uh_arr = np.where(np.isfinite(uh_arr), uh_arr, 0.0)
    phi_ref_vals = np.where(np.isfinite(phi_ref_vals), phi_ref_vals, 0.0)

    # Percent error (%)
    denom = np.maximum(np.abs(phi_ref_vals), args.tau)
    pct = 100.0 * np.abs(uh_arr - phi_ref_vals) / denom

    # Heuristic guard against accidentally reading a mask field (e.g., phi_top_helper full of NaNs):
    # if >50% are exactly 0 and denom>tau almost everywhere, warn.
    zeros = np.count_nonzero(pct == 0.0)
    if zeros > 0.5 * pct.size:
        if rank == 0:
            print("[WARN] Many zeros in percent error; double-check you are using the solution field (try --field phi).")

    # Write as a CG1 for ParaView friendliness (even if original degree>1)
    V1 = fem.functionspace(mesh, ("Lagrange", 1))
    pct_fun = fem.Function(V1, name="percent_error")
    # Interpolate nodal array uh->V1 via an expression to get a clean CG1 field
    # (we interpolate numeric array by building a temporary Function on V and interpolating)
    tmp = fem.Function(V, name="tmp")
    tmp.x.array[:] = pct
    pct_fun.interpolate(tmp)

    stem = os.path.splitext(args.phi_xdmf)[0]
    out = f"{stem}_percent_error.xdmf"
    if rank == 0:
        print(f"[WRITE] {out}")
    with io.XDMFFile(comm, out, "w") as xf:
        xf.write_mesh(mesh)
        xf.write_function(pct_fun)

    if rank == 0:
        # Quick summary
        fin = np.isfinite(pct)
        print(f"[SUMMARY] %error: min={np.nanmin(pct[fin]):.3e}  median={np.nanmedian(pct[fin]):.3e}  max={np.nanmax(pct[fin]):.3e}  (N={pct.size})")

if __name__ == "__main__":
    main()
