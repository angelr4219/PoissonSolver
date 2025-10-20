from __future__ import annotations
import argparse, csv, os, sys
from typing import Sequence, List
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, io
from petsc4py import PETSc

# Allow both: module run (-m verify.compare_rect_vs_analytic) and script run
try:
    from .analytic_rect import phi0_rect
except Exception:
    sys.path.insert(0, os.path.dirname(__file__))
    from analytic_rect import phi0_rect  # type: ignore


def parse_xs(xs_str: str, a: float) -> List[float]:
    out: List[float] = []
    for token in xs_str.split(","):
        t = token.strip()
        if not t:
            continue
        if "a" in t:
            t = t.replace("a", str(a))
            out.append(float(eval(t, {"__builtins__": {}}, {})))
        else:
            out.append(float(t))
    return out


def gmax(comm, x: float) -> float:
    return comm.allreduce(float(x), op=MPI.MAX)


def gsum(comm, x: float) -> float:
    return comm.allreduce(float(x), op=MPI.SUM)


def load_mesh_and_field(phi_xdmf: str, field_name: str, comm: MPI.Intracomm):
    """
    Load mesh and scalar nodal field from XDMF.

    Tries DOLFINx native read first (XDMFFile.read_function).
    If not available in this build, falls back to:
      (1) read mesh via dolfinx.io.XDMFFile
      (2) read nodal data via meshio.xdmf.TimeSeriesReader (step 0)
      (3) map to CG1 dofs
    """
    # Try native read_function path
    try:
        with io.XDMFFile(comm, phi_xdmf, "r") as xdmf:
            try:
                mesh = xdmf.read_mesh(name="mesh")
            except Exception:
                mesh = xdmf.read_mesh()
        V = fem.functionspace(mesh, ("CG", 1))
        phi_h = fem.Function(V, name=field_name)
        with io.XDMFFile(comm, phi_xdmf, "r") as xdmf:
            # Some builds support read_function(u, name=...), try both named/unnamed
            try:
                xdmf.read_function(phi_h, name=field_name)  # type: ignore[attr-defined]
            except Exception:
                xdmf.read_function(phi_h)  # type: ignore[attr-defined]
        return mesh, V, phi_h
    except AttributeError:
        # This build doesn't have read_function API; proceed to fallback
        pass
    except Exception:
        # Any issues above -> fallback
        pass

    # Fallback: mesh through dolfinx, field through meshio TimeSeriesReader
    import meshio

    # Read mesh (dolfinx)
    with io.XDMFFile(comm, phi_xdmf, "r") as xdmf:
        try:
            mesh = xdmf.read_mesh(name="mesh")
        except Exception:
            mesh = xdmf.read_mesh()
    V = fem.functionspace(mesh, ("CG", 1))
    phi_h = fem.Function(V, name=field_name)

    # Read nodal data
    with meshio.xdmf.TimeSeriesReader(phi_xdmf) as reader:
        points, cells = reader.read_points_cells()
        try:
            # Newer meshio
            _, point_data, _ = reader.read_data(0)
        except TypeError:
            data = reader.read_data(0)
            # Older meshio may return (t, point_data, cell_data) or dict-like
            if isinstance(data, tuple):
                point_data = data[1]
            else:
                point_data = data["point_data"]

    pd = point_data
    key = field_name if field_name in pd else ("phi" if "phi" in pd else next(iter(pd.keys())))
    if key != field_name and comm.rank == 0:
        print(f"[WARN] Using point_data field '{key}' (no '{field_name}' found).")
    vals = np.asarray(pd[key], dtype=float).reshape(-1)

    if vals.size != phi_h.x.array.size:
        raise RuntimeError(
            f"Point-data size {vals.size} != FE dof count {phi_h.x.array.size}. "
            "Ensure the XDMF stores point/vertex data for CG1 on this mesh."
        )

    phi_h.x.array[:] = vals
    return mesh, V, phi_h


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Relative error vs analytic rectangular-gate potential (same mesh/space)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phi-xdmf", required=True, help="XDMF file with mesh + φ_h")
    parser.add_argument("--a", type=float, required=True, help="Half gate size (m)")
    parser.add_argument("--xs", required=True, help="CSV gate centres (e.g. -2a,0,2a or raw metres)")
    parser.add_argument("--Vs", required=True, help="CSV gate voltages (e.g. 0.25,0.10,0.25)")
    parser.add_argument("--tau", type=float, default=1e-9, help="Denominator floor for relative error")
    parser.add_argument("--out-stem", type=str, default=None, help="Output stem under results/verify/")
    zg = parser.add_mutually_exclusive_group()
    zg.add_argument("--z-from-mesh-midplane", action="store_true", help="Evaluate analytic at nodal z (3D mesh).")
    zg.add_argument("--z", dest="z_const", type=float, default=None, help="Constant z plane if not using mesh z.")
    parser.add_argument("--sample-line", nargs=5, type=float, metavar=("y", "z", "x0", "x1", "n"),
                        help="Sample along a line y,z from x0 to x1 (n points) and write CSV.")
    parser.add_argument("--field-name", type=str, default="phi", help="XDMF Attribute/field name for φ_h")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(f"[args] phi={args.phi_xdmf} a={args.a} xs={args.xs} Vs={args.Vs} tau={args.tau} field={args.field_name}")
    rank = comm.rank

    xs_vals = parse_xs(args.xs, args.a)
    vs_vals = [float(v) for v in args.Vs.split(",")]
    if len(xs_vals) != len(vs_vals):
        raise SystemExit(f"xs ({len(xs_vals)}) and Vs ({len(vs_vals)}) must have same length")

    base_name = os.path.splitext(os.path.basename(args.phi_xdmf))[0]
    outdir = os.path.join("results", "verify")
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
    out_stem = args.out_stem or os.path.join(outdir, base_name)

    # Load mesh + field
    mesh, V, phi_h = load_mesh_and_field(args.phi_xdmf, args.field_name, comm)
    if comm.rank==0: print('[step] loaded mesh+phi_h')

    # --- Scalar check (version-agnostic): require scalar-valued space ---
try:
    _tmp = fem.Function(V)
    is_scalar = (getattr(_tmp, "value_size", 1) == 1)
except Exception:
    # Fallback: assume scalar if we can't construct a Function, which is rare
    is_scalar = True

if not is_scalar:
    if rank == 0:
        print("[ERROR] Space is vector-valued; aborting.")
    sys.exit(2)

    # Build analytic reference on same dofs
    dof_xyz = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
    x, y = dof_xyz[:, 0], dof_xyz[:, 1]
    z_nodes = dof_xyz[:, 2] if dof_xyz.shape[1] == 3 else np.zeros_like(x)
    z_eval = z_nodes if args.z_from_mesh_midplane else (z_nodes if args.z_const is None else np.full_like(x, args.z_const, dtype=float))

    phi_ref_vals = phi0_rect(x, y, z_eval, a=args.a, xs=xs_vals, Vs=vs_vals)
    if comm.rank==0: print('[step] building analytic on dofs')
    phi_ref = fem.Function(V, name="phi_ref")
    phi_ref.x.array[:] = phi_ref_vals

    # Nodal relative error
    denom = np.maximum(np.abs(phi_ref.x.array), args.tau)
    r_dof = np.abs(phi_h.x.array - phi_ref.x.array) / denom
    if comm.rank==0: print('[step] nodal rel error done')

    # DG0 projection for cellwise error: r(x) = |φ_h-φ_ref| / max(|φ_ref|, τ)
    V0 = fem.functionspace(mesh, ("DG", 0))
    u = ufl.TrialFunction(V0)
    v = ufl.TestFunction(V0)
    r_expr = ufl.sqrt((phi_h - phi_ref)**2) / ufl.max_value(ufl.sqrt((phi_ref)**2), args.tau)
    aP = fem.form(u * v * ufl.dx)
    LP = fem.form(r_expr * v * ufl.dx)
    r_cell = fem.Function(V0, name="rel_error_cell")
    if comm.rank==0: print('[step] cellwise rel error done')
    fem.petsc.LinearProblem(aP, LP, u=r_cell, petsc_options={"ksp_type": "preonly", "pc_type": "jacobi"}).solve()

    # Global stats
    Linf = gmax(comm, np.max(r_dof) if r_dof.size else 0.0)
    L2 = np.sqrt(gsum(comm, float(np.dot(r_dof, r_dof))))
    arr = r_cell.x.array
    cmax = gmax(comm, np.max(arr) if arr.size else 0.0)
    cmean = gsum(comm, float(np.sum(arr))) / gsum(comm, float(arr.size if arr.size else 1))

    # Write outputs
    if rank == 0:
        print(f"[REL-DOF] Linf={Linf:.6e} L2={L2:.6e}")
        print(f"[REL-CELL] max={cmax:.6e} mean={cmean:.6e}")

    if comm.rank==0: print('[step] writing outputs')
    with io.XDMFFile(comm, f"{out_stem}_rel_error_cellwise.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(r_cell)

    rel_dof_fun = fem.Function(V, name="rel_error_dof")
    rel_dof_fun.x.array[:] = r_dof
    with io.XDMFFile(comm, f"{out_stem}_rel_error_nodal.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(rel_dof_fun)

    if rank == 0:
        with open(f"{out_stem}_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phi_xdmf", args.phi_xdmf])
            w.writerow(["a", args.a])
            w.writerow(["xs", ";".join(map(str, xs_vals))])
            w.writerow(["Vs", ";".join(map(str, vs_vals))])
            w.writerow(["tau", args.tau])
            w.writerow(["REL_DOF_Linf", f"{Linf:.6e}"])
            w.writerow(["REL_DOF_L2", f"{L2:.6e}"])
            w.writerow(["REL_CELL_max", f"{cmax:.6e}"])
            w.writerow(["REL_CELL_mean", f"{cmean:.6e}"])

    # Optional sampling along a line
    if args.sample_line is not None:
        y0, z0, x0, x1, n = args.sample_line
        n = int(n)
        xs_line = np.linspace(x0, x1, n)
        ys_line = np.full(n, y0)
        zs_line = np.full(n, z0)
        phi_ref_line = phi0_rect(xs_line, ys_line, zs_line, args.a, xs_vals, vs_vals)
        pts = np.vstack((xs_line, ys_line, zs_line))
        vals = np.zeros(n, dtype=float)
        fem.Function.eval(phi_h, vals, pts)
        delta = vals - phi_ref_line
        rline = np.abs(delta) / np.maximum(np.abs(phi_ref_line), args.tau)
        if rank == 0:
            with open(f"{out_stem}_line_sample.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "phi_h", "phi_ref", "delta", "rel"])
                for xi, ph, pr, de, rr in zip(xs_line, vals, phi_ref_line, delta, rline):
                    w.writerow([f"{xi:.9e}", f"{ph:.9e}", f"{pr:.9e}", f"{de:.9e}", f"{rr:.9e}"])

if __name__ == "__main__":
    main()
