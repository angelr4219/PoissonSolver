from __future__ import annotations
<<<<<<< HEAD
import json, datetime, numpy as np
from pathlib import Path
from mpi4py import MPI
from dolfinx import fem
from .geometry import build_box
from .materials import make_eps_r_constant, make_eps_r_two_layer
from .bc import gate_top_and_bottom_bcs
from .forms import build_forms
from .solve import solve_linear
from .verify import (compute_metrics, sample_dof_line, write_dg0_projection,
                     write_relerr_dg0, phi0_rect_local)
from .io import write_all_in_one, write_metrics_json, write_line_csv

def run_once(RUN_ROOT, Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, nprobe,
             tag, deg, eps_mode, eps_const, eps_tox, eps_top, eps_bulk,
             write_relerr, write_eps):
    outprefix = str(Path(RUN_ROOT) / tag)
    Path(outprefix).parent.mkdir(parents=True, exist_ok=True)

    domain = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", deg))

    eps_r_field = (make_eps_r_two_layer(domain, eps_tox, eps_top, eps_bulk)
                   if eps_mode == "two" else
                   make_eps_r_constant(domain, eps_const))

    bcs = gate_top_and_bottom_bcs(V, a, xs_gates, Vs_gates, rect_tol=1e-8, ztol=1e-9)
    a_form, L_form, _, _ = build_forms(V, eps_r_field)
    uh, info = solve_linear(a_form, L_form, bcs)

    # top helper (for viz)
    V1 = fem.functionspace(domain, ("Lagrange", 1))
    pads_on_top = fem.Function(V1, name="phi_top_helper")
    X1 = V1.tabulate_dof_coordinates().reshape((-1, 3))
    Z = domain.geometry.x[:, 2]
    z_min = domain.comm.allreduce(float(Z.min()))
    vals3d = np.full(X1.shape[0], np.nan, dtype=float)
    on_top = np.isclose(X1[:, 2], z_min, atol=1e-9)
    for xi, Vi in zip(xs_gates, Vs_gates):
        in_rect = on_top & (X1[:, 0] >= (xi - a)) & (X1[:, 0] <= (xi + a)) & (X1[:, 1] >= -a) & (X1[:, 1] <=  a)
        vals3d[in_rect] = Vi
    pads_on_top.x.array[:] = vals3d

    # metrics vs analytic free-space field
    uE = fem.Function(V, name="phi_exact")
    def _eval_exact(X):
        return np.array([phi0_rect_local(X[0,i], X[1,i], X[2,i], a, xs_gates, Vs_gates)
                         for i in range(X.shape[1])], dtype=float)
    uE.interpolate(_eval_exact)
    mets = compute_metrics(domain, uh, uE, qdeg=4)

    if MPI.COMM_WORLD.rank == 0:
        write_metrics_json(f"{outprefix}_metrics", tag, deg, mets)

    write_all_in_one(domain, uh, pads_on_top, outprefix, eps_r_field if write_eps else None)
    write_dg0_projection(domain, uh, outprefix)

    if write_relerr and eps_mode == "const":
        write_relerr_dg0(domain, uh, a, xs_gates, Vs_gates, outprefix, qdeg=4, tau=1e-9)

    xs_line, uh_line = sample_dof_line(uh, zbar, a, nprobe, h)
    phi0_line = np.array([phi0_rect_local(x, 0.0, zbar, a, xs_gates, Vs_gates) for x in xs_line])

    # edge-safe stats
    dx = np.median(np.diff(xs_line)) if xs_line.size > 1 else a*0.01
    band = 2.0*abs(dx)
    edges = np.concatenate([xs_gates - a, xs_gates + a])
    mask = (np.abs(xs_line) <= 2*a)
    for e_edge in edges:
        mask &= (np.abs(xs_line - e_edge) > band)
    diffs = np.abs(uh_line[mask] - phi0_line[mask])
    err_max = float(np.max(diffs)) if diffs.size else float("nan")
    err_l2  = float(np.sqrt(np.mean(diffs**2))) if diffs.size else float("nan")

    if MPI.COMM_WORLD.rank == 0:
        csv_path = write_line_csv(outprefix, xs_line, uh_line, phi0_line)
        print(f"[LINE] {csv_path}")
        print(f"[{tag}] max|Δφ| = {err_max:.4e} V,  L2 = {err_l2:.4e} V")
    return err_max, err_l2

def main():
    import argparse, numpy as np
    p = argparse.ArgumentParser(description="Rectangular-gate benchmark with spatial εr")
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--zfill", type=float, required=True)
    p.add_argument("--xs", type=str, required=True)
    p.add_argument("--Vs", type=str, required=True)
    p.add_argument("--h", type=float, default=5e-9)
    p.add_argument("--pad", type=float, default=None)
    p.add_argument("--run-root", type=str, default=None)
    p.add_argument("--relerr", type=int, default=0)
    p.add_argument("--write-eps", type=int, default=0)
    p.add_argument("--probe_z", type=float, default=None)
    p.add_argument("--probe_n", type=int, default=401)
    p.add_argument("--epsr", type=float, default=11.7)
    p.add_argument("--two-layer", action="store_true")
    p.add_argument("--t_ox", type=float, default=5e-9)
    p.add_argument("--epsr_top", type=float, default=3.9)
    p.add_argument("--epsr_bulk", type=float, default=11.7)
    args = p.parse_args()

    a = args.a
    zbar = args.probe_z if args.probe_z is not None else a
    xs_gates = np.array(json.loads(args.xs), dtype=float)
    Vs_gates = np.array(json.loads(args.Vs), dtype=float)
    H = args.zfill; h = args.h; deg = args.deg
    write_relerr = bool(args.relerr); write_eps = bool(args.write_eps)
    eps_mode = "two" if args.two_layer else "const"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_ROOT = Path(args.run_root) if args.run_root else Path("results")/"runs"/stamp
    if MPI.COMM_WORLD.rank == 0:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"==> Run root: {RUN_ROOT}")

    pads = [args.pad] if args.pad is not None else [2.0, 3.0, 4.0, 5.0]
    errs = []
    for pval in pads:
        Lx = Ly = 2.0 * pval * a
        tag = f"p{int(pval)}a_deg{deg}"
        if MPI.COMM_WORLD.rank == 0:
            print(f"---- solving: {tag}  (Lx=Ly={Lx:.3e} m, H={H:.3e} m, h≈{h:.1e}, deg={deg})")
        em, el2 = run_once(
            RUN_ROOT, Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, args.probe_n,
            tag, deg,
            eps_mode, args.epsr, args.t_ox, args.epsr_top, args.epsr_bulk,
            write_relerr, write_eps
        )
        errs.append((tag, em, el2))

    if MPI.COMM_WORLD.rank == 0:
        summary_csv = RUN_ROOT / "summary_lineprobe.csv"
        with summary_csv.open("w") as f:
            f.write("tag,err_max,err_l2\n")
            for tag, em, el2 in errs:
                f.write(f"{tag},{em:.6e},{el2:.6e}\n")
        print(f"[SUMMARY] {summary_csv}")
        print(f"==> Finished. Outputs in: {RUN_ROOT}")
=======
import argparse, sys, subprocess, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=["2d","3d"])
    ap.add_argument("--mpirun", type=int, default=1)
    ap.add_argument("--options_file", type=str, default="petsc.options")
    args, rest = ap.parse_known_args()

    script = "verify/dielectric_interface_2D.py" if args.demo=="2d" else "verify/image_charge_3D.py"
    cmd = []
    if args.mpirun > 1:
        cmd += ["mpirun", "-n", str(args.mpirun)]
    cmd += [sys.executable, script, "--options_file", args.options_file]
    cmd += rest
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))
>>>>>>> 7b176cafc436c4f4f7c51f1364ef42c02d769d99

if __name__ == "__main__":
    main()
