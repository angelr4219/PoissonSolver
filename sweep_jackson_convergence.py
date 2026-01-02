#!/usr/bin/env python3
import argparse
import math
import subprocess
from pathlib import Path

def observed_orders(xs, errs):
    """Return list of observed slopes between consecutive points: log(e2/e1)/log(x2/x1)."""
    orders = []
    for i in range(1, len(xs)):
        x1, x2 = xs[i-1], xs[i]
        e1, e2 = errs[i-1], errs[i]
        if e1 <= 0 or e2 <= 0:
            orders.append(float("nan"))
        else:
            orders.append(math.log(e2/e1) / math.log(x2/x1))
    return orders

def run_case(python, solver, common_args, h, deg, outdir, tag, write_xdmf=False):
    basename = f"jackson_h{h:.2e}_p{deg}".replace("+", "").replace(".", "p")
    csv_path = outdir / f"{tag}.csv"

    cmd = [
        python, solver,
        *common_args,
        "--h", f"{h}",
        "--deg", f"{deg}",
        "--outdir", str(outdir),
        "--basename", basename,
        "--csv", str(csv_path),
        "--tag", tag,
    ]
    if not write_xdmf:
        cmd.append("--no-xdmf")

    # Run and stream output
    print("\n" + "="*90)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="h- and p-convergence sweeps for Jackson interface benchmark")
    ap.add_argument("--solver", type=str, default="jackson_point_charge_interface.py")
    ap.add_argument("--outdir", type=str, default="results/jackson_pointcharge/convergence")
    ap.add_argument("--python", type=str, default="/dolfinx-env/bin/python3")

    # Physics / geometry defaults matching your CLI
    ap.add_argument("--Lx", type=float, default=1e-7)
    ap.add_argument("--Ly", type=float, default=1e-7)
    ap.add_argument("--Lz", type=float, default=1e-7)
    ap.add_argument("--eps1r", type=float, default=11.7)
    ap.add_argument("--eps2r", type=float, default=3.9)
    ap.add_argument("--q", type=float, default=1.602176634e-19)
    ap.add_argument("--z0", type=float, default=1e-8)
    ap.add_argument("--sigma", type=float, default=5e-9)

    # Sweep controls
    ap.add_argument("--h_list", type=str, default="1.6e-8,1.2e-8,8e-9,6e-9,4e-9")
    ap.add_argument("--p_list", type=str, default="1,2,3,4")
    ap.add_argument("--h_for_p", type=float, default=8e-9)
    ap.add_argument("--p_for_h", type=int, default=1)

    # If you want XDMF for ONLY the finest case, set this true
    ap.add_argument("--xdmf_finest_only", action="store_true", help="Write XDMF only for finest h in h-study and max p in p-study.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    common = [
        "--Lx", str(args.Lx), "--Ly", str(args.Ly), "--Lz", str(args.Lz),
        "--eps1r", str(args.eps1r), "--eps2r", str(args.eps2r),
        "--q", str(args.q), "--z0", str(args.z0), "--sigma", str(args.sigma),
    ]

    hs = [float(s.strip()) for s in args.h_list.split(",") if s.strip()]
    ps = [int(s.strip()) for s in args.p_list.split(",") if s.strip()]

    # --------------------------
    # h-convergence (fixed p)
    # --------------------------
    tag_h = f"h_study_p{args.p_for_h}"
    for i, h in enumerate(hs):
        write_xdmf = False
        if args.xdmf_finest_only and (i == len(hs) - 1):
            write_xdmf = True
        run_case(args.python, args.solver, common, h=h, deg=args.p_for_h, outdir=outdir, tag=tag_h, write_xdmf=write_xdmf)

    # --------------------------
    # p-convergence (fixed h)
    # --------------------------
    tag_p = f"p_study_h{args.h_for_p:.2e}".replace("+", "").replace(".", "p")
    for i, p in enumerate(ps):
        write_xdmf = False
        if args.xdmf_finest_only and (i == len(ps) - 1):
            write_xdmf = True
        run_case(args.python, args.solver, common, h=args.h_for_p, deg=p, outdir=outdir, tag=tag_p, write_xdmf=write_xdmf)

    print("\nDone. CSV outputs:")
    print(f"  {outdir}/{tag_h}.csv")
    print(f"  {outdir}/{tag_p}.csv")
    print("\nTip: compute observed orders from the CSV (log slopes).")

if __name__ == "__main__":
    main()
