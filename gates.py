# verify/gates.py
import argparse, json, time, sys

def main():
    p = argparse.ArgumentParser(description="Run one rect-gate case (timed).")
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--zfill", type=float, required=True)
    p.add_argument("--pad", type=float, default=3.0)
    p.add_argument("--xs", type=float, nargs="+", required=True)
    p.add_argument("--Vs", type=float, nargs="+", required=True)
    p.add_argument("--module", default="gate_benchmark", help="Module with run_gates(..)")
    p.add_argument("--entry", default="run_gates", help="Function name to call")
    args = p.parse_args()

    sys.path.append("/app")  # so repo root is importable in Docker
    mod = __import__(args.module, fromlist=["*"])
    fn = getattr(mod, args.entry, None)
    if not callable(fn):
        raise SystemExit(f"Entry '{args.entry}' not found in {args.module}")

    t0 = time.perf_counter()
    res = fn(nx=args.nx, deg=args.deg, a=args.a, zfill=args.zfill, pad=args.pad,
             xs=args.xs, Vs=args.Vs)
    res["t_total"] = time.perf_counter() - t0

    keep = ("nx","deg","h","dof","REL_CELL_mean","REL_CELL_max","L2","H1","t_total",
            "t_assemble","t_solve","ksp_iters")
    out = {k: res[k] for k in keep if k in res}
    out["nx"] = args.nx
    out["deg"] = args.deg
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

