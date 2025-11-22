# verify/gates.py
import argparse, json, time, sys, math
sys.path.append("/app")
import gate_benchmark as GB  # uses your run_once(...)

def infer_dims(a: float, pad: float, zfill: float):
    # 3 gates of width 2a centered at [-2a,0,2a] → span ≈ 6a; add pad*a each side in x and y
    Lx = 2*pad*a + 6*a           # total width
    Ly = 2*pad*a                 # vertical extent (in-plane)
    H  = zfill + pad*a           # out-of-plane height (to field-free top)
    return Lx, Ly, H

def main():
    p = argparse.ArgumentParser(description="Run one rect-gate case via gate_benchmark.run_once (timed).")
    p.add_argument("--nx_hint", type=int, default=128, help="Will set h ≈ min(Lx,Ly,H)/nx_hint")
    p.add_argument("--h", type=float, default=None, help="If provided (meters), use this mesh size directly")
    p.add_argument("--deg", type=int, default=1)  # kept for symmetry; GB.run_once uses 'h' only
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--zfill", type=float, required=True)
    p.add_argument("--pad", type=float, default=3.0)
    p.add_argument("--xs", required=True, help='JSON list or comma string, e.g. "[-7e-8,0,7e-8]"')
    p.add_argument("--Vs", required=True, help='JSON list or comma string, e.g. "[0.25,0.10,0.25]"')
    p.add_argument("--outprefix", default="results/rect", help="Prefix for any outputs GB writes")
    args = p.parse_args()

    # parse list-ish inputs
    def parse_list(s): 
        s=s.strip()
        if s.startswith("["): return [float(x) for x in json.loads(s)]
        return [float(x) for x in s.split(",")]
    xs = parse_list(args.xs)
    Vs = parse_list(args.Vs)

    Lx, Ly, H = infer_dims(args.a, args.pad, args.zfill)
    h = args.h if args.h else min(Lx, Ly, H)/float(args.nx_hint)

    t0 = time.perf_counter()
    res = GB.run_once(Lx=Lx, Ly=Ly, H=H, h=h, a=args.a,
                      xs_gates=xs, Vs_gates=Vs, zbar=args.zfill,
                      outprefix=args.outprefix)
    t_total = time.perf_counter() - t0

    # Build a tidy JSON; include timing + anything GB returned
    out = {"h": h, "nx_hint": args.nx_hint, "deg": args.deg, "t_total": t_total}
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(v, (int, float, str)): out[k] = v
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

