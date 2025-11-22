# verify/mesh_sweep.py
# Reuse your existing run_* functions. No re-derivations here.
import argparse, csv, time, importlib, json, math
from typing import Any, Dict, Iterable

def import_callable(dotted: str):
    # "pkg.mod:func" -> callable
    if ":" not in dotted:
        raise ValueError("Use dotted path like 'pkg.module:function'")
    mod, fn = dotted.split(":", 1)
    m = importlib.import_module(mod)
    return getattr(m, fn)

def parse_nxs(nxs_arg: str, min_nx: int, levels: int, mult: float) -> Iterable[int]:
    if nxs_arg:
        return [int(x) for x in nxs_arg.split(",")]
    nxs = []
    nx = float(min_nx)
    for _ in range(levels):
        nxs.append(int(round(nx)))
        nx *= mult
    return nxs

KNOWN_KEYS = [
    # errors commonly returned by your codes — we'll record any that appear
    "L2","H1","H1_semi","Linf","Linf_dof",
    "REL_CELL_max","REL_CELL_mean","REL_DOF_L2","REL_DOF_Linf",
    # mesh/solver info
    "h","dof","ndof","ksp_iters","assemble_time","solve_time","t_assemble","t_solve","t_total"
]

def run_once(fn, nx: int, deg: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Call your solver function and time it.
    t0 = time.perf_counter()
    result = fn(nx=nx, deg=deg, **kwargs)
    t_total = time.perf_counter() - t0

    row = {"nx": nx, "deg": deg, "t_total": t_total}

    # If your function already timed parts, keep them; otherwise just total.
    if isinstance(result, dict):
        for k, v in result.items():
            # Keep only JSON-friendly simple types
            if isinstance(v, (int, float, str)) and math.isfinite(v) if isinstance(v, float) else True:
                row[k] = v

    # Normalize some common aliases
    if "ndof" in row and "dof" not in row:
        row["dof"] = row["ndof"]

    return row

def header_for(rows: Iterable[Dict[str, Any]]):
    # Start with guaranteed columns, then union of known keys that actually appear.
    base = ["case","nx","deg","t_total"]
    seen = set()
    for r in rows:
        for k in KNOWN_KEYS:
            if k in r: seen.add(k)
    # Keep a stable order for nice CSVs
    ordered = base + [k for k in KNOWN_KEYS if k in seen]
    # Plus any extra unknown-but-useful keys returned by your function
    extras = sorted(k for r in rows for k in r.keys() if k not in ordered)
    return ordered + [e for e in extras if e not in ("case","nx","deg","t_total")]

def main():
    p = argparse.ArgumentParser(description="Mesh-size sweep harness (reuses your run_* functions)")
    p.add_argument("--fn", required=True,
                   help="Dotted path to your solver function, e.g. 'verify.mms:run_mms' or 'bench.rect:run_gates'")
    p.add_argument("--case", default="CASE", help="Name to tag in CSV (e.g. MMS, GATES)")
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--nxs", default="", help="Comma list of nx values (e.g. '32,48,64'). If empty, uses min/max/levels.")
    p.add_argument("--min_nx", type=int, default=32)
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--mult", type=float, default=1.5, help="Geometric growth for nx if --nxs not provided")
    p.add_argument("--out", default="results/mesh_sweep.csv")
    p.add_argument("--kwargs", default="{}", help="JSON dict forwarded to your function (gate params, etc.)")
    args = p.parse_args()

    fn = import_callable(args.fn)
    kwargs = json.loads(args.kwargs)
    nxs = parse_nxs(args.nxs, args.min_nx, args.levels, args.mult)

    rows = []
    for nx in nxs:
        row = run_once(fn, nx=nx, deg=args.deg, kwargs=kwargs)
        row["case"] = args.case
        rows.append(row)
        # stream to console for quick feedback
        errs = {k: row[k] for k in ("L2","H1","REL_CELL_mean","REL_CELL_max","Linf_dof") if k in row}
        print(f"[{args.case}] nx={nx:>4} deg={args.deg} dof={row.get('dof','?')} "
              f"t_total={row['t_total']:.3f}s  {errs}")

    # Write CSV
    header = header_for(rows)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
