#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import math
import os

def finite_float(x: str) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def read_max_mean_rel(path: str, tau: float) -> tuple[float, float, int]:
    """
    Returns (max_rel, mean_rel, N) where rel is either:
      - direct column 'relerr' if present, OR
      - computed |delta|/max(|phi_pads|, tau) if columns exist.
    """
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []

        has_relerr = "relerr" in header
        has_delta = ("delta" in header) and ("phi_pads" in header)

        if not has_relerr and not has_delta:
            raise RuntimeError(
                f"{path}: need either column 'relerr' OR columns 'delta' and 'phi_pads'. "
                f"Header={header}"
            )

        vals: list[float] = []
        for row in r:
            if has_relerr:
                v = finite_float(row.get("relerr", ""))
                if v is not None:
                    vals.append(v)
            else:
                d = finite_float(row.get("delta", ""))
                p = finite_float(row.get("phi_pads", ""))
                if d is None or p is None:
                    continue
                rel = abs(d) / max(abs(p), tau)
                if math.isfinite(rel):
                    vals.append(rel)

    if len(vals) == 0:
        return float("nan"), float("nan"), 0

    return max(vals), sum(vals) / len(vals), len(vals)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print only max_rel and mean_rel for Anderson CSVs (pads relerr or delta-based rel)."
    )
    ap.add_argument("--csv", action="append", required=True, help="CSV file path (repeatable).")
    ap.add_argument("--label", action="append", default=[], help="Optional label per CSV (repeatable).")
    ap.add_argument("--tau", type=float, default=1e-9, help="Floor for denominator when computing delta rel (default 1e-9).")
    args = ap.parse_args()

    labels = args.label
    if len(labels) != len(args.csv):
        labels = [os.path.basename(p) for p in args.csv]

    print("\nCASE,max_rel,mean_rel,N\n" + "-" * 40)
    for path, lab in zip(args.csv, labels):
        max_rel, mean_rel, n = read_max_mean_rel(path, tau=args.tau)
        print(f"{lab},{max_rel:.6e},{mean_rel:.6e},{n}")
    print("")

if __name__ == "__main__":
    main()
