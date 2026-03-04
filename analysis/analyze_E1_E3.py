#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple, Optional


def read_csv(path: str) -> Tuple[List[str], List[Dict[str, float]]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        rows: List[Dict[str, float]] = []
        for row in r:
            d: Dict[str, float] = {}
            for k, v in row.items():
                if v is None:
                    continue
                try:
                    d[k] = float(v)
                except Exception:
                    # ignore non-numeric columns
                    pass
            rows.append(d)
    return header, rows


def first_present(header: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in header:
            return c
    return None


def compute_rel_errors(
    rows: List[Dict[str, float]],
    num_key: str,
    ref_key: str,
    tau: float,
) -> List[float]:
    rels: List[float] = []
    for d in rows:
        if num_key not in d or ref_key not in d:
            continue
        num = d[num_key]
        ref = d[ref_key]

        # mask points where reference ~ 0
        if abs(ref) < tau:
            continue

        denom = max(abs(ref), tau)
        rel = abs(num - ref) / denom

        if math.isfinite(rel):
            rels.append(rel)

    return rels


def stats_E1_E3(rels: List[float]) -> Tuple[float, float]:
    if len(rels) == 0:
        return float("nan"), float("nan")
    E1 = max(rels)
    E3 = sum(rels) / len(rels)
    return E1, E3


def analyze_one_csv(
    path: str,
    tau: float,
    # If these are provided, we use them. Otherwise we auto-detect.
    num_key: Optional[str] = None,
    ref_key: Optional[str] = None,
    # If the CSV already contains a relerr column, we still recompute by default
    # because you explicitly want masking based on |ref|<tau.
) -> Tuple[int, int, float, float, str, str]:
    header, rows = read_csv(path)

    # Auto-detect common Anderson-style columns
    if num_key is None:
        num_key = first_present(header, ["phi_h", "phi_mode", "phi", "phi_extruded", "phi_cavity"])
    if ref_key is None:
        ref_key = first_present(header, ["phi0", "phi_ref", "phi_pads_on_this_mesh", "phi_pads"])

    if num_key is None or ref_key is None:
        raise RuntimeError(
            f"Could not auto-detect num/ref columns in {path}\n"
            f"Header = {header}\n"
            f"Provide --num_key and --ref_key to force them."
        )

    rels = compute_rel_errors(rows, num_key, ref_key, tau)
    E1, E3 = stats_E1_E3(rels)
    return len(rows), len(rels), E1, E3, num_key, ref_key


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute ONLY E1 (max rel) and E3 (mean rel) on the sampling line, masking |ref|<tau."
    )
    ap.add_argument("--tau", type=float, default=1e-9, help="Mask threshold: ignore points where |ref| < tau.")
    ap.add_argument("--csv", action="append", default=[], help="CSV path to analyze. Can be repeated.")
    ap.add_argument("--label", action="append", default=[], help="Label for each --csv (same count).")
    ap.add_argument("--num_key", action="append", default=[], help="Force num column for each CSV (optional).")
    ap.add_argument("--ref_key", action="append", default=[], help="Force ref column for each CSV (optional).")
    args = ap.parse_args()

    if not args.csv:
        raise SystemExit("Provide at least one --csv path.")

    # Normalize lists to match number of csvs
    n = len(args.csv)
    labels = args.label if len(args.label) == n else [os.path.basename(p) for p in args.csv]

    def pick(lst: List[str], i: int) -> Optional[str]:
        return lst[i] if i < len(lst) and lst[i] else None

    print("\n=== E1 / E3 line errors (mask |ref| < tau) ===")
    print(f"tau = {args.tau:g}\n")
    print(f"{'CASE':22s} {'N_used':>8s} {'E1=max_rel':>14s} {'E3=mean_rel':>14s}  (num, ref)")
    print("-" * 92)

    for i, path in enumerate(args.csv):
        lab = labels[i]
        nk = pick(args.num_key, i)
        rk = pick(args.ref_key, i)

        n_total, n_used, E1, E3, used_nk, used_rk = analyze_one_csv(path, args.tau, nk, rk)
        print(f"{lab:22s} {n_used:8d} {E1:14.4e} {E3:14.4e}  ({used_nk}, {used_rk})")

    print("")


if __name__ == "__main__":
    main()
