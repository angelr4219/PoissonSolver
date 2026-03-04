#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ErrStats:
    n_total: int
    n_used: int
    max_rel: float   # E1
    l2_rel: float    # E2


def _read_csv(path: str) -> Tuple[List[str], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        for row in r:
            d: Dict[str, float] = {}
            for k, v in row.items():
                if v is None:
                    continue
                try:
                    d[k] = float(v)
                except Exception:
                    pass
            rows.append(d)
    return header, rows


def _compute_stats(
    rows: List[Dict[str, float]],
    phi_num_key: str,
    phi_ref_key: str,
    tau: float,
) -> ErrStats:
    n_total = len(rows)
    rels: List[float] = []
    num2 = 0.0
    den2 = 0.0

    for d in rows:
        if (phi_num_key not in d) or (phi_ref_key not in d):
            continue
        num = d[phi_num_key]
        ref = d[phi_ref_key]

        # Mask near-zero reference to avoid nonsense "huge relative errors"
        if abs(ref) < tau:
            continue

        denom = max(abs(ref), tau)
        r = abs(num - ref) / denom
        rels.append(r)

        num2 += (num - ref) ** 2
        den2 += (ref) ** 2

    n_used = len(rels)
    if n_used == 0:
        return ErrStats(n_total=n_total, n_used=0, max_rel=float("nan"), l2_rel=float("nan"))

    max_rel = max(rels)
    l2_rel = math.sqrt(num2 / den2) if den2 > 0 else float("nan")
    return ErrStats(n_total=n_total, n_used=n_used, max_rel=max_rel, l2_rel=l2_rel)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute E1 (max rel) and E2 (L2 rel) from Anderson line-sample CSVs.")
    ap.add_argument("--root", required=True, help="Root folder containing 01_pads, 02_extruded, 03_cavity.")
    ap.add_argument("--basename", default="three_gates", help="Basename used in the runs.")
    ap.add_argument("--tau", type=float, default=1e-9, help="Mask points where |ref| < tau.")
    ap.add_argument("--out_csv", default="", help="Optional: write summary CSV here.")
    args = ap.parse_args()

    root = args.root
    b = args.basename
    tau = args.tau

    pads_csv = os.path.join(root, "01_pads", f"{b}_pads_line_sample.csv")
    ext_delta_csv = os.path.join(root, "02_extruded", f"{b}_extruded_delta_line_sample.csv")
    cav_delta_csv = os.path.join(root, "03_cavity", f"{b}_cavity_delta_line_sample.csv")

    summary: List[Tuple[str, str, ErrStats]] = []

    # ---- Pads: compare phi_h vs phi0 ----
    if os.path.exists(pads_csv):
        hdr, rows = _read_csv(pads_csv)
        candidates_num = ["phi_h", "phi", "phi_num"]
        candidates_ref = ["phi0", "phi_ref"]

        num_key = next((k for k in candidates_num if k in hdr), None)
        ref_key = next((k for k in candidates_ref if k in hdr), None)

        if num_key is None or ref_key is None:
            raise RuntimeError(
                f"Pads CSV missing expected columns.\n"
                f"Found: {hdr}\nNeed num in {candidates_num} and ref in {candidates_ref}."
            )

        stats = _compute_stats(rows, num_key, ref_key, tau)
        summary.append(("pads (vs analytic)", pads_csv, stats))
    else:
        print(f"[WARN] Missing pads CSV: {pads_csv}")

    # ---- Extruded: compare phi_mode vs phi_pads_on_this_mesh ----
    if os.path.exists(ext_delta_csv):
        hdr, rows = _read_csv(ext_delta_csv)
        mode_key = next((k for k in ["phi_mode", "phi_extruded", "phi"] if k in hdr), None)
        pads_key = next((k for k in ["phi_pads", "phi_pads_on_this_mesh", "phi_ref"] if k in hdr), None)

        if mode_key is None or pads_key is None:
            raise RuntimeError(
                f"Extruded delta CSV missing expected columns.\n"
                f"Found: {hdr}\nNeed mode in [phi_mode, phi_extruded, phi] and pads in [phi_pads, phi_pads_on_this_mesh, phi_ref]."
            )

        stats = _compute_stats(rows, mode_key, pads_key, tau)
        summary.append(("extruded (vs pads)", ext_delta_csv, stats))
    else:
        print(f"[WARN] Missing extruded delta CSV: {ext_delta_csv}")

    # ---- Cavity: compare phi_mode vs phi_pads_on_this_mesh ----
    if os.path.exists(cav_delta_csv):
        hdr, rows = _read_csv(cav_delta_csv)
        mode_key = next((k for k in ["phi_mode", "phi_cavity", "phi"] if k in hdr), None)
        pads_key = next((k for k in ["phi_pads", "phi_pads_on_this_mesh", "phi_ref"] if k in hdr), None)

        if mode_key is None or pads_key is None:
            raise RuntimeError(
                f"Cavity delta CSV missing expected columns.\n"
                f"Found: {hdr}\nNeed mode in [phi_mode, phi_cavity, phi] and pads in [phi_pads, phi_pads_on_this_mesh, phi_ref]."
            )

        stats = _compute_stats(rows, mode_key, pads_key, tau)
        summary.append(("cavity (vs pads)", cav_delta_csv, stats))
    else:
        print(f"[WARN] Missing cavity delta CSV: {cav_delta_csv}")

    # ---- Print ----
    print("\n=== Anderson line errors (E1, E2) ===")
    print(f"root = {root}")
    print(f"tau  = {tau:g}\n")
    print(f"{'CASE':22s} {'N_used':>8s} {'E1=max_rel':>14s} {'E2=L2_rel':>14s}")
    print("-" * 66)
    for case, path, s in summary:
        print(f"{case:22s} {s.n_used:8d} {s.max_rel:14.4e} {s.l2_rel:14.4e}")
    print("")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "csv_path", "n_total", "n_used", "E1_max_rel", "E2_L2_rel", "tau"])
            for case, path, s in summary:
                w.writerow([case, path, s.n_total, s.n_used, s.max_rel, s.l2_rel, tau])
        print(f"[INFO] Wrote summary CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
