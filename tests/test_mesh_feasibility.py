"""
test_mesh_feasibility.py

Estimate DOFs / cells / memory BEFORE meshing to reject insane requests.
Pure Python — no dolfinx import needed. Fast, runs locally or in Docker.

Domain: Lx=300nm, Ly=300nm, Lz=150nm (device box used in 4-gate tests).

Usage:
    python3 tests/test_mesh_feasibility.py [--max-dofs N] [--save-csv] [--outdir PATH]
    ./run_dolfinx.sh tests/test_mesh_feasibility.py [args]
"""

import argparse
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Domain dimensions (nm)
# ---------------------------------------------------------------------------
Lx_nm = 300.0
Ly_nm = 300.0
Lz_nm = 150.0

# ---------------------------------------------------------------------------
# Candidate mesh sizes
# ---------------------------------------------------------------------------
H_NM_VALUES = [20, 10, 5, 2, 1, 0.5, 0.1, 0.001]


def compute_row(h_nm: float, max_dofs: int) -> dict:
    nx = math.ceil(Lx_nm / h_nm)
    ny = math.ceil(Ly_nm / h_nm)
    nz = math.ceil(Lz_nm / h_nm)

    approx_hex_cells = nx * ny * nz
    approx_tet_cells = approx_hex_cells * 6
    approx_P1_dofs   = (nx + 1) * (ny + 1) * (nz + 1)
    approx_P2_dofs   = approx_P1_dofs * 7   # rough factor
    mem_GB_P1        = approx_P1_dofs * 8 / 1e9  # float64

    if approx_P1_dofs < 500_000:
        status = "OK"
    elif approx_P1_dofs < max_dofs:
        status = "aggressive"
    else:
        status = "reject"

    return {
        "h_nm":          h_nm,
        "nx":            nx,
        "ny":            ny,
        "nz":            nz,
        "hex_cells":     approx_hex_cells,
        "tet_cells":     approx_tet_cells,
        "P1_dofs":       approx_P1_dofs,
        "P2_dofs":       approx_P2_dofs,
        "mem_GB":        mem_GB_P1,
        "status":        status,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'h_nm':>8} | {'nx':>7} | {'ny':>7} | {'nz':>7} | "
        f"{'hex_cells':>14} | {'tet_cells':>14} | "
        f"{'P1_dofs':>12} | {'P2_dofs':>12} | "
        f"{'mem_GB':>10} | {'status':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['h_nm']:>8.4g} | {r['nx']:>7,} | {r['ny']:>7,} | {r['nz']:>7,} | "
            f"{r['hex_cells']:>14,} | {r['tet_cells']:>14,} | "
            f"{r['P1_dofs']:>12,} | {r['P2_dofs']:>12,} | "
            f"{r['mem_GB']:>10.4f} | {r['status']:>10}"
        )
    print(sep)


def save_csv(rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "mesh_feasibility.csv"
    import csv
    fieldnames = ["h_nm", "nx", "ny", "nz", "hex_cells", "tet_cells",
                  "P1_dofs", "P2_dofs", "mem_GB", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to: {csv_path}")


def print_recommendation(rows: list[dict]) -> None:
    safe       = [r["h_nm"] for r in rows if r["status"] == "OK"]
    aggressive = [r["h_nm"] for r in rows if r["status"] == "aggressive"]
    reject     = [r["h_nm"] for r in rows if r["status"] == "reject"]

    def fmt_list(lst):
        return ", ".join(f"{v}nm" for v in lst) if lst else "(none)"

    print()
    print("=" * 60)
    print("RECOMMENDATION SUMMARY")
    print("=" * 60)
    print(f"  SAFE for laptop (P1 DOFs < 500k): {fmt_list(safe)}")
    print(f"  AGGRESSIVE (needs workstation):    {fmt_list(aggressive)}")
    print(f"  REJECT (do not attempt uniform):   {fmt_list(reject)}")
    print("=" * 60)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate mesh feasibility without building a mesh."
    )
    parser.add_argument(
        "--max-dofs", type=int, default=5_000_000,
        metavar="INT",
        help="P1 DOF threshold above which status becomes 'reject' (default: 5,000,000)"
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Write results to <outdir>/mesh_feasibility.csv"
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("tests/test_mesh_feasibility/output"),
        metavar="PATH",
        help="Output directory for CSV (default: tests/test_mesh_feasibility/output)"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print()
    print(f"Mesh Feasibility Estimator")
    print(f"  Domain: Lx={Lx_nm}nm  Ly={Ly_nm}nm  Lz={Lz_nm}nm")
    print(f"  max-dofs threshold: {args.max_dofs:,}")
    print()

    rows = [compute_row(h, args.max_dofs) for h in H_NM_VALUES]
    print_table(rows)
    print_recommendation(rows)

    if args.save_csv:
        save_csv(rows, args.outdir)


if __name__ == "__main__":
    main()
