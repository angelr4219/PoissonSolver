#!/usr/bin/env python3
"""
scripts/stage_june26.py
-----------------------
Organise VTK reference files and DOLFINx XDMF/H5 outputs into a single
comparison campaign folder ("june 26" by default) so that the comparison
notebook always knows where to look.

Layout produced
---------------
<root>/
  vtk/          VTK reference file(s)
  fem/          XDMF + H5 pairs (subdirectory structure from source preserved)
  notebooks/    comparison notebook(s)
  outputs/      comparison attempt outputs  (created on demand by notebook)
  notes/        manifest JSON and README

Usage examples
--------------
  # Copy the standard VTK reference + scan benchmark_results and disk_Q1e_p3_dense:
  python3 scripts/stage_june26.py

  # Explicit extra VTK and FEM files:
  python3 scripts/stage_june26.py --vtk path/to/extra.vtk --fem path/to/case.xdmf path/to/case.h5

  # Move instead of copy:
  python3 scripts/stage_june26.py --move
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Default file locations
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

DEFAULT_VTK = [REPO_ROOT / "basePotential3d.vtk"]

# Directories to scan for XDMF/H5 pairs automatically.
SCAN_DIRS = [
    REPO_ROOT / "benchmark_results",
    REPO_ROOT / "disk_Q1e_p3_dense",
]

# Files / patterns to skip when auto-discovering.
SKIP_PATTERNS = ["__pycache__", ".git", "output", ".DS_Store"]

XDMF_SUFFIXES = {".xdmf", ".xmdf"}
H5_SUFFIXES   = {".h5", ".hdf5"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_skip(path: Path) -> bool:
    for part in path.parts:
        for pat in SKIP_PATTERNS:
            if pat in part:
                return True
    return False


def _safe_copy(src: Path, dst: Path, move: bool) -> Path | None:
    """Copy or move src → dst.

    Returns None if src is missing (prints a warning) or if dst already exists
    with the same size (idempotent re-run).  If dst exists with a different
    size, a _dupN name is used.
    """
    if not src.exists():
        print(f"[MISSING] {src} — skipping")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return None   # already staged, nothing to do
        stem, suf = dst.stem, dst.suffix
        k = 1
        while True:
            candidate = dst.with_name(f"{stem}_dup{k}{suf}")
            if not candidate.exists():
                dst = candidate
                break
            k += 1
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)
    return dst


def _discover_xdmf_h5(scan_dirs: list[Path]) -> list[tuple[Path, Path | None]]:
    """Return list of (xdmf_path, h5_path_or_None) from scan_dirs."""
    pairs: list[tuple[Path, Path | None]] = []
    seen: set[Path] = set()

    for base in scan_dirs:
        if not base.is_dir():
            continue
        for xdmf in sorted(base.rglob("*.xdmf")) + sorted(base.rglob("*.xmdf")):
            if _should_skip(xdmf):
                continue
            key = xdmf.resolve()
            if key in seen:
                continue
            seen.add(key)
            # Find matching H5 (same stem, same dir)
            h5 = None
            for suf in [".h5", ".hdf5"]:
                candidate = xdmf.with_suffix(suf)
                if candidate.exists():
                    h5 = candidate
                    break
            pairs.append((xdmf, h5))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage VTK and FEM files for the june 26 comparison campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", default=str(REPO_ROOT / "june 26"),
                    help="Staging root folder")
    ap.add_argument("--move", action="store_true",
                    help="Move files instead of copying them")
    ap.add_argument("--vtk", nargs="*", metavar="FILE",
                    help="Extra VTK files to include (default VTK is always copied)")
    ap.add_argument("--fem", action="append", default=[], metavar="FILE",
                    help="Explicit XDMF or H5 file to stage; repeat for multiple files")
    ap.add_argument("--no-auto-scan", action="store_true",
                    help="Disable automatic scan of benchmark_results and disk_Q1e_p3_dense")
    args = ap.parse_args()

    root   = Path(args.root)
    vtk_d  = root / "vtk"
    fem_d  = root / "fem"
    nb_d   = root / "notebooks"
    out_d  = root / "outputs"
    notes_d = root / "notes"

    for d in [vtk_d, fem_d, nb_d, out_d, notes_d]:
        d.mkdir(parents=True, exist_ok=True)

    op = "MOVED" if args.move else "COPIED"
    staged: list[dict] = []

    # ── VTK files ────────────────────────────────────────────────────────
    vtk_sources = list(DEFAULT_VTK)
    if args.vtk:
        vtk_sources += [Path(p) for p in args.vtk]

    for src in vtk_sources:
        if not src.exists():
            print(f"[SKIP] VTK not found: {src}")
            continue
        dst = _safe_copy(src, vtk_d / src.name, args.move)
        if dst is None:
            print(f"[SKIP already staged] VTK: {src.name}")
            continue
        print(f"{op} VTK: {src.relative_to(REPO_ROOT)} → {dst.relative_to(root)}")
        staged.append({"kind": "vtk", "src": str(src), "dst": str(dst)})

    # ── Auto-discover XDMF/H5 ────────────────────────────────────────────
    auto_pairs: list[tuple[Path, Path | None]] = []
    if not args.no_auto_scan:
        auto_pairs = _discover_xdmf_h5(SCAN_DIRS)

    # ── Explicit FEM files ────────────────────────────────────────────────
    explicit_fem: list[Path] = [Path(p) for p in (args.fem or [])]

    # Pair explicit files by stem.
    xdmf_explicit: dict[str, Path] = {}
    h5_explicit:   dict[str, Path] = {}
    for p in explicit_fem:
        if p.suffix in XDMF_SUFFIXES:
            xdmf_explicit[p.stem] = p
        elif p.suffix in H5_SUFFIXES:
            h5_explicit[p.stem] = p

    extra_pairs: list[tuple[Path, Path | None]] = []
    for stem, xdmf in xdmf_explicit.items():
        extra_pairs.append((xdmf, h5_explicit.get(stem)))
    for stem, h5 in h5_explicit.items():
        if stem not in xdmf_explicit:
            extra_pairs.append((h5, None))

    all_pairs = auto_pairs + extra_pairs
    seen_xdmf: set[Path] = set()

    for xdmf, h5 in all_pairs:
        key = xdmf.resolve()
        if key in seen_xdmf:
            continue
        seen_xdmf.add(key)

        # Preserve subdirectory relative to REPO_ROOT so filenames don't collide.
        try:
            rel_dir = xdmf.parent.relative_to(REPO_ROOT)
        except ValueError:
            rel_dir = Path(xdmf.parent.name)

        dst_xdmf = _safe_copy(xdmf, fem_d / rel_dir / xdmf.name, args.move)
        if dst_xdmf is None:
            print(f"[SKIP already staged] {xdmf.name}")
        else:
            print(f"{op} XDMF: {xdmf.name} → {dst_xdmf.relative_to(root)}")
            staged.append({"kind": "xdmf", "src": str(xdmf), "dst": str(dst_xdmf)})

        if h5 and h5.exists():
            dst_h5 = _safe_copy(h5, fem_d / rel_dir / h5.name, args.move)
            if dst_h5 is None:
                print(f"[SKIP already staged] {h5.name}")
            else:
                print(f"{op} H5:   {h5.name} → {dst_h5.relative_to(root)}")
                staged.append({"kind": "h5", "src": str(h5), "dst": str(dst_h5)})
        else:
            if h5:
                print(f"[WARN] H5 not found for {xdmf.name}: {h5}")

    # ── Manifest + README ─────────────────────────────────────────────────
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root),
        "moved": args.move,
        "n_staged": len(staged),
        "files": staged,
    }
    manifest_path = notes_d / "stage_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    readme = notes_d / "README_june26.md"
    if readme.exists():
        print(f"[SKIP] README already exists: {readme}")
    else:
        readme.write_text(
        "# June 26 MaSQE vs DOLFINx comparison\n\n"
        "## Folder layout\n\n"
        "| Folder | Contents |\n"
        "|--------|----------|\n"
        "| `vtk/` | MaSQE reference VTK potential(s) |\n"
        "| `fem/` | DOLFINx XDMF + H5 outputs |\n"
        "| `notebooks/` | Comparison notebook(s) |\n"
        "| `outputs/` | Per-attempt comparison outputs |\n"
        "| `notes/` | Manifests and run notes |\n\n"
        "## Primary comparison\n\n"
        "Reference: `vtk/basePotential3d.vtk`\n"
        "- Grid: 51 × 51 × 111 (rectilinear)\n"
        "- x: [−250, +250] nm, y: [−250, +250] nm, z: [0, 255] nm\n"
        "- Scalar: `basePotential`, range [−12, −1] V\n\n"
        "## Unit convention\n\n"
        "VTK coordinates are in **nm**.\n"
        "DOLFINx mesh coordinates are in **metres** (multiply by 1e9 to get nm).\n\n"
        "## ⚠ Do not compare basePotential3d.vtk against the staged FEM benchmarks\n\n"
        "`fem/benchmark_results/` and `fem/disk_Q1e_p3_dense/` are staged for solver\n"
        "verification only.  They solve **different physics** from the MaSQE reference:\n\n"
        "| Folder | Problem | phi range | Domain |\n"
        "|--------|---------|-----------|--------|\n"
        "| `benchmark_results/` | Point-charge in vacuum | 0 – 0.16 V | ±100 nm cube |\n"
        "| `disk_Q1e_p3_dense/` | Charged disk | ~mV | ±300 nm × ±200 nm |\n"
        "| **MaSQE reference** | **Device bias, layered stack** | **−12 to −1 V** | **±250 nm × 0–255 nm** |\n\n"
        "A comparison plot between these will be mathematically valid but physically\n"
        "meaningless.  Pretty plots from mismatched physics are worse than no plots\n"
        "because they look convincing.\n\n"
        "## Matching device-scale FEM case (target)\n\n"
        "The valid MaSQE comparison requires a DOLFINx solve with:\n\n"
        "- Domain: 500 nm × 500 nm × 255 nm (x: −250 to +250, y: −250 to +250, z: 0 to 255)\n"
        "- Top/gate surface (z = 0): φ = −1 V (Dirichlet)\n"
        "- Bottom contact (z = 255 nm): φ = −12 V (Dirichlet)\n"
        "- Charge density: ρ = 0\n"
        "- Same layered dielectric stack and ε values as MaSQE\n"
        "- Coordinate convention: nm in DOLFINx mesh, stored as metres in H5\n\n"
        "Planned refinement ladder (add one at a time; compare before moving on):\n\n"
        "| Case | h [nm] | Notes |\n"
        "|------|--------|-------|\n"
        "| A    | 20     | Diagnostic — establishes shape, not accuracy |\n"
        "| B    | 10     | Coarse reference |\n"
        "| C    |  5     | First serious comparison |\n"
        "| D    |  3     | Better |\n"
        "| E    |  2     | Aggressive (uniform); use local refinement below this |\n\n"
        "Once a case XDMF/H5 pair is generated, stage it with:\n\n"
        "```bash\n"
        "python3 scripts/stage_june26.py --fem path/to/case.xdmf path/to/case.h5\n"
        "```\n\n"
        "Then add only that case to `CASES` in the notebook before running the comparison.\n",
        encoding="utf-8",
    )

    print()
    print(f"Staged {len(staged)} file(s) under: {root}")
    print(f"Manifest: {manifest_path}")
    print(f"README:   {readme}")


if __name__ == "__main__":
    main()
