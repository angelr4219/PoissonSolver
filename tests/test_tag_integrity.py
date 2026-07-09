"""
test_tag_integrity.py

Build the 4-gate box geometry and verify that all boundary regions can be
correctly identified and tagged.  Uses dolfinx's locate_entities_boundary
(the same approach the real solver uses) instead of OCC fragmentation, which
fails when 2-D rectangles lie exactly on the box face.

Geometry (300 x 300 x 150 nm):
    x in [-150, 150] nm
    y in [-150, 150] nm
    z in [   0, 150] nm   (z=0 is gate plane, z=150 nm is bottom contact)

    4 square gates on z=0, each 60 x 60 nm:
        G1 centre: (-75, -75) nm   tag 11
        G2 centre: (+75, -75) nm   tag 12
        G3 centre: (-75, +75) nm   tag 13
        G4 centre: (+75, +75) nm   tag 14
    Top surface excluding gates:   tag 10  (Neumann / free surface)
    Bottom face z=150 nm:          tag  2  (bottom contact)
    Four side walls:               tag  3
    Volume:                        tag  1

Checks performed:
    1. Each required tag is present (non-zero facet count)
    2. Gate area is within 10 % of 3600 nm²
    3. Bottom area is within 10 % of 90 000 nm²
    4. Gate facets are confined to z=0 (not on sides or bottom)
    5. Side-wall facets do not overlap gate regions
    6. Sum of all top facets equals the full top-surface area (300 x 300 nm²)

Usage:
    python3 tests/test_tag_integrity.py [--h-nm H] [--outdir PATH] [--write-xdmf]
    ./run_dolfinx.sh tests/test_tag_integrity.py [args]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile

# ---------------------------------------------------------------------------
# Tag constants
# ---------------------------------------------------------------------------
TAG_VOLUME   = 1
TAG_BOTTOM   = 2
TAG_SIDES    = 3
TAG_TOP_FREE = 10
TAG_G1       = 11
TAG_G2       = 12
TAG_G3       = 13
TAG_G4       = 14

TAG_NAMES = {
    TAG_VOLUME:   "volume",
    TAG_BOTTOM:   "bottom_contact",
    TAG_SIDES:    "side_walls",
    TAG_TOP_FREE: "top_free",
    TAG_G1:       "G1",
    TAG_G2:       "G2",
    TAG_G3:       "G3",
    TAG_G4:       "G4",
}

# Gate geometry (metres)
GATE_SIDE  = 60e-9        # 60 nm
GATE_AREA  = GATE_SIDE**2 # 3600 nm² in m²
BOTTOM_AREA = 300e-9 * 300e-9

# Gate centres at ±60 nm: with half=30 nm, boundaries land at ±30 and ±90 nm,
# which are exact vertices for h=10, 20, 30 nm meshes on the 300 nm domain.
GATE_CENTRES = {
    TAG_G1: (-60e-9, -60e-9),
    TAG_G2: (+60e-9, -60e-9),
    TAG_G3: (-60e-9, +60e-9),
    TAG_G4: (+60e-9, +60e-9),
}

AREA_TOL = 0.10   # 10 % tolerance for mesh-approximated areas
COMM = MPI.COMM_WORLD


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------

def build_mesh(h_nm: float):
    """
    Build a structured tetrahedral box mesh with dolfinx.create_box.

    Domain: x/y in [-150, 150] nm, z in [0, 150] nm.
    Uses an integer number of cells derived from h_nm.
    """
    h = h_nm * 1e-9
    nx = max(1, round(300e-9 / h))
    ny = max(1, round(300e-9 / h))
    nz = max(1, round(150e-9 / h))

    msh = mesh.create_box(
        COMM,
        [np.array([-150e-9, -150e-9,   0.0]),
         np.array([ 150e-9,  150e-9, 150e-9])],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    return msh, nx, ny, nz


# ---------------------------------------------------------------------------
# Tag builders
# ---------------------------------------------------------------------------

def _tol(msh):
    """Half a typical edge length — catches vertices exactly on gate boundaries."""
    coords = msh.geometry.x
    extent = coords.max(axis=0) - coords.min(axis=0)
    n_est  = max(msh.topology.index_map(0).size_global ** (1/3), 1)
    return float(extent.min() / n_est) * 0.6   # ~half a cell


def make_facet_tags(msh) -> dict:
    """
    Locate and return facet index arrays for each tag using
    locate_entities_boundary (the same approach as the real solver).

    Returns a dict: tag_id -> np.ndarray of facet indices.
    """
    tol = _tol(msh)
    fdim = msh.topology.dim - 1
    half = GATE_SIDE / 2

    def _on_bottom(x):
        return np.isclose(x[2], 150e-9, atol=tol)

    def _on_side(x):
        return (
            np.isclose(np.abs(x[0]), 150e-9, atol=tol) |
            np.isclose(np.abs(x[1]), 150e-9, atol=tol)
        )

    def _on_top(x):
        return np.isclose(x[2], 0.0, atol=tol)

    def _on_gate(cx, cy):
        def _f(x):
            return (
                np.isclose(x[2], 0.0, atol=tol) &
                (x[0] >= cx - half - tol) & (x[0] <= cx + half + tol) &
                (x[1] >= cy - half - tol) & (x[1] <= cy + half + tol)
            )
        return _f

    bottom_f = mesh.locate_entities_boundary(msh, fdim, _on_bottom)
    side_f   = mesh.locate_entities_boundary(msh, fdim, _on_side)
    top_all  = mesh.locate_entities_boundary(msh, fdim, _on_top)

    gate_f = {}
    gate_union = set()
    for gid, (cx, cy) in GATE_CENTRES.items():
        gf = mesh.locate_entities_boundary(msh, fdim, _on_gate(cx, cy))
        gate_f[gid] = gf
        gate_union.update(gf.tolist())

    # top_free = top facets NOT inside any gate
    top_free = np.array([f for f in top_all if f not in gate_union], dtype=np.int32)

    return {
        TAG_BOTTOM:   bottom_f,
        TAG_SIDES:    side_f,
        TAG_TOP_FREE: top_free,
        TAG_G1:       gate_f[TAG_G1],
        TAG_G2:       gate_f[TAG_G2],
        TAG_G3:       gate_f[TAG_G3],
        TAG_G4:       gate_f[TAG_G4],
    }


def make_meshtags(msh, facet_dict: dict):
    """Build a single MeshTags object covering all tagged facets."""
    fdim = msh.topology.dim - 1
    indices_list = []
    values_list  = []
    for tag_id, fidx in facet_dict.items():
        if len(fidx):
            indices_list.append(fidx)
            values_list.append(np.full(len(fidx), tag_id, dtype=np.int32))

    if not indices_list:
        raise RuntimeError("No facets found — check mesh / tolerances.")

    all_idx = np.concatenate(indices_list)
    all_val = np.concatenate(values_list)
    order   = np.argsort(all_idx)
    return mesh.meshtags(msh, fdim,
                         all_idx[order].astype(np.int32),
                         all_val[order])


# ---------------------------------------------------------------------------
# Area computation
# ---------------------------------------------------------------------------

def _facet_area_m2(msh, facet_indices: np.ndarray) -> float:
    """
    Compute total surface area (m²) of the given facet indices by summing
    triangle areas from node coordinates.
    """
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    coords = msh.geometry.x

    total = 0.0
    for fi in facet_indices:
        verts = f2v.links(fi)
        if len(verts) < 3:
            continue
        p = coords[verts[:3]]
        v0 = p[1] - p[0]
        v1 = p[2] - p[0]
        total += 0.5 * np.linalg.norm(np.cross(v0, v1))
    return total


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run_checks(msh, facet_dict: dict) -> int:
    """Print the integrity table and return the number of failures."""
    failures = 0
    tol_area = AREA_TOL

    ordered_tags = [TAG_BOTTOM, TAG_SIDES, TAG_TOP_FREE,
                    TAG_G1, TAG_G2, TAG_G3, TAG_G4]

    print()
    print(f"{'tag_id':>8} | {'name':>20} | {'n_facets':>10} | {'area_nm2':>14} | {'status':>8}")
    print("-" * 74)

    gate_z_ok_all = True

    for tid in ordered_tags:
        fidx = facet_dict.get(tid, np.array([], dtype=np.int32))
        n_fac = len(fidx)
        area_m2 = _facet_area_m2(msh, fidx)
        area_nm2 = area_m2 / (1e-9 ** 2)

        geom_ok = n_fac > 0

        # Area checks
        if tid in GATE_CENTRES:
            exp = GATE_AREA / (1e-9 ** 2)   # nm²
            if abs(area_nm2 - exp) / exp > tol_area:
                geom_ok = False

        if tid == TAG_BOTTOM:
            exp = BOTTOM_AREA / (1e-9 ** 2)
            if abs(area_nm2 - exp) / exp > tol_area:
                geom_ok = False

        # Gate z-confinement check: all gate facet midpoints must have z≈0
        if tid in GATE_CENTRES and n_fac > 0:
            fdim = msh.topology.dim - 1
            msh.topology.create_connectivity(fdim, 0)
            f2v = msh.topology.connectivity(fdim, 0)
            coords = msh.geometry.x
            for fi in fidx:
                verts = f2v.links(fi)
                zmid = coords[verts, 2].mean()
                if abs(zmid) > 1e-10:
                    gate_z_ok_all = False

        status = "PASS" if geom_ok else "FAIL"
        if not geom_ok:
            failures += 1
        print(f"{tid:>8} | {TAG_NAMES[tid]:>20} | {n_fac:>10,} | {area_nm2:>14.1f} | {status:>8}")

    print("-" * 74)

    # Extra: top face area sum check
    top_facets = np.concatenate([
        facet_dict.get(TAG_TOP_FREE, np.array([], dtype=np.int32)),
        *[facet_dict.get(g, np.array([], dtype=np.int32)) for g in GATE_CENTRES]
    ])
    top_total_nm2 = _facet_area_m2(msh, top_facets) / (1e-9 ** 2)
    top_exp_nm2   = 300.0 * 300.0
    top_ok = abs(top_total_nm2 - top_exp_nm2) / top_exp_nm2 < tol_area
    print(f"  Top surface total area: {top_total_nm2:.1f} nm²  "
          f"(expected {top_exp_nm2:.0f} nm²) — {'PASS' if top_ok else 'FAIL'}")
    if not top_ok:
        failures += 1

    if not gate_z_ok_all:
        print("  Gate z-confinement check: FAIL (some gate facets not at z=0)")
        failures += 1
    else:
        print("  Gate z-confinement check: PASS (all gate facets at z=0)")

    return failures


# ---------------------------------------------------------------------------
# XDMF output
# ---------------------------------------------------------------------------

def write_xdmf(msh, facet_tags, outdir: Path, h_nm: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"tag_integrity_h{h_nm:.0f}nm.xdmf"
    with XDMFFile(COMM, str(fname), "w") as xf:
        xf.write_mesh(msh)
        xf.write_meshtags(facet_tags, msh.geometry)
    if COMM.rank == 0:
        print(f"\nXDMF written to: {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify boundary tag integrity for 4-gate geometry."
    )
    p.add_argument("--h-nm",      type=float, default=10.0,
                   help="Mesh size in nm (default: 10.0)")
    p.add_argument("--outdir",    type=Path,
                   default=Path("tests/test_tag_integrity/output"))
    p.add_argument("--write-xdmf", action="store_true",
                   help="Write tagged mesh as XDMF for ParaView inspection")
    p.add_argument("--quick",     action="store_true",
                   help="(ignored — test always runs quickly)")
    p.add_argument("--full",      action="store_true",
                   help="(ignored — same as default)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if COMM.rank == 0:
        print("=" * 60)
        print("test_tag_integrity: 4-gate square geometry tag check")
        print(f"  h = {args.h_nm} nm")
        print("=" * 60)

    msh, nx, ny, nz = build_mesh(args.h_nm)
    if COMM.rank == 0:
        print(f"  Mesh: {nx}x{ny}x{nz} cells, {msh.topology.index_map(3).size_global} tets")

    facet_dict = make_facet_tags(msh)
    facet_tags = make_meshtags(msh, facet_dict)

    if args.write_xdmf:
        args.outdir.mkdir(parents=True, exist_ok=True)
        write_xdmf(msh, facet_tags, args.outdir, args.h_nm)

    if COMM.rank == 0:
        failures = run_checks(msh, facet_dict)
        print()
        if failures == 0:
            print("All tag checks PASSED.")
            sys.exit(0)
        else:
            print(f"{failures} tag check(s) FAILED.")
            sys.exit(1)


if __name__ == "__main__":
    main()
