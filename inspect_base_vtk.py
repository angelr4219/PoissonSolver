#!/usr/bin/env python3

"""
Print exactly what a legacy ASCII VTK RECTILINEAR_GRID file contains,
before comparing it against anything.

This matters because coordinate-convention mismatches between a
reference VTK and a FEniCS run are easy to miss and produce
meaningless comparison numbers. Run this FIRST, every time, on any
new VTK reference file.
"""

import argparse

import numpy as np


def read_vtk_rectilinear(path):
    with open(path) as f:
        lines = f.readlines()
    i = [0]

    def nxt():
        l = lines[i[0]]
        i[0] += 1
        return l

    header_version = nxt().strip()
    title = nxt().strip()
    fmt = nxt().strip()
    dataset_line = nxt().strip()
    nx, ny, nz = (int(v) for v in nxt().split()[1:4])

    def read_coords(n_expected):
        header = nxt().split()
        n = int(header[1])
        assert n == n_expected
        vals = []
        while len(vals) < n:
            vals.extend(float(v) for v in nxt().split())
        return np.array(vals, dtype=np.float64)

    x = read_coords(nx)
    y = read_coords(ny)
    z = read_coords(nz)

    npts = int(nxt().split()[1])  # POINT_DATA n

    fields = {}
    while i[0] < len(lines):
        header = nxt().split()
        if len(header) < 2 or header[0].upper() != "SCALARS":
            continue
        field_name = header[1]
        nxt()  # LOOKUP_TABLE default
        vals = []
        while len(vals) < npts:
            vals.extend(float(v) for v in nxt().split())
        fields[field_name] = np.array(vals, dtype=np.float64).reshape((nz, ny, nx))

    return {
        "header_version": header_version,
        "title": title,
        "format": fmt,
        "dataset": dataset_line,
        "x": x, "y": y, "z": z,
        "fields": fields,
    }


def spacing_summary(coord, name):
    d = np.diff(coord)
    if len(d) == 0:
        return f"  {name}: single point at {coord[0]:.3f}"
    uniq = np.unique(np.round(d, 6))
    if len(uniq) == 1:
        return f"  {name}: {len(coord)} points, [{coord.min():.3f}, {coord.max():.3f}] nm, uniform spacing {uniq[0]:.4f} nm"
    return (f"  {name}: {len(coord)} points, [{coord.min():.3f}, {coord.max():.3f}] nm, "
            f"non-uniform spacing {uniq.min():.4f}-{uniq.max():.4f} nm ({len(uniq)} distinct values)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vtk_path")
    args = p.parse_args()

    d = read_vtk_rectilinear(args.vtk_path)

    print("=" * 78)
    print(f"VTK FILE: {args.vtk_path}")
    print("=" * 78)
    print(f"  format          : {d['header_version']} / {d['format']}")
    print(f"  title           : {d['title']}")
    print(f"  dataset type    : {d['dataset']}")
    print(f"  dimensions      : {len(d['x'])} x {len(d['y'])} x {len(d['z'])} "
          f"= {len(d['x'])*len(d['y'])*len(d['z']):,} points")
    print()
    print("GRID (assumed nm -- verify against the source, VTK has no embedded units):")
    print(spacing_summary(d["x"], "x"))
    print(spacing_summary(d["y"], "y"))
    print(spacing_summary(d["z"], "z"))
    print()
    print(f"COORDINATE CONVENTION NOTE: z runs [{d['z'].min():.1f}, {d['z'].max():.1f}] nm here.")
    print("This repo's FEniCS AFM-tip case uses z<0 = air, z=0 = sample surface,")
    print("z>0 = device (increasing INTO the sample). Confirm whether this VTK's")
    print("z=0 means the same physical thing (sample surface) before comparing --")
    print("do not assume z_FEniCS == z_VTK just because both start at/near 0.")
    print()
    print(f"SCALAR FIELDS ({len(d['fields'])}):")
    for name, arr in d["fields"].items():
        print(f"  {name}: min={arr.min():.6f}  max={arr.max():.6f}  "
              f"mean={arr.mean():.6f}  shape(z,y,x)={arr.shape}")
        # Value at the exact center of the top (z=z.min()) and bottom (z=z.max())
        # planes -- the fastest way to spot a localized gate/tip footprint vs a
        # flat boundary condition, and to read off the bottom BC value directly.
        cy, cx = len(d["y"]) // 2, len(d["x"]) // 2
        print(f"    top plane (z={d['z'][0]:.1f}nm)   center={arr[0, cy, cx]:.6f}  "
              f"corner={arr[0, 0, 0]:.6f}")
        print(f"    bottom plane (z={d['z'][-1]:.1f}nm) center={arr[-1, cy, cx]:.6f}  "
              f"corner={arr[-1, 0, 0]:.6f}")


if __name__ == "__main__":
    main()
