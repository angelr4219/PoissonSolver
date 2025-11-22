#!/usr/bin/env python3
import os, sys, math
import numpy as np

# Minimal meshio-only verifier (no dolfinx I/O needed).
# - Loads phi-like point_data from XDMF
# - Builds analytic reference at same nodes
# - Writes nodal & cellwise relative error XDMFs
#
# Usage:
#   python3 verify_simple.py <case>.xdmf --a 3.5e-8 --xs -7e-8,0,7e-8 --Vs 0.25,0.10,0.25 --z 3.5e-8 [--tau 1e-9]

def parse_csv_floats(s):
    return [float(t.strip()) for t in s.split(",") if t.strip()]

def g_uvz(u, v, z):
    return (1.0/(2.0*np.pi))*np.arctan2(u*v, z*np.sqrt(u*u+v*v+z*z))

def phi0_rect(x, y, z, a, xs, Vs):
    # Local arctan reference (minus sign per paper)
    z = np.where(np.abs(z)>0, z, np.finfo(float).eps)
    out = np.zeros_like(x, dtype=float)
    for xi, Vi in zip(xs, Vs):
        out += Vi * (
            g_uvz(a - xi + x, a + y, z) +
            g_uvz(a - xi + x, a - y, z) +
            g_uvz(a + xi - x, a + y, z) +
            g_uvz(a + xi - x, a - y, z)
        )
    return -out

def main(argv):
    import argparse
    p = argparse.ArgumentParser("verify_simple")
    p.add_argument("xdmf", help="XDMF produced by rect_gate_benchmark.py")
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--xs", type=str, required=True, help="comma list, e.g. -7e-8,0,7e-8")
    p.add_argument("--Vs", type=str, required=True, help="comma list, e.g. 0.25,0.10,0.25")
    p.add_argument("--z",  type=float, required=True, help="compare at constant z plane")
    p.add_argument("--tau", type=float, default=1e-9)
    args = p.parse_args(argv)

    xs = parse_csv_floats(args.xs)
    Vs = parse_csv_floats(args.Vs)
    assert len(xs)==len(Vs), "xs and Vs must match length"

    # --- read points/cells + point_data with meshio
    import meshio
    from meshio.xdmf import TimeSeriesReader, TimeSeriesWriter

    with TimeSeriesReader(args.xdmf) as r:
        points, cells = r.read_points_cells()
        data = r.read_data(0)
        # meshio API variants: (pd, cd) or (t, pd, cd)
        if isinstance(data, tuple) and len(data)==3:
            _, point_data, cell_data = data
        else:
            point_data, cell_data = data

    # choose a phi-like point field
    key = None
    for cand in ("phi", "phi (partial)", "phi_top_helper"):
        if cand in point_data:
            key = cand
            break
    if key is None:
        raise SystemExit("No phi-like point_data. Keys: " + ", ".join(point_data.keys()))

    phi_nodes = point_data[key].ravel()
    P = points
    x, y = P[:,0], P[:,1]
    z = np.full_like(x, args.z)

    # analytic reference at the same points
    pref = phi0_rect(x, y, z, args.a, np.array(xs), np.array(Vs))

    # nodal relative error
    denom = np.maximum(np.abs(pref), args.tau)
    rnod = np.abs(phi_nodes - pref)/denom

    # write nodal rel error (point data)
    base = os.path.splitext(args.xdmf)[0]
    out_nodal = f"{base}_rel_error_nodal.xdmf"
    with TimeSeriesWriter(out_nodal) as w:
        w.write_points_cells(points, cells)
        w.write_data(0.0, point_data={"rel_error_nodal": rnod.astype(np.float64)})

    # "cellwise DG0" approximate as mean of nodal rel error over each cell's nodes
    # meshio cells could be mixed; build per-cell average using connectivity
    rcells = []
    for block in cells:
        conn = block.data
        # average over vertices per element
        vals = np.array([rnod[conn_i].mean() for conn_i in conn], dtype=np.float64)
        rcells.append(vals)
    cell_data = {"rel_error_cellwise": rcells}

    out_cell = f"{base}_rel_error_cellwise.xdmf"
    with TimeSeriesWriter(out_cell) as w:
        w.write_points_cells(points, cells)
        w.write_data(0.0, cell_data=cell_data)

    # simple summary
    Linf = float(np.max(rnod)) if rnod.size else 0.0
    L2   = float(np.sqrt(np.sum(rnod*rnod)))
    # combine all cell arrays for stats
    all_cell = np.concatenate(rcells) if rcells else np.zeros(0)
    cmax = float(np.max(all_cell)) if all_cell.size else 0.0
    cmean = float(np.mean(all_cell)) if all_cell.size else 0.0

    summ = f"{base}_summary.csv"
    new  = not os.path.exists(summ)
    import csv
    with open(summ, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["case","a","xs","Vs","tau","REL_CELL_max","REL_CELL_mean","REL_DOF_Linf","REL_DOF_L2"])
        w.writerow([args.xdmf, args.a, args.xs, args.Vs, args.tau,
                    f"{cmax:.6e}", f"{cmean:.6e}", f"{Linf:.6e}", f"{L2:.6e}"])

    print(f"[OK] {args.xdmf}")
    print(f"     nodal: Linf={Linf:.6e}  L2={L2:.6e}")
    print(f"     cell : max={cmax:.6e}  mean={cmean:.6e}")
    print(f"     wrote: {out_nodal}, {out_cell}, {summ}")

if __name__ == "__main__":
    main(sys.argv[1:])
