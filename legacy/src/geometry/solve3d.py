#!/usr/bin/env python3
from __future__ import annotations
import argparse
from mpi4py import MPI

from geom import Gate2D, Gate3D, build_rect_with_rod_holes_2d, build_cube_with_rod_holes_3d
from materials import eps_constant_DG0  # or eps_from_cell_tags for piecewise
from point_charge import gaussian_rho, solve_with_point_charge

def parse_gate2d(s: str) -> Gate2D:
    cx, cy, wx, wy, V = [float(x) for x in s.split(",")]
    return Gate2D(cx, cy, wx, wy, V)

def parse_gate3d(s: str) -> Gate3D:
    cx, cy, cz, wx, wy, wz, V = [float(x) for x in s.split(",")]
    return Gate3D(cx, cy, cz, wx, wy, wz, V)

def get_args():
    p = argparse.ArgumentParser("Run 2D/3D Poisson with gates + point charge")
    p.add_argument("--dim", choices=["2d", "3d"], required=True)

    # Geometry
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--Lz", type=float, default=1.0)  # used in 3D
    p.add_argument("--h",  type=float, default=0.06)

    # Gates
    p.add_argument("--gate2d", action="append", type=parse_gate2d, default=[],
                   help="2D gate 'cx,cy,wx,wy,V'")
    p.add_argument("--gate3d", action="append", type=parse_gate3d, default=[],
                   help="3D gate 'cx,cy,cz,wx,wy,wz,V'")

    # Permittivity (constant; swap for piecewise via eps_from_cell_tags)
    p.add_argument("--eps_r", type=float, default=11.7)

    # Point charge
    p.add_argument("--Q", type=float, default=0.0)
    p.add_argument("--qpos", type=str, default="0.75,0.25,0.25", help="comma-separated pos")
    p.add_argument("--sigma", type=float, default=0.03)

    # Output
    p.add_argument("--outfile", type=str, default="results/run")
    return p.parse_args()

def main():
    args = get_args()
    comm = MPI.COMM_WORLD

    if args.dim == "2d":
        domain, cell_tags, facet_tags = build_rect_with_rod_holes_2d(
            comm, args.Lx, args.Ly, args.h, args.gate2d)
        eps_dg = eps_constant_DG0(domain, args.eps_r)
        qpos_xy = tuple(float(s) for s in args.qpos.split(",")[:2])
        rho = gaussian_rho(domain, args.Q, qpos_xy, args.sigma)
        gate_tags_V = [(100 + i, g.V) for i, g in enumerate(args.gate2d)]
    else:
        domain, cell_tags, facet_tags = build_cube_with_rod_holes_3d(
            comm, args.Lx, args.Ly, args.Lz, args.h, args.gate3d)
        eps_dg = eps_constant_DG0(domain, args.eps_r)
        qpos_xyz = tuple(float(s) for s in args.qpos.split(",")[:3])
        rho = gaussian_rho(domain, args.Q, qpos_xyz, args.sigma)
        gate_tags_V = [(100 + i, g.V) for i, g in enumerate(args.gate3d)]

    solve_with_point_charge(domain, facet_tags, gate_tags_V, eps_dg, rho, args.outfile)

if __name__ == "__main__":
    main()

