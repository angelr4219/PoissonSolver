from __future__ import annotations

import argparse
from pathlib import Path

from poisson_solver import (
    COMM,
    RANK,
    E_CHARGE,
    BoxSpec,
    SphereSpec,
    DiskSpec,
    CubeSpec,
    build_centered_box_mesh,
    build_sphere_problem,
    build_disk_problem,
    build_cube_problem,
    solve_poisson_box,
    write_pre_fields,
    write_mesh_and_functions,
    function_for_xdmf,
)


def nm(x_nm: float) -> float:
    return 1.0e-9 * x_nm


def parse_args():
    p = argparse.ArgumentParser(
        description="Run embedded-charge Poisson solves for sphere, disk, or cube."
    )

    p.add_argument("--shape", choices=["sphere", "disk", "cube"], required=True)
    p.add_argument("--outdir", type=str, default="Results/shape_run")

    p.add_argument("--Lx_nm", type=float, default=400.0)
    p.add_argument("--Ly_nm", type=float, default=400.0)
    p.add_argument("--Lz_nm", type=float, default=400.0)
    p.add_argument("--h_nm", type=float, default=10.0)

    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--eps_r", type=float, default=11.7)

    p.add_argument("--periodic", choices=["none", "x", "y"], default="none")

    # Common center
    p.add_argument("--xc_nm", type=float, default=0.0)
    p.add_argument("--yc_nm", type=float, default=0.0)
    p.add_argument("--zc_nm", type=float, default=0.0)

    # Sphere / disk
    p.add_argument("--R_nm", type=float, default=30.0)

    # Disk thickness
    p.add_argument("--t_nm", type=float, default=6.0)

    # Cube side
    p.add_argument("--a_nm", type=float, default=60.0)

    # Charge in units of electron charge
    p.add_argument("--Q_e", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    box = BoxSpec(
        Lx=nm(args.Lx_nm),
        Ly=nm(args.Ly_nm),
        Lz=nm(args.Lz_nm),
        h=nm(args.h_nm),
    )
    msh, nx, ny, nz = build_centered_box_mesh(COMM, box)

    Q_total = float(args.Q_e) * E_CHARGE

    if args.shape == "sphere":
        spec = SphereSpec(
            xc=nm(args.xc_nm),
            yc=nm(args.yc_nm),
            zc=nm(args.zc_nm),
            R=nm(args.R_nm),
        )
        rho, epsilon, region_id, inside, q_assigned = build_sphere_problem(
            msh, spec=spec, eps_r=args.eps_r, Q_total=Q_total
        )

    elif args.shape == "disk":
        spec = DiskSpec(
            xc=nm(args.xc_nm),
            yc=nm(args.yc_nm),
            zc=nm(args.zc_nm),
            R=nm(args.R_nm),
            t=nm(args.t_nm),
        )
        rho, epsilon, region_id, inside, q_assigned = build_disk_problem(
            msh, spec=spec, eps_r=args.eps_r, Q_total=Q_total
        )

    elif args.shape == "cube":
        spec = CubeSpec(
            xc=nm(args.xc_nm),
            yc=nm(args.yc_nm),
            zc=nm(args.zc_nm),
            a=nm(args.a_nm),
        )
        rho, epsilon, region_id, inside, q_assigned = build_cube_problem(
            msh, spec=spec, eps_r=args.eps_r, Q_total=Q_total
        )

    else:
        raise ValueError(f"Unsupported shape {args.shape}")

    if RANK == 0:
        print("")
        print("=== SHAPE RUN ===")
        print(f"shape          : {args.shape}")
        print(f"Q target [C]   : {Q_total:.12e}")
        print(f"Q assigned [C] : {q_assigned:.12e}")
        print(f"mesh           : {nx} x {ny} x {nz}")
        print(f"eps_r          : {args.eps_r}")
        print(f"periodic       : {args.periodic}")
        print("")

    write_pre_fields(COMM, msh, outdir, rho, epsilon, region_id)

    phi_h, _mpc = solve_poisson_box(
        msh,
        Lx=box.Lx,
        Ly=box.Ly,
        Lz=box.Lz,
        epsilon=epsilon,
        rho=rho,
        degree=args.deg,
        periodic=args.periodic,
    )

    phi_out = function_for_xdmf(phi_h, msh)
    write_mesh_and_functions(COMM, msh, outdir / "solution.xdmf", [phi_out])


if __name__ == "__main__":
    main()
