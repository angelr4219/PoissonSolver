from __future__ import annotations

import argparse
from pathlib import Path

from dolfinx import fem

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
    write_phi_file,
    write_cell_fields_file,
    print_written_files,
)


def nm(x_nm: float) -> float:
    return 1.0e-9 * x_nm


def parse_args():
    p = argparse.ArgumentParser(
        description="Run embedded-charge Poisson solves for sphere, disk, or cube."
    )

    p.add_argument("--shape", choices=["sphere", "disk", "cube"], required=True)
    p.add_argument("--outdir", type=str, required=True)

    p.add_argument("--Lx_nm", type=float, default=400.0)
    p.add_argument("--Ly_nm", type=float, default=400.0)
    p.add_argument("--Lz_nm", type=float, default=400.0)
    p.add_argument("--h_nm", type=float, default=10.0)

    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--eps_r", type=float, default=11.7)
    p.add_argument("--periodic", choices=["none", "x", "y", "xy"], default="none")

    p.add_argument("--xc_nm", type=float, default=0.0)
    p.add_argument("--yc_nm", type=float, default=0.0)
    p.add_argument("--zc_nm", type=float, default=0.0)

    p.add_argument("--R_nm", type=float, default=30.0)
    p.add_argument("--t_nm", type=float, default=6.0)
    p.add_argument("--a_nm", type=float, default=60.0)

    p.add_argument("--Q_e", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)

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

    else:
        spec = CubeSpec(
            xc=nm(args.xc_nm),
            yc=nm(args.yc_nm),
            zc=nm(args.zc_nm),
            a=nm(args.a_nm),
        )
        rho, epsilon, region_id, inside, q_assigned = build_cube_problem(
            msh, spec=spec, eps_r=args.eps_r, Q_total=Q_total
        )

    Q0 = rho.function_space
    shape_mask = fem.Function(Q0, name="shape_mask")
    shape_mask.x.array[:] = 0.0
    shape_mask.x.array[inside] = 1.0

    if RANK == 0:
        print("")
        print("=== SHAPE RUN ===")
        print(f"shape          : {args.shape}")
        print(f"Q target [C]   : {Q_total:.12e}")
        print(f"Q assigned [C] : {q_assigned:.12e}")
        print(f"mesh           : {nx} x {ny} x {nz}")
        print(f"eps_r          : {args.eps_r}")
        print(f"periodic       : {args.periodic}")
        print(f"outdir         : {outdir}")
        print("")

    phi, _mpc = solve_poisson_box(
        msh,
        Lx=box.Lx,
        Ly=box.Ly,
        Lz=box.Lz,
        epsilon=epsilon,
        rho=rho,
        degree=args.deg,
        periodic=args.periodic,
    )

    xdmf_phi = write_phi_file(
        msh=msh,
        phi=phi,
        outdir=outdir,
        basename="phi_solution",
    )

    xdmf_aux = write_cell_fields_file(
        msh=msh,
        outdir=outdir,
        fields=[rho, epsilon, region_id, shape_mask],
        basename="cell_fields",
    )

    print_written_files(xdmf_phi, xdmf_aux)


if __name__ == "__main__":
    main()
