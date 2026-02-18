from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh, io


@dataclass
class WellBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    rho: float


def _parse_well(s: str) -> WellBox:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 7:
        raise ValueError(f'--well expects "xmin,xmax,ymin,ymax,zmin,zmax,rho" (7 values), got: {s}')
    vals = list(map(float, parts))
    return WellBox(*vals)


def _ufl_box_indicator(x, w: WellBox):
    inside_x = ufl.And(ufl.ge(x[0], w.xmin), ufl.le(x[0], w.xmax))
    inside_y = ufl.And(ufl.ge(x[1], w.ymin), ufl.le(x[1], w.ymax))
    inside_z = ufl.And(ufl.ge(x[2], w.zmin), ufl.le(x[2], w.zmax))
    inside = ufl.And(ufl.And(inside_x, inside_y), inside_z)
    return ufl.conditional(inside, 1.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description="3D Poisson on a cube/box with extruded rectangular well sources.")
    parser.add_argument("--Lx", type=float, default=300e-9)
    parser.add_argument("--Ly", type=float, default=300e-9)
    parser.add_argument("--Lz", type=float, default=120e-9)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--eps", type=float, default=11.7, help="Relative permittivity (constant).")
    parser.add_argument("--phi0", type=float, default=0.0, help="Dirichlet phi on outer boundary.")
    parser.add_argument(
        "--well",
        action="append",
        default=[],
        help='Well box: "xmin,xmax,ymin,ymax,zmin,zmax,rho". Can be repeated.',
    )
    parser.add_argument("--out", type=str, default="out_cube", help="Output prefix (XDMF files).")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    # Mesh
    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    p1 = np.array([args.Lx, args.Ly, args.Lz], dtype=np.float64)
    domain = mesh.create_box(
        comm, [p0, p1], [args.nx, args.ny, args.nz],
        cell_type=mesh.CellType.tetrahedron
    )

    # Attach measures to this domain (critical!)
    dx = ufl.Measure("dx", domain=domain)

    # Function spaces
    V = fem.functionspace(domain, ("CG", args.degree))
    V0 = fem.functionspace(domain, ("DG", 0))

    # UFL fields
    x = ufl.SpatialCoordinate(domain)
    wells: List[WellBox] = [_parse_well(s) for s in args.well]

    rho_ufl = PETSc.ScalarType(0.0)
    for w in wells:
        rho_ufl += PETSc.ScalarType(w.rho) * _ufl_box_indicator(x, w)

    eps = fem.Constant(domain, PETSc.ScalarType(args.eps))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
    L = rho_ufl * v * dx

    # Dirichlet BC on entire boundary
    phi0 = fem.Constant(domain, PETSc.ScalarType(args.phi0))

    def on_boundary(xpts: np.ndarray) -> np.ndarray:
        tol = 1e-12
        return (
            np.isclose(xpts[0], 0.0, atol=tol) |
            np.isclose(xpts[0], args.Lx, atol=tol) |
            np.isclose(xpts[1], 0.0, atol=tol) |
            np.isclose(xpts[1], args.Ly, atol=tol) |
            np.isclose(xpts[2], 0.0, atol=tol) |
            np.isclose(xpts[2], args.Lz, atol=tol)
        )

    bdofs = fem.locate_dofs_geometrical(V, on_boundary)
    bc = fem.dirichletbc(phi0, bdofs, V)

    # Solve
    problem = fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
            "ksp_max_it": 5000,
        },
    )
    phi = problem.solve()
    phi.name = "phi"

    # Global min/max
    local_min = np.min(phi.x.array) if phi.x.array.size else np.inf
    local_max = np.max(phi.x.array) if phi.x.array.size else -np.inf
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        print("=== extruded_wells_3d_poisson_cube ===")
        print(f"Domain: Lx={args.Lx:g}, Ly={args.Ly:g}, Lz={args.Lz:g}")
        print(f"Mesh: nx={args.nx}, ny={args.ny}, nz={args.nz} (tet)")
        print(f"CG degree: {args.degree}")
        print(f"eps_r: {args.eps}")
        print(f"BC: phi={args.phi0} on entire boundary")
        if wells:
            print("Wells:")
            for i, w in enumerate(wells):
                print(f"  {i}: [{w.xmin:g},{w.xmax:g}]x[{w.ymin:g},{w.ymax:g}]x[{w.zmin:g},{w.zmax:g}] rho={w.rho:g}")
        else:
            print("Wells: (none) rho=0 everywhere")
        print(f"phi min={gmin:.6e}, phi max={gmax:.6e}")

    # Output rho for viz
    rho_f = fem.Function(V0)
    rho_f.name = "rho"
    rho_f.interpolate(fem.Expression(rho_ufl, V0.element.interpolation_points()))

    phi_path = f"{args.out}_phi.xdmf"
    rho_path = f"{args.out}_rho.xdmf"

    with io.XDMFFile(comm, phi_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)

    with io.XDMFFile(comm, rho_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(rho_f)

    if comm.rank == 0:
        print(f"Wrote: {phi_path}")
        print(f"Wrote: {rho_path}")


if __name__ == "__main__":
    main()
