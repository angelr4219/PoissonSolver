#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import traceback

import numpy as np
from mpi4py import MPI

import gmsh
from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc


def gmsh_build_box_with_refine_to_disk_surface(
    comm: MPI.Comm,
    Lx: float, Ly: float, Lz: float,
    R: float,
    z0: float,
    n_diam: int,
    refine_band_R: float,
    far_h_factor: float,
    msh_path: Path,
):
    if comm.rank != 0:
        return None

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("box_refine_disk")
    occ = gmsh.model.occ

    x0 = -0.5 * Lx
    y0 = -0.5 * Ly
    zmin = -0.5 * Lz
    box = occ.addBox(x0, y0, zmin, Lx, Ly, Lz)

    # Helper disk surface for size field
    c = occ.addCircle(0.0, 0.0, z0, R)
    cl = occ.addCurveLoop([c])
    disk_surf = occ.addPlaneSurface([cl])

    occ.synchronize()

    pg_bulk = 1
    gmsh.model.addPhysicalGroup(3, [box], pg_bulk)
    gmsh.model.setPhysicalName(3, pg_bulk, "bulk")

    h_near = (2.0 * R) / float(n_diam)
    h_far = far_h_factor * h_near

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", [disk_surf])

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h_near)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", refine_band_R * R)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h_near)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)

    gmsh.model.mesh.generate(3)
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    return dict(h_near=h_near, h_far=h_far)


def build_common_cellwise_fields(msh, epsr: float):
    eps0 = 8.8541878128e-12
    Q0 = fem.functionspace(msh, ("DG", 0))

    rho = fem.Function(Q0); rho.name = "rho"
    epsilon = fem.Function(Q0); epsilon.name = "epsilon"
    region_id = fem.Function(Q0); region_id.name = "region_id"

    rho.x.array[:] = 0.0
    epsilon.x.array[:] = eps0 * epsr
    region_id.x.array[:] = 1.0
    return rho, epsilon, region_id, Q0


def stamp_disk_charge(rho, Q0, R: float, t: float, Q: float, z0: float):
    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    r2 = x[:, 0]**2 + x[:, 1]**2
    inside = (r2 <= R * R) & (np.abs(x[:, 2] - z0) <= 0.5 * t)
    rho_disk = Q / (math.pi * R * R * t)
    rho.x.array[inside] = rho_disk
    return inside, rho_disk


def write_pre_fields(comm, msh, outdir: Path, rho, epsilon, region_id):
    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / "pre_fields.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(rho)
        xdmf.write_function(epsilon)
        xdmf.write_function(region_id)
    if comm.rank == 0:
        print(f"Wrote: {xdmf_path} (and .h5 sidecar)  [pre-solve]")


def solve_and_write_mesh_fields(comm, msh, outdir: Path, p_solve: int, rho, epsilon, region_id):
    Vp = fem.functionspace(msh, ("CG", p_solve))
    u = ufl.TrialFunction(Vp)
    v = ufl.TestFunction(Vp)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.full(X.shape[1], True, dtype=bool))
    bc_dofs = fem.locate_dofs_topological(Vp, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, Vp)

    a = ufl.inner(epsilon * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 4000,
            # Helpful if PETSc is silently dying:
            # "ksp_monitor": None,
            # "ksp_converged_reason": None,
        },
    )
    uh = problem.solve()

    # Output phi as CG1 for your dolfinx XDMF writer constraint
    V1 = fem.functionspace(msh, ("CG", 1))
    phi = fem.Function(V1); phi.name = "phi"
    phi.interpolate(uh)

    local_min = float(np.min(phi.x.array))
    local_max = float(np.max(phi.x.array))
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    if comm.rank == 0:
        print(f"phi(min/max) = {gmin:.6e}, {gmax:.6e}")

    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / "mesh_fields.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(epsilon)
        xdmf.write_function(region_id)

    if comm.rank == 0:
        print(f"Wrote: {xdmf_path} (and .h5 sidecar)  [post-solve]")


def run_one_disk(comm, base_out: Path, Lx, Ly, Lz, Rnm, t, Q, z0, epsr, p, n_diam, refine_band_R, far_h_factor):
    R = Rnm * 1e-9
    outdir = base_out / f"disk_R{int(Rnm)}nm"
    msh_path = outdir / "mesh.msh"

    info = gmsh_build_box_with_refine_to_disk_surface(
        comm, Lx, Ly, Lz, R, z0, n_diam, refine_band_R, far_h_factor, msh_path
    )

    if comm.rank == 0:
        print("\n=== DISK CASE ===")
        print(f"R = {Rnm} nm, t = {t*1e9:.2f} nm, Q = {Q:.3e} C, p = {p}")
        print(f"h_near = {info['h_near']*1e9:.3f} nm, h_far = {info['h_far']*1e9:.3f} nm")
        print(f"mesh: {msh_path}")

    msh, _, _ = gmshio.read_from_msh(str(msh_path), comm, gdim=3)

    rho, epsilon, region_id, Q0 = build_common_cellwise_fields(msh, epsr)
    inside, rho_disk = stamp_disk_charge(rho, Q0, R, t, Q, z0)

    n_local = int(np.count_nonzero(inside))
    n_global = comm.allreduce(n_local, op=MPI.SUM)
    if comm.rank == 0:
        print(f"rho_disk = {rho_disk:.6e} C/m^3, charged cells = {n_global}")

    # Always write pre-solve fields so you can verify output even if solve dies
    write_pre_fields(comm, msh, outdir, rho, epsilon, region_id)

    # Solve and write full fields
    try:
        solve_and_write_mesh_fields(comm, msh, outdir, p, rho, epsilon, region_id)
        if comm.rank == 0:
            print("DONE (solve+write)")
    except Exception as e:
        if comm.rank == 0:
            print("SOLVE FAILED WITH EXCEPTION:")
            print(e)
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--Lx", type=float, default=500e-9)
    ap.add_argument("--Ly", type=float, default=500e-9)
    ap.add_argument("--Lz", type=float, default=200e-9)

    ap.add_argument("--epsr", type=float, default=12.0)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--Q", type=float, default=1.602176634e-19)

    ap.add_argument("--out", type=str, default="out_mesh_fields")

    ap.add_argument("--disk_radii_nm", type=str, default="10,20,30,50,100")
    ap.add_argument("--t", type=float, default=4e-9)
    ap.add_argument("--z0", type=float, default=0.0)

    ap.add_argument("--n_diam", type=int, default=32)
    ap.add_argument("--refine_band_R", type=float, default=8.0)
    ap.add_argument("--far_h_factor", type=float, default=25.0)

    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    base_out = Path(args.out)

    radii = [float(s.strip()) for s in args.disk_radii_nm.split(",") if s.strip()]
    for Rnm in radii:
        run_one_disk(comm, base_out, args.Lx, args.Ly, args.Lz,
                     Rnm, args.t, args.Q, args.z0, args.epsr, args.p,
                     args.n_diam, args.refine_band_R, args.far_h_factor)


if __name__ == "__main__":
    main()
