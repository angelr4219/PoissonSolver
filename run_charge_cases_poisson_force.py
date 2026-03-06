#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import traceback
import os

import numpy as np
from mpi4py import MPI

import gmsh
from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc


def gmsh_build_box_with_refine_to_disk_surface(comm, Lx, Ly, Lz, R, z0, n_diam, refine_band_R, far_h_factor, msh_path):
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

    # Helper disk surface
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


def build_fields_disk(msh, R, t, Q, z0, epsr):
    eps0 = 8.8541878128e-12
    Q0 = fem.functionspace(msh, ("DG", 0))

    rho = fem.Function(Q0); rho.name = "rho"
    epsilon = fem.Function(Q0); epsilon.name = "epsilon"
    region_id = fem.Function(Q0); region_id.name = "region_id"

    rho.x.array[:] = 0.0
    epsilon.x.array[:] = eps0 * epsr
    region_id.x.array[:] = 1.0

    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    r2 = x[:, 0]**2 + x[:, 1]**2
    inside = (r2 <= R * R) & (np.abs(x[:, 2] - z0) <= 0.5 * t)

    rho_disk = Q / (math.pi * R * R * t)
    rho.x.array[inside] = rho_disk
    return rho, epsilon, region_id, inside, rho_disk


def write_pre_fields(comm, msh, outdir, rho, epsilon, region_id):
    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / "pre_fields.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(rho)
        xdmf.write_function(epsilon)
        xdmf.write_function(region_id)
    if comm.rank == 0:
        print(f"Wrote: {xdmf_path} (+ .h5) [pre-solve]")


def force_solve_and_write(comm, msh, outdir, p_solve, rho, epsilon, region_id, petsc_monitor):
    Vp = fem.functionspace(msh, ("CG", p_solve))
    u = ufl.TrialFunction(Vp)
    v = ufl.TestFunction(Vp)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.full(X.shape[1], True, dtype=bool))
    bc_dofs = fem.locate_dofs_topological(Vp, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, Vp)

    a = ufl.inner(epsilon * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",      # good default in this container
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
        "ksp_max_it": 20000,
    }
    if petsc_monitor:
        opts["ksp_monitor"] = None
        opts["ksp_converged_reason"] = None

    problem = LinearProblem(a, L, bcs=[bc], petsc_options=opts)
    uh = problem.solve()

    # Output phi as CG1 for XDMF compatibility with linear mesh
    V1 = fem.functionspace(msh, ("CG", 1))
    phi = fem.Function(V1); phi.name = "phi"
    phi.interpolate(uh)

    local_min = float(np.min(phi.x.array))
    local_max = float(np.max(phi.x.array))
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    if comm.rank == 0:
        print(f"phi(min/max) = {gmin:.6e}, {gmax:.6e}")

    xdmf_path = outdir / "mesh_fields.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(epsilon)
        xdmf.write_function(region_id)

    if comm.rank == 0:
        print(f"Wrote: {xdmf_path} (+ .h5) [post-solve]")
        print("DONE")


def run_disk_case(comm, base_out, Lx, Ly, Lz, Rnm, t, Q, z0, epsr, p, n_diam, refine_band_R, far_h_factor, petsc_monitor):
    R = Rnm * 1e-9
    outdir = Path(base_out) / f"disk_R{int(Rnm)}nm"
    msh_path = outdir / "mesh.msh"

    info = gmsh_build_box_with_refine_to_disk_surface(comm, Lx, Ly, Lz, R, z0, n_diam, refine_band_R, far_h_factor, msh_path)

    if comm.rank == 0:
        print("\n=== DISK CASE ===")
        print(f"R = {Rnm} nm, t = {t*1e9:.2f} nm, Q = {Q:.3e} C, p = {p}")
        print(f"h_near = {info['h_near']*1e9:.3f} nm, h_far = {info['h_far']*1e9:.3f} nm")
        print(f"mesh: {msh_path}")

    msh, _, _ = gmshio.read_from_msh(str(msh_path), comm, gdim=3)

    rho, epsilon, region_id, inside, rho_disk = build_fields_disk(msh, R, t, Q, z0, epsr)
    n_local = int(np.count_nonzero(inside))
    n_global = comm.allreduce(n_local, op=MPI.SUM)

    if comm.rank == 0:
        print(f"rho_disk = {rho_disk:.6e} C/m^3, charged cells = {n_global}")

    # Always write pre-solve file
    write_pre_fields(comm, msh, outdir, rho, epsilon, region_id)

    # Now "force" attempt solve.
    # If OS kills the process (OOM), nothing can catch it.
    try:
        force_solve_and_write(comm, msh, outdir, p, rho, epsilon, region_id, petsc_monitor=petsc_monitor)
    except Exception as e:
        if comm.rank == 0:
            (outdir / "FAILED.txt").write_text(
                "Solve failed with Python-level exception:\n\n"
                + repr(e) + "\n\n"
                + traceback.format_exc()
            )
            print("SOLVE FAILED (Python-level). Wrote FAILED.txt")
            print(e)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--Lx", type=float, default=500e-9)
    ap.add_argument("--Ly", type=float, default=500e-9)
    ap.add_argument("--Lz", type=float, default=200e-9)

    ap.add_argument("--disk_radii_nm", type=str, default="10,20,30,50,100")
    ap.add_argument("--t", type=float, default=4e-9)
    ap.add_argument("--z0", type=float, default=0.0)

    ap.add_argument("--Q", type=float, default=1.602176634e-19)
    ap.add_argument("--epsr", type=float, default=12.0)
    ap.add_argument("--p", type=int, default=3)

    ap.add_argument("--n_diam", type=int, default=64)
    ap.add_argument("--refine_band_R", type=float, default=12.0)
    ap.add_argument("--far_h_factor", type=float, default=30.0)

    ap.add_argument("--out", type=str, default="out_mesh_fields")
    ap.add_argument("--petsc_monitor", action="store_true", help="Print KSP monitor and converged reason")

    args = ap.parse_args()
    comm = MPI.COMM_WORLD

    radii = [float(s.strip()) for s in args.disk_radii_nm.split(",") if s.strip()]
    for Rnm in radii:
        run_disk_case(
            comm, args.out, args.Lx, args.Ly, args.Lz,
            Rnm, args.t, args.Q, args.z0, args.epsr, args.p,
            args.n_diam, args.refine_band_R, args.far_h_factor,
            petsc_monitor=args.petsc_monitor
        )


if __name__ == "__main__":
    main()
