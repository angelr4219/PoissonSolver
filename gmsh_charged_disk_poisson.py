#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from mpi4py import MPI

import gmsh
from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc


def gmsh_build_box_with_refinement_field(
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
    gmsh.model.add("charged_disk_box")
    occ = gmsh.model.occ

    # Box centered at origin
    x0 = -0.5 * Lx
    y0 = -0.5 * Ly
    zmin = -0.5 * Lz
    box = occ.addBox(x0, y0, zmin, Lx, Ly, Lz)

    # Helper disk surface for sizing field (no boolean ops)
    c = occ.addCircle(0.0, 0.0, z0, R)
    cl = occ.addCurveLoop([c])
    disk_surf = occ.addPlaneSurface([cl])

    occ.synchronize()

    # Physical group for volume (stable id = 1)
    pg_bulk = 1
    gmsh.model.addPhysicalGroup(3, [box], pg_bulk)
    gmsh.model.setPhysicalName(3, pg_bulk, "bulk")

    # Mesh sizing
    h_disk = (2.0 * R) / float(n_diam)
    h_far = far_h_factor * h_disk

    # Distance-based size field
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", [disk_surf])

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h_disk)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", refine_band_R * R)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h_disk)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)

    gmsh.model.mesh.generate(3)

    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    return dict(h_disk=h_disk, h_far=h_far)


def build_cellwise_fields(
    msh,
    R: float, t: float, Q: float,
    z0: float,
    epsr_bulk: float,
):
    """
    Create DG0 (cellwise) fields:
      - rho (C/m^3): disk region gets rho_disk, else 0
      - epsilon (F/m): bulk epsilon everywhere for now
      - region_id (int-like): 1 everywhere for now
    """
    eps0 = 8.8541878128e-12
    eps_bulk = eps0 * epsr_bulk

    Q0 = fem.functionspace(msh, ("DG", 0))

    rho = fem.Function(Q0)
    rho.name = "rho"
    rho.x.array[:] = 0.0

    epsilon = fem.Function(Q0)
    epsilon.name = "epsilon"
    epsilon.x.array[:] = eps_bulk

    region_id = fem.Function(Q0)
    region_id.name = "region_id"
    region_id.x.array[:] = 1.0  # store as float for XDMF; ParaView still treats it fine

    # Disk indicator using DG0 dof coords (cell midpoints)
    x = Q0.tabulate_dof_coordinates().reshape((-1, 3))
    r2 = x[:, 0]**2 + x[:, 1]**2
    inside = (r2 <= R * R) & (np.abs(x[:, 2] - z0) <= 0.5 * t)

    rho_disk = Q / (math.pi * R * R * t)
    rho.x.array[inside] = rho_disk

    return rho, epsilon, region_id, inside, rho_disk


def solve_and_write_all_fields(
    comm: MPI.Comm,
    msh_path: Path,
    outdir: Path,
    R: float, t: float, Q: float,
    z0: float,
    epsr_bulk: float,
    p_solve: int,
    write_gate_scaffold: bool,
):
    msh, cell_tags, facet_tags = gmshio.read_from_msh(str(msh_path), comm, gdim=3)

    # Cellwise fields
    rho, epsilon, region_id, inside, rho_disk = build_cellwise_fields(
        msh=msh, R=R, t=t, Q=Q, z0=z0, epsr_bulk=epsr_bulk
    )

    # Solve in degree p_solve
    Vp = fem.functionspace(msh, ("CG", p_solve))
    u = ufl.TrialFunction(Vp)
    v = ufl.TestFunction(Vp)

    # Outer boundary phi=0
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.full(X.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(Vp, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, Vp)

    # Use epsilon field in the weak form (cellwise DG0 is OK)
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
        },
    )
    uh = problem.solve()
    uh.name = "phi_p"

    # Diagnostics
    local_min = float(np.min(uh.x.array))
    local_max = float(np.max(uh.x.array))
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    n_local = int(np.count_nonzero(inside))
    n_global = comm.allreduce(n_local, op=MPI.SUM)

    if comm.rank == 0:
        print("=== Solve summary ===")
        print(f"p_solve = {p_solve}")
        print(f"rho_disk = {rho_disk:.6e} C/m^3")
        print(f"charged DG0 cells = {n_global}")
        print(f"phi(min/max) = {gmin:.6e}, {gmax:.6e}")

    # Output constraint: write_function requires degree == mesh geometry degree.
    # Mesh is linear, so output phi as CG1 for visualization.
    V1 = fem.functionspace(msh, ("CG", 1))
    phi = fem.Function(V1)
    phi.name = "phi"
    phi.interpolate(uh)

    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / "mesh_fields.xdmf"

    # One XDMF file for everything
    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(epsilon)
        xdmf.write_function(region_id)

        # ---- gate_id scaffold ----
        # This disk script does not create facet physical groups for gates.
        # When your CIF/gate workflow provides facet_tags, we can add:
        #   gate_id = DG0-on-facets or a separate boundary mesh export.
        # For now: optional debug dump of facet_tags existence.
        if write_gate_scaffold and comm.rank == 0:
            has_facets = facet_tags is not None
            print(f"[gate_id scaffold] facet_tags present: {has_facets}")
            # If you later want: write facet_tags to XDMF as MeshTags (supported in newer builds).

    if comm.rank == 0:
        print(f"Wrote: {xdmf_path} (and .h5 sidecar)")
        print("Open mesh_fields.xdmf in ParaView.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--R", type=float, required=True, help="Disk radius in meters (e.g. 80e-9)")
    ap.add_argument("--t", type=float, default=4e-9, help="Disk thickness in meters")
    ap.add_argument("--Q", type=float, default=1.602176634e-19, help="Total charge in Coulombs")
    ap.add_argument("--out", type=str, default="disk_Q1e", help="Output directory")

    ap.add_argument("--p", type=int, default=2, help="Polynomial degree for solve (default 2)")
    ap.add_argument("--epsr", type=float, default=12.0, help="Bulk relative permittivity (default 12)")
    ap.add_argument("--z0", type=float, default=0.0, help="Disk center z location (m)")

    ap.add_argument("--Lx", type=float, default=600e-9)
    ap.add_argument("--Ly", type=float, default=600e-9)
    ap.add_argument("--Lz", type=float, default=400e-9)

    ap.add_argument("--n_diam", type=int, default=28,
                    help="Target elements across disk diameter: h_disk=2R/n_diam (default 28)")
    ap.add_argument("--refine_band_R", type=float, default=8.0,
                    help="Refinement band thickness in units of R (default 8R)")
    ap.add_argument("--far_h_factor", type=float, default=25.0,
                    help="h_far = far_h_factor*h_disk (default 25)")

    ap.add_argument("--gate_scaffold", action="store_true",
                    help="Print facet_tags presence (placeholder for future gate_id export).")

    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    outdir = Path(args.out)
    msh_path = outdir / "mesh.msh"

    info = gmsh_build_box_with_refinement_field(
        comm=comm,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        R=args.R,
        z0=args.z0,
        n_diam=args.n_diam,
        refine_band_R=args.refine_band_R,
        far_h_factor=args.far_h_factor,
        msh_path=msh_path,
    )

    if comm.rank == 0:
        print("=== Mesh summary ===")
        print(f"R = {args.R:.3e} m ({args.R*1e9:.1f} nm), t = {args.t:.3e} m")
        print(f"n_diam = {args.n_diam} -> h_disk = {info['h_disk']:.3e} m ({info['h_disk']*1e9:.3f} nm)")
        print(f"h_far  = {info['h_far']:.3e} m ({info['h_far']*1e9:.3f} nm)")
        print(f"refine_band_R = {args.refine_band_R}R -> DistMax = {args.refine_band_R*args.R:.3e} m")
        print(f"Wrote mesh: {msh_path}")

    solve_and_write_all_fields(
        comm=comm,
        msh_path=msh_path,
        outdir=outdir,
        R=args.R, t=args.t, Q=args.Q,
        z0=args.z0,
        epsr_bulk=args.epsr,
        p_solve=args.p,
        write_gate_scaffold=args.gate_scaffold,
    )


if __name__ == "__main__":
    main()
