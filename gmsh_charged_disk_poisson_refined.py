#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from mpi4py import MPI

import gmsh

from dolfinx import fem, io, mesh
from dolfinx.io import gmshio
import ufl
from petsc4py import PETSc


def build_mesh_with_refined_disk(
    comm: MPI.Comm,
    Lx: float, Ly: float, Lz: float,
    R: float, t: float,
    z0: float,
    n_diam: int,
    refine_band_R: float,
    far_h_factor: float,
    msh_path: Path,
):
    """
    Geometry: box (Lx x Ly x Lz) centered at (0,0,0)
    Charged disk region: thin cylinder radius R, thickness t, centered at z=z0.
    Mesh: fine near cylinder, coarse far away via gmsh size field.
    """
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

    # Thin cylinder (disk volume region)
    # Cylinder defined by base center and axis vector
    cyl_z0 = z0 - 0.5 * t
    cyl = occ.addCylinder(0.0, 0.0, cyl_z0, 0.0, 0.0, t, R)

    # Fragment so cylinder becomes a separate volume inside the box
    out = occ.fragment([(3, box)], [(3, cyl)])
    occ.synchronize()

    # Extract volumes after fragment
    vols = gmsh.model.getEntities(dim=3)

    # Identify "disk volume" by bounding box overlap near z0 and radius ~ R
    disk_vol_tags = []
    bulk_vol_tags = []

    for (dim, tag) in vols:
        bb = gmsh.model.getBoundingBox(dim, tag)  # (xmin, ymin, zmin, xmax, ymax, zmax)
        xmin, ymin, zmin_bb, xmax, ymax, zmax_bb = bb

        # Heuristic: disk vol should be around origin in x,y with extent ~R and in z extent ~t near z0
        xy_extent = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        z_center = 0.5 * (zmin_bb + zmax_bb)

        is_disk = (xy_extent <= 1.2 * R) and (abs(z_center - z0) <= 0.6 * t) and ((zmax_bb - zmin_bb) <= 2.0 * t)

        if is_disk:
            disk_vol_tags.append(tag)
        else:
            bulk_vol_tags.append(tag)

    if len(disk_vol_tags) != 1:
        gmsh.finalize()
        raise RuntimeError(
            f"Expected exactly 1 disk volume after fragment, got {len(disk_vol_tags)}.\n"
            f"disk_vol_tags={disk_vol_tags}\n"
            f"(If this happens, increase heuristic tolerance or inspect the fragment output.)"
        )

    disk_tag = disk_vol_tags[0]

    # Physical groups
    pg_bulk = 1
    pg_disk = 2
    gmsh.model.addPhysicalGroup(3, bulk_vol_tags, pg_bulk)
    gmsh.model.setPhysicalName(3, pg_bulk, "bulk")
    gmsh.model.addPhysicalGroup(3, [disk_tag], pg_disk)
    gmsh.model.setPhysicalName(3, pg_disk, "disk")

    # Mesh sizing:
    # h_disk from n_diam elements across diameter: h = 2R / n_diam
    h_disk = (2.0 * R) / float(n_diam)
    h_far = far_h_factor * h_disk

    # Gmsh size field: distance to disk volume boundary
    # Get boundary surfaces of disk volume
    disk_boundary = gmsh.model.getBoundary([(3, disk_tag)], oriented=False, recursive=False)
    disk_surf_tags = [tag for (d, tag) in disk_boundary if d == 2]
    if len(disk_surf_tags) == 0:
        gmsh.finalize()
        raise RuntimeError("Disk boundary surfaces not found for mesh sizing field.")

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", disk_surf_tags)

    # Refine band: from 0 to refine_band_R * R
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
    gmsh.write(str(msh_path))

    # Return some sizing info for logs
    info = dict(h_disk=h_disk, h_far=h_far, disk_tag=disk_tag, pg_bulk=pg_bulk, pg_disk=pg_disk)
    gmsh.finalize()
    return info


def solve_poisson(comm: MPI.Comm, msh_path: Path, Q: float, R: float, t: float, epsr: float, outdir: Path):
    # Read mesh from gmsh
    msh, cell_tags, facet_tags = gmshio.read_from_msh(str(msh_path), comm, gdim=3)

    # Constants
    eps0 = 8.8541878128e-12
    eps = eps0 * epsr

    # Function space
    V = fem.functionspace(msh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Piecewise rho: rho = Q / (pi R^2 t) inside disk volume, 0 elsewhere
    rho_disk = Q / (math.pi * R * R * t)

    # DG0 for rho
    Q0 = fem.functionspace(msh, ("DG", 0))
    rho = fem.Function(Q0)
    rho.x.array[:] = 0.0

    # Physical group IDs (must match build_mesh)
    pg_disk = 2

    disk_cells = cell_tags.find(pg_disk)
    if disk_cells.size == 0:
        raise RuntimeError("No disk cells found (cell_tags.find(2) returned empty). Tagging failed.")

    rho.x.array[disk_cells] = rho_disk

    # Dirichlet BC: phi=0 on entire outer boundary (simple reference)
    # (This is usually what you want for a boxed domain benchmark.)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, V)

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    problem = fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000,
        },
    )
    uh = problem.solve()

    # Print quick diagnostics
    local_min = float(np.min(uh.x.array))
    local_max = float(np.max(uh.x.array))
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    if comm.rank == 0:
        print(f"phi min/max: {gmin:.6e}, {gmax:.6e}")
        print(f"rho_disk = {rho_disk:.6e} C/m^3")

    # Output
    outdir.mkdir(parents=True, exist_ok=True)
    with io.XDMFFile(comm, str(outdir / "mesh_phi.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        uh.name = "phi"
        xdmf.write_function(uh)

    return uh


def main():
    ap = argparse.ArgumentParser(description="Charged disk Poisson with radius-aware refined gmsh mesh.")
    ap.add_argument("--Rnm", type=float, required=True, help="Disk radius in nm, e.g. 10")
    ap.add_argument("--t", type=float, default=4e-9, help="Disk thickness in meters (default 4e-9)")
    ap.add_argument("--Q", type=float, default=1.602176634e-19, help="Total charge in Coulombs (default 1e)")
    ap.add_argument("--z0", type=float, default=0.0, help="Disk center z location in meters (default 0)")
    ap.add_argument("--Lx", type=float, default=600e-9, help="Domain size x in meters")
    ap.add_argument("--Ly", type=float, default=600e-9, help="Domain size y in meters")
    ap.add_argument("--Lz", type=float, default=400e-9, help="Domain size z in meters")
    ap.add_argument("--epsr", type=float, default=12.0, help="Relative permittivity (uniform)")

    # Mesh controls
    ap.add_argument("--n_diam", type=int, default=24,
                    help="Target elements across disk diameter. h_disk = 2R/n_diam (default 24)")
    ap.add_argument("--refine_band_R", type=float, default=6.0,
                    help="Refine band thickness in units of R for size field DistMax (default 6R)")
    ap.add_argument("--far_h_factor", type=float, default=20.0,
                    help="h_far = far_h_factor * h_disk (default 20)")

    ap.add_argument("--out", type=str, default="out_disk_refined", help="Output folder")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    R = args.Rnm * 1e-9
    outdir = Path(args.out) / f"disk_R{int(args.Rnm)}nm"

    # Build mesh
    msh_path = outdir / "mesh.msh"
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    info = build_mesh_with_refined_disk(
        comm=comm,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        R=R, t=args.t,
        z0=args.z0,
        n_diam=args.n_diam,
        refine_band_R=args.refine_band_R,
        far_h_factor=args.far_h_factor,
        msh_path=msh_path,
    )

    if comm.rank == 0:
        print("=== Mesh sizing summary ===")
        print(f"R = {args.Rnm:.1f} nm, t = {args.t:.3e} m")
        print(f"n_diam = {args.n_diam} -> h_disk = {info['h_disk']:.3e} m ({info['h_disk']*1e9:.3f} nm)")
        print(f"h_far  = {info['h_far']:.3e} m ({info['h_far']*1e9:.3f} nm)")
        print(f"refine_band_R = {args.refine_band_R}R -> DistMax = {args.refine_band_R*R:.3e} m")

    # Solve
    solve_poisson(comm, msh_path, Q=args.Q, R=R, t=args.t, epsr=args.epsr, outdir=outdir)


if __name__ == "__main__":
    main()
