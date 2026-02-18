from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import ufl
from dolfinx import fem, io
from dolfinx.io import gmshio, VTXWriter
from dolfinx.fem.petsc import LinearProblem


EPS0 = 8.8541878128e-12  # F/m


@dataclass
class PhysTags:
    box_vol: int = 1
    disk_vol: int = 2
    outer_bnd: int = 11


def model_to_mesh_robust(model, comm, rank: int, gdim: int = 3):
    out = gmshio.model_to_mesh(model, comm, rank=rank, gdim=gdim)
    if isinstance(out, (tuple, list)):
        if len(out) < 3:
            raise RuntimeError(f"gmshio.model_to_mesh returned {len(out)} items, expected at least 3")
        return out[0], out[1], out[2]
    raise RuntimeError("Unexpected gmshio.model_to_mesh return type")


def build_gmsh_box_with_thin_disk_scaled(
    Lx: float, Ly: float, Lz: float,
    R: float, t: float, z0: float,
    h_box: float, h_disk: float,
    phys: PhysTags,
    comm: MPI.Comm,
    gmsh_scale: float,
):
    if comm.rank != 0:
        return

    s = gmsh_scale
    Lx_s, Ly_s, Lz_s = Lx * s, Ly * s, Lz * s
    R_s, t_s, z0_s = R * s, t * s, z0 * s
    h_box_s, h_disk_s = h_box * s, h_disk * s

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("charged_disk_box_scaled")
    occ = gmsh.model.occ

    x0 = -Lx_s / 2.0
    y0 = -Ly_s / 2.0
    zmin = -Lz_s / 2.0
    zmax = +Lz_s / 2.0

    box = occ.addBox(x0, y0, zmin, Lx_s, Ly_s, Lz_s)
    disk = occ.addCylinder(0.0, 0.0, z0_s - t_s / 2.0, 0.0, 0.0, t_s, R_s)
    occ.synchronize()

    occ.fragment([(3, box)], [(3, disk)])
    occ.synchronize()

    occ.removeAllDuplicates()
    occ.synchronize()

    vols = gmsh.model.getEntities(dim=3)
    if len(vols) < 2:
        raise RuntimeError(f"Expected >=2 volumes after fragment, got {len(vols)}")

    def bbox(ent):
        return gmsh.model.getBoundingBox(ent[0], ent[1])

    disk_vol = None
    tol = 1e-6
    for v in vols:
        bb = bbox(v)
        dx, dy, dz = (bb[3]-bb[0]), (bb[4]-bb[1]), (bb[5]-bb[2])
        if abs(dz - t_s) < tol and dx <= 2.2 * R_s and dy <= 2.2 * R_s:
            cx, cy, cz = 0.5*(bb[0]+bb[3]), 0.5*(bb[1]+bb[4]), 0.5*(bb[2]+bb[5])
            if abs(cx) < tol and abs(cy) < tol and abs(cz - z0_s) < tol:
                disk_vol = v
                break
    if disk_vol is None:
        candidates = []
        for v in vols:
            bb = bbox(v)
            candidates.append(((bb[3]-bb[0])*(bb[4]-bb[1])*(bb[5]-bb[2]), v))
        candidates.sort(key=lambda x: x[0])
        disk_vol = candidates[0][1]

    box_vols = [v for v in vols if v != disk_vol]

    gmsh.model.addPhysicalGroup(3, [disk_vol[1]], phys.disk_vol)
    gmsh.model.setPhysicalName(3, phys.disk_vol, "disk_volume")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in box_vols], phys.box_vol)
    gmsh.model.setPhysicalName(3, phys.box_vol, "box_volume")

    surfs = gmsh.model.getEntities(dim=2)
    outer = []
    for srf in surfs:
        bb = bbox(srf)
        if (abs(bb[0]-x0) < tol and abs(bb[3]-x0) < tol) or \
           (abs(bb[0]-(x0+Lx_s)) < tol and abs(bb[3]-(x0+Lx_s)) < tol) or \
           (abs(bb[1]-y0) < tol and abs(bb[4]-y0) < tol) or \
           (abs(bb[1]-(y0+Ly_s)) < tol and abs(bb[4]-(y0+Ly_s)) < tol) or \
           (abs(bb[2]-zmin) < tol and abs(bb[5]-zmin) < tol) or \
           (abs(bb[2]-zmax) < tol and abs(bb[5]-zmax) < tol):
            outer.append(srf[1])
    outer = sorted(set(outer))
    if not outer:
        raise RuntimeError("Failed to identify outer boundary surfaces.")

    gmsh.model.addPhysicalGroup(2, outer, phys.outer_bnd)
    gmsh.model.setPhysicalName(2, phys.outer_bnd, "outer_boundary")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(h_disk_s, h_box_s))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(h_disk_s, h_box_s))

    disk_surfs = gmsh.model.getBoundary([disk_vol], oriented=False, recursive=False)
    disk_surfs = [e[1] for e in disk_surfs if e[0] == 2]
    if disk_surfs:
        fdist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fdist, "FacesList", disk_surfs)
        fthr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMin", h_disk_s)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMax", h_box_s)
        gmsh.model.mesh.field.setNumber(fthr, "DistMin", 0.5 * R_s)
        gmsh.model.mesh.field.setNumber(fthr, "DistMax", 2.0 * R_s)
        gmsh.model.mesh.field.setAsBackgroundMesh(fthr)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=float, default=600e-9)
    p.add_argument("--Ly", type=float, default=600e-9)
    p.add_argument("--Lz", type=float, default=400e-9)
    p.add_argument("--R", type=float, default=80e-9)
    p.add_argument("--t", type=float, default=4e-9)
    p.add_argument("--z0", type=float, default=0.0)
    p.add_argument("--eps_r", type=float, default=11.7)
    p.add_argument("--Q", type=float, default=1.602176634e-19)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--h_box", type=float, default=20e-9)
    p.add_argument("--h_disk", type=float, default=6e-9)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--gmsh_scale", type=float, default=1e9)
    p.add_argument("--out", type=str, default="out_disk")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    phys = PhysTags()

    build_gmsh_box_with_thin_disk_scaled(
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        R=args.R, t=args.t, z0=args.z0,
        h_box=args.h_box, h_disk=args.h_disk,
        phys=phys, comm=comm, gmsh_scale=args.gmsh_scale,
    )

    if rank == 0:
        msh, ct, ft = model_to_mesh_robust(gmsh.model, comm, rank=0, gdim=3)
        gmsh.finalize()
    else:
        msh, ct, ft = model_to_mesh_robust(None, comm, rank=0, gdim=3)

    msh.geometry.x[:] /= args.gmsh_scale

    V = fem.functionspace(msh, ("CG", args.degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps = fem.Constant(msh, PETSc.ScalarType(args.eps_r * EPS0))

    area = math.pi * args.R**2
    Q = args.Q if args.sigma is None else args.sigma * area
    vol = area * args.t
    rho_disk = Q / vol

    W0 = fem.functionspace(msh, ("DG", 0))
    rho = fem.Function(W0)
    rho.x.array[:] = 0.0
    disk_cells = ct.find(phys.disk_vol)
    rho.x.array[disk_cells] = rho_disk

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    outer_facets = ft.find(phys.outer_bnd)
    outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)
    bc0 = fem.dirichletbc(PETSc.ScalarType(0.0), outer_dofs, V)

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=[bc0],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
    )
    phi = problem.solve()
    phi.name = "phi"

    local_min = np.min(phi.x.array) if phi.x.array.size else 0.0
    local_max = np.max(phi.x.array) if phi.x.array.size else 0.0
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    Q_num_local = fem.assemble_scalar(fem.form(rho * ufl.dx))
    Q_num = comm.allreduce(Q_num_local, op=MPI.SUM)

    if rank == 0:
        print("=== Charged disk Poisson run ===")
        print(f"R,t,z0       = {args.R:.3e}, {args.t:.3e}, {args.z0:.3e} m")
        print(f"Q target     = {Q:.6e} C")
        print(f"Q assembled  = {Q_num:.6e} C")
        print(f"phi min/max  = {gmin:.6e}, {gmax:.6e} V")

    # --- Output ---
    # 1) Parallel-safe output (ParaView): .bp
    out_bp = f"{args.out}.bp"
    with VTXWriter(comm, out_bp, [phi], engine="BP4") as vtx:
        vtx.write(0.0)
    if rank == 0:
        print(f"Wrote: {out_bp}")

    # 2) Convenience XDMF: interpolate to CG1 so degree matches mesh geometry degree
    V1 = fem.functionspace(msh, ("CG", 1))
    phi1 = fem.Function(V1)
    phi1.name = "phi"
    phi1.interpolate(phi)

    out_xdmf = f"{args.out}.xdmf"
    with io.XDMFFile(comm, out_xdmf, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi1)
    if rank == 0:
        print(f"Wrote: {out_xdmf}")


if __name__ == "__main__":
    main()
