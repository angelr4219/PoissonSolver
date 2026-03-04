#!/usr/bin/env python3
import argparse
import os
import numpy as np
from mpi4py import MPI

from dolfinx import mesh, fem, io, geometry
import ufl
from petsc4py import PETSc

EPS0 = 8.8541878128e-12  # F/m

from dolfinx.fem.petsc import LinearProblem  # dolfinx 0.10.0 has this


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=float, default=300e-9)
    p.add_argument("--Ly", type=float, default=300e-9)
    p.add_argument("--Lz", type=float, default=200e-9)
    p.add_argument("--xc", type=float, default=0.0)
    p.add_argument("--yc", type=float, default=0.0)

    p.add_argument("--h", type=float, default=10e-9)
    p.add_argument("--deg", type=int, default=1)

    p.add_argument("--R", type=float, default=50e-9)
    p.add_argument("--Vdisk", type=float, default=0.1)

    p.add_argument("--tox", type=float, default=0.0)
    p.add_argument("--eps_ox", type=float, default=3.9)
    p.add_argument("--eps_si", type=float, default=11.7)
    p.add_argument("--eps_uniform", type=float, default=None)

    p.add_argument("--rho0", type=float, default=0.0)
    p.add_argument("--xq", type=float, default=0.0)
    p.add_argument("--yq", type=float, default=0.0)
    p.add_argument("--zq", type=float, default=-60e-9)
    p.add_argument("--sigma", type=float, default=15e-9)

    p.add_argument("--outdir", type=str, default="results_disk")
    p.add_argument("--basename", type=str, default="disk")
    p.add_argument("--probe_n", type=int, default=201)
    return p.parse_args()


def near(a, b, tol):
    return np.isclose(a, b, atol=tol)


def locate_facets(mesh3d, Lx, Ly, Lz, xc, yc, R):
    tdim = mesh3d.topology.dim
    fdim = tdim - 1
    xmin, xmax = -Lx/2, Lx/2
    ymin, ymax = -Ly/2, Ly/2
    zmin, zmax = -Lz, 0.0
    tol = 1e-10 * max(Lx, Ly, Lz) + 1e-12

    def on_top(x): return near(x[2], zmax, tol)
    def on_bottom(x): return near(x[2], zmin, tol)
    def on_sides(x):
        return (near(x[0], xmin, tol) | near(x[0], xmax, tol) |
                near(x[1], ymin, tol) | near(x[1], ymax, tol))

    def in_disk(x):
        r2 = (x[0] - xc)**2 + (x[1] - yc)**2
        return on_top(x) & (r2 <= (R + tol)**2)

    def top_outside_disk(x):
        r2 = (x[0] - xc)**2 + (x[1] - yc)**2
        return on_top(x) & (r2 > (R + tol)**2)

    facets_disk = mesh.locate_entities_boundary(mesh3d, fdim, in_disk)
    facets_top_out = mesh.locate_entities_boundary(mesh3d, fdim, top_outside_disk)
    facets_bottom = mesh.locate_entities_boundary(mesh3d, fdim, on_bottom)
    facets_sides = mesh.locate_entities_boundary(mesh3d, fdim, on_sides)
    return facets_disk, facets_top_out, facets_bottom, facets_sides


def build_epsilon(mesh3d, tox, eps_ox, eps_si, eps_uniform):
    V0 = fem.functionspace(mesh3d, ("DG", 0))
    eps = fem.Function(V0)

    tdim = mesh3d.topology.dim
    num_cells = mesh3d.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    mid = mesh.compute_midpoints(mesh3d, tdim, cells)
    zc = mid[:, 2]

    if eps_uniform is not None:
        eps_r = float(eps_uniform) * np.ones_like(zc)
    else:
        tox = float(tox)
        eps_r = np.where(zc > -tox, float(eps_ox), float(eps_si))

    eps.x.array[:] = (EPS0 * eps_r).astype(eps.x.array.dtype)
    return eps


def probe_line_z(mesh3d, phi, xc, yc, zmin, zmax, npts):
    comm = mesh3d.comm
    zvals = np.linspace(zmin, zmax, npts)
    pts = np.vstack([np.full_like(zvals, xc), np.full_like(zvals, yc), zvals]).T

    bb = geometry.bb_tree(mesh3d, mesh3d.topology.dim)
    cand = geometry.compute_collisions_points(bb, pts)
    coll = geometry.compute_colliding_cells(mesh3d, cand, pts)

    cell_ids = np.full(len(pts), -1, dtype=np.int32)
    for i in range(len(pts)):
        c = coll.links(i)
        if len(c) > 0:
            cell_ids[i] = c[0]

    local_vals = np.full(len(pts), np.nan, dtype=np.float64)
    mask = cell_ids != -1
    if np.any(mask):
        vals = phi.eval(pts[mask], cell_ids[mask])
        local_vals[mask] = vals.reshape(-1)

    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        phivals = np.full(len(pts), np.nan, dtype=np.float64)
        for i in range(len(pts)):
            for r in range(len(gathered)):
                v = gathered[r][i]
                if np.isfinite(v):
                    phivals[i] = v
                    break
        return zvals, phivals
    return None, None


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    os.makedirs(args.outdir, exist_ok=True)

    nx = max(2, int(round(args.Lx / args.h)))
    ny = max(2, int(round(args.Ly / args.h)))
    nz = max(2, int(round(args.Lz / args.h)))

    domain = mesh.create_box(
        comm,
        [np.array([-args.Lx/2, -args.Ly/2, -args.Lz], dtype=np.float64),
         np.array([ args.Lx/2,  args.Ly/2,  0.0], dtype=np.float64)],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron
    )

    dx = ufl.Measure("dx", domain=domain)
    tdim = domain.topology.dim
    fdim = tdim - 1

    if comm.rank == 0:
        print("=== disk_gate_poisson ===")
        print(f"Domain: Lx={args.Lx:.3e} Ly={args.Ly:.3e} Lz={args.Lz:.3e} (z in [-Lz,0])")
        print(f"Mesh: nx={nx} ny={ny} nz={nz}  (h~{args.h:.3e}), deg={args.deg}")
        print(f"Disk: center=({args.xc:.3e},{args.yc:.3e}) R={args.R:.3e} Vdisk={args.Vdisk:.3e}")
        if args.eps_uniform is not None:
            print(f"Dielectric: uniform eps_r={args.eps_uniform}")
        else:
            print(f"Dielectric: tox={args.tox:.3e} eps_ox={args.eps_ox} eps_si={args.eps_si}")
        print("Charge: rho0=0 (Laplace)" if args.rho0 == 0.0 else f"Charge: rho0={args.rho0:.3e} C/m^3")

    V = fem.functionspace(domain, ("CG", args.deg))

    facets_disk, facets_top_out, facets_bottom, facets_sides = locate_facets(
        domain, args.Lx, args.Ly, args.Lz, args.xc, args.yc, args.R
    )
    dofs_disk = fem.locate_dofs_topological(V, fdim, facets_disk)
    dofs_top_out = fem.locate_dofs_topological(V, fdim, facets_top_out)
    dofs_bottom = fem.locate_dofs_topological(V, fdim, facets_bottom)
    dofs_sides = fem.locate_dofs_topological(V, fdim, facets_sides)

    if comm.rank == 0:
        print(f"BC dof counts: disk={len(dofs_disk)} top_out={len(dofs_top_out)} bottom={len(dofs_bottom)} sides={len(dofs_sides)}")
        if len(dofs_disk) == 0:
            raise RuntimeError("Disk facet selection failed (0 dofs).")

    bcs = [
        fem.dirichletbc(PETSc.ScalarType(args.Vdisk), dofs_disk, V),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_top_out, V),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_bottom, V),
        fem.dirichletbc(PETSc.ScalarType(0.0), dofs_sides, V),
    ]

    eps = build_epsilon(domain, args.tox, args.eps_ox, args.eps_si, args.eps_uniform)

    # IMPORTANT: keep RHS as a real Form even when rho0 = 0
    rho0 = fem.Constant(domain, PETSc.ScalarType(args.rho0))
    X = ufl.SpatialCoordinate(domain)
    r2 = (X[0] - args.xq)**2 + (X[1] - args.yq)**2 + (X[2] - args.zq)**2
    gaussian = ufl.exp(-r2 / (2.0 * args.sigma**2))
    # If rho0 = 0, this is identically 0 but still carries proper form structure
    rho = rho0 * gaussian

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * dx
    L = rho * v * dx

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-14,
        "ksp_max_it": 2000
    }

    prefix = f"{args.basename}_"
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options_prefix=prefix,
                            petsc_options=petsc_options)
    phi = problem.solve()
    phi.name = "phi"

    local_min = np.min(phi.x.array.real)
    local_max = np.max(phi.x.array.real)
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    if comm.rank == 0:
        print(f"phi min={gmin:.6e}  max={gmax:.6e}")

    zvals, phivals = probe_line_z(domain, phi, args.xc, args.yc, -args.Lz, 0.0, args.probe_n)
    if comm.rank == 0:
        probe_path = os.path.join(args.outdir, f"{args.basename}_probe_centerline.txt")
        np.savetxt(probe_path, np.column_stack([zvals, phivals]),
                   header="z (m)    phi(xc,yc,z) (V)")
        print(f"Wrote probe: {probe_path}")

    xdmf_path = os.path.join(args.outdir, f"{args.basename}.xdmf")
    with io.XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        eps.name = "eps_abs"
        xdmf.write_function(eps)

    if comm.rank == 0:
        print(f"Wrote XDMF: {xdmf_path}")
        print("Done.")


if __name__ == "__main__":
    main()
