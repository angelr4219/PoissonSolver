#!/usr/bin/env python3
# 3D box split into two dielectrics by z=z_split.
# Solves: -div( eps * grad(phi) ) = rho, with grounded outer boundary (phi=0).
# Writes phi (CG1) and eps_r (DG0) to XDMF.

import argparse, json, datetime
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

def build_box(Lx, Ly, Lz, h):
    nx = max(4, int(np.ceil(Lx/h)))
    ny = max(4, int(np.ceil(Ly/h)))
    nz = max(4, int(np.ceil(Lz/h)))
    p0 = np.array([-Lx/2, -Ly/2, -Lz/2], dtype=np.double)
    p1 = np.array([ +Lx/2, +Ly/2, +Lz/2], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

def eps_field(V, z_split, epsr_top, epsr_bottom, eps0=8.8541878128e-12):
    x = ufl.SpatialCoordinate(V.mesh)
    epsr = ufl.conditional(ufl.gt(x[2], z_split),
                           PETSc.ScalarType(epsr_top),
                           PETSc.ScalarType(epsr_bottom))
    return epsr * PETSc.ScalarType(eps0)

def rho_gaussian_3d(V, q, x0, y0, z0, sigma):
    # normalized 3D Gaussian: integral = q
    x = ufl.SpatialCoordinate(V.mesh)
    r2 = (x[0]-x0)**2 + (x[1]-y0)**2 + (x[2]-z0)**2
    norm = PETSc.ScalarType(q / ((sigma*np.sqrt(2*np.pi))**3))
    return norm * ufl.exp(-r2/(2*sigma**2))

def solve(Lx, Ly, Lz, h, z_split, epsr_top, epsr_bottom,
          q, x0, y0, z0, sigma, deg, run_root=None):
    comm = MPI.COMM_WORLD
    msh = build_box(Lx, Ly, Lz, h)
    V = fem.functionspace(msh, ("Lagrange", deg))

    # Dirichlet phi=0 on all outer faces (grounded box)
    tdim = msh.topology.dim
    facets = mesh.locate_entities_boundary(msh, tdim-1, lambda X: np.full(X.shape[1], True))
    dofs = fem.locate_dofs_topological(V, tdim-1, facets)
    bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)]

    eps = eps_field(V, z_split, epsr_top, epsr_bottom)
    rho = rho_gaussian_3d(V, q, x0, y0, z0, sigma)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dxm = ufl.dx(metadata={"quadrature_degree": 4})
    a_form = ufl.inner(eps*ufl.grad(u), ufl.grad(v)) * dxm
    L_form = rho * v * dxm

    uh = LinearProblem(a_form, L_form, bcs=bcs,
                       petsc_options={"ksp_type":"cg","pc_type":"hypre",
                                      "ksp_rtol":1e-10,"ksp_error_if_not_converged":True},
                       petsc_options_prefix="lp_").solve()
    uh.name = "phi"

    # eps_r as DG0 (interpolate at DG0 dof coords, i.e., cell centers)
    W = fem.functionspace(msh, ("Discontinuous Lagrange", 0))
    epsr_fun = fem.Function(W, name="eps_r")
    def _epsr_at_dofs(X):
        # X shape (3, N) are DG0 dof (cell-center) coords
        return np.where(X[2] > z_split, epsr_top, epsr_bottom).astype(float)
    epsr_fun.interpolate(_epsr_at_dofs)

    # Write outputs
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(run_root) if run_root else Path("results")/"runs"/f"split3d_{stamp}"
    if comm.rank==0: root.mkdir(parents=True, exist_ok=True)
    xdmf = root/"phi_eps.xdmf"
    with io.XDMFFile(comm, str(xdmf), "w") as xf:
        xf.write_mesh(msh)
        V1 = fem.functionspace(msh, ("Lagrange", 1))
        phi1 = fem.Function(V1, name="phi"); phi1.interpolate(uh)
        xf.write_function(phi1)
        xf.write_function(epsr_fun)

    if comm.rank==0:
        (root/"params.json").write_text(json.dumps({
          "Lx":Lx, "Ly":Ly, "Lz":Lz, "h":h, "z_split":z_split,
          "epsr_top":epsr_top, "epsr_bottom":epsr_bottom,
          "q":q, "x0":x0, "y0":y0, "z0":z0, "sigma":sigma, "deg":deg,
          "xdmf": str(xdmf)
        }, indent=2))
        print("[WRITE]", xdmf)
        print("[SUMMARY]", root/"params.json")

def parse_args():
    p = argparse.ArgumentParser(description="3D split-dielectric + point charge")
    p.add_argument("--Lx", type=float, default=1.4e-6)
    p.add_argument("--Ly", type=float, default=1.4e-6)
    p.add_argument("--Lz", type=float, default=1.4e-6)
    p.add_argument("--h",  type=float, default=5e-8)
    p.add_argument("--z_split", type=float, default=0.0)
    p.add_argument("--epsr_top", type=float, default=3.9)
    p.add_argument("--epsr_bottom", type=float, default=11.7)
    p.add_argument("--q", type=float, default=1.602176634e-19) # 1e (C)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--z0", type=float, default=2.0e-7)
    p.add_argument("--sigma", type=float, default=2.5e-8)
    p.add_argument("--deg", type=int, default=2)
    p.add_argument("--run-root", type=str, default=None)
    return p.parse_args()

if __name__=="__main__":
    args = parse_args()
    solve(**vars(args))
