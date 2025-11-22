#!/usr/bin/env python3
# Dielectric sphere in uniform field: ∇·(ε∇φ)=0 with ε jump at r=a.
# Writes phi (CG1) and a DG0 relative-error heatmap (err_rel) to XDMF.

import argparse, json, datetime
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

def build_box(L, h):
    nx = ny = nz = max(4, int(np.ceil(L/h)))
    p0 = np.array([-L/2, -L/2, -L/2], dtype=np.double)
    p1 = np.array([ +L/2, +L/2, +L/2], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

def epsilon_field(V, center, a, eps_in, eps_out):
    x = ufl.SpatialCoordinate(V.mesh)
    r = ufl.sqrt((x[0]-center[0])**2 + (x[1]-center[1])**2 + (x[2]-center[2])**2)
    return ufl.conditional(ufl.lt(r, a), eps_in, eps_out)

def analytic_phi(xyz, a, eps_in, eps_out, E0, center=(0,0,0)):
    X, Y, Z = xyz[0]-center[0], xyz[1]-center[1], xyz[2]-center[2]
    r = np.sqrt(X*X + Y*Y + Z*Z) + 1e-300
    cos_th = Z / r
    alpha = (eps_in - eps_out) / (eps_in + 2*eps_out)
    phi_out = -E0*r*cos_th + E0*alpha*(a**3)/(r**2)*cos_th      # r >= a
    phi_in  = -E0*(3*eps_out)/(eps_in + 2*eps_out) * r * cos_th # r <= a
    return np.where(r <= a, phi_in, phi_out)

def solve_dielectric_sphere(L, a, E0, eps_r_in, eps_r_out, h, deg, relerr_tau, run_root=None):
    comm = MPI.COMM_WORLD
    eps0 = 8.8541878128e-12
    eps_in  = PETSc.ScalarType(eps0*eps_r_in)
    eps_out = PETSc.ScalarType(eps0*eps_r_out)
    center = (0.0, 0.0, 0.0)

    # Mesh & FE space
    msh = build_box(L, h)
    V = fem.functionspace(msh, ("Lagrange", deg))

    # Dirichlet on outer boundary: phi = -E0 * z (uniform field)
    phi_bc_fun = fem.Function(V)
    phi_bc_fun.interpolate(lambda X: -E0 * X[2])  # X has shape (3, N)
    facets = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda X: np.full(X.shape[1], True)
    )
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bcs = [fem.dirichletbc(phi_bc_fun, dofs)]

    # Weak form: ∫ ε ∇u·∇v dx = 0
    eps = epsilon_field(V, center, a, eps_in, eps_out)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dxm = ufl.dx
    a_form = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * dxm
    zero = fem.Constant(msh, PETSc.ScalarType(0.0))   # domain-bound RHS constant
    L_form = zero * v * dxm

    uh = LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10,
                       "ksp_error_if_not_converged":True},
        petsc_options_prefix="lp_"
    ).solve()
    uh.name = "phi"

    # Analytic in same space
    uE = fem.Function(V, name="phi_exact")
    uE.interpolate(lambda X: analytic_phi(X, a, float(eps_in), float(eps_out), E0, center))

    # Error norms
    dxq = ufl.dx(metadata={"quadrature_degree": 4})
    e = uh - uE
    L2  = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(e, e) * dxq))))
    H1s = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dxq))))
    Linf_nodes = float(np.max(np.abs(uh.x.array))) if uh.x.array.size else 0.0

    # Relative-error heatmap (DG0): project |e|/(|uE|+tau)
    W = fem.functionspace(msh, ("Discontinuous Lagrange", 0))
    uT, vv = ufl.TrialFunction(W), ufl.TestFunction(W)
    num = ufl.sqrt(ufl.max_value(e*e, 0.0))
    den = ufl.sqrt(ufl.max_value(uE*uE, 0.0)) + relerr_tau
    f_rel = num / den
    aP = ufl.inner(uT, vv) * dxq
    bP = ufl.inner(f_rel, vv) * dxq
    w_rel = LinearProblem(
        aP, bP,
        petsc_options={"ksp_type":"preonly","pc_type":"lu"},
        petsc_options_prefix="proj_"
    ).solve()
    w_rel.name = "err_rel"

    # Write outputs
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(run_root) if run_root else Path("results")/"runs"/f"dielectric_sphere_{stamp}"
    if comm.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    xdmf_path = root / "phi_and_err.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xf:
        xf.write_mesh(msh)
        V1 = fem.functionspace(msh, ("Lagrange", 1))
        phi1 = fem.Function(V1, name="phi"); phi1.interpolate(uh)
        xf.write_function(phi1)   # potential
        xf.write_function(w_rel)  # DG0 relative error

    if comm.rank == 0:
        (root/"summary.json").write_text(json.dumps({
            "L": L, "a": a, "E0": E0,
            "eps_r_in": eps_r_in, "eps_r_out": eps_r_out,
            "deg": deg, "h": h,
            "L2": L2, "H1_semi": H1s, "Linf_nodes": Linf_nodes,
            "xdmf": str(xdmf_path)
        }, indent=2))
        print("[WRITE]", xdmf_path)
        print("[SUMMARY]", root/"summary.json")
    return xdmf_path

def parse_args():
    p = argparse.ArgumentParser(description="Dielectric sphere benchmark with analytic comparison")
    p.add_argument("--L", type=float, default=1.4, help="Box length (m)")
    p.add_argument("--a", type=float, default=0.35, help="Sphere radius (m)")
    p.add_argument("--E0", type=float, default=1.0, help="Uniform field magnitude")
    p.add_argument("--epsr_in", type=float, default=10.0, help="Relative permittivity inside sphere")
    p.add_argument("--epsr_out", type=float, default=1.0, help="Relative permittivity outside")
    p.add_argument("--h", type=float, default=0.04, help="Target element size (m)")
    p.add_argument("--deg", type=int, default=2, help="Lagrange degree")
    p.add_argument("--relerr_tau", type=float, default=1e-9, help="Relative-error denominator tau")
    p.add_argument("--run-root", type=str, default=None, help="Output folder (default: results/runs/...)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    solve_dielectric_sphere(L=args.L, a=args.a, E0=args.E0,
                            eps_r_in=args.epsr_in, eps_r_out=args.epsr_out,
                            h=args.h, deg=args.deg, relerr_tau=args.relerr_tau,
                            run_root=args.run_root)
