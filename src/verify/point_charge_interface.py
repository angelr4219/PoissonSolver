#!/usr/bin/env python3
# DOLFINx 0.10 verifier: empty box, single dielectric + point charge,
# two-layer interface + point charge (jump/smooth/nlayer), with E, D, Dz projections.
import json
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
EPS0 = 8.8541878128e-12

def build_box(Lx, Ly, H, h):
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, -H/2], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H/2], dtype=np.double)
    return mesh.create_box(COMM, [p0, p1], (nx, ny, nz), cell_type=mesh.CellType.tetrahedron)

def make_eps_r_constant(domain, epsr):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r"); eps.x.array[:] = epsr
    return eps

def make_eps_r_two_layer(domain, eps1, eps2):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")
    def _eps(X):
        z = X[2]
        return np.where(z >= 0.0, float(eps1), float(eps2))
    eps.interpolate(_eps); return eps

def make_eps_r_smooth(domain, eps_top, eps_bot, z0=0.0, width=2.0e-8):
    if width <= 0: raise ValueError("width must be > 0")
    W = fem.functionspace(domain, ("Lagrange", 1))
    eps = fem.Function(W, name="eps_r")
    def _eps(X):
        z = X[2]
        S = 0.5*(1.0 + np.tanh((z - z0)/width))
        return eps_bot + (eps_top - eps_bot)*S
    eps.interpolate(_eps); return eps

def make_eps_r_nlayer(domain, z_edges, eps_list):
    z_edges = np.asarray(z_edges, dtype=float)
    eps_list = np.asarray(eps_list, dtype=float)
    if z_edges.ndim!=1 or eps_list.ndim!=1: raise ValueError("z_edges/eps_list must be 1D")
    if len(z_edges)<2 or not np.all(np.diff(z_edges)>0): raise ValueError("z_edges must be increasing")
    if len(eps_list)!=len(z_edges)-1: raise ValueError("len(eps_list)=len(z_edges)-1")
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")
    z_lo, z_hi = z_edges[:-1], z_edges[1:]
    def _eps(X):
        z = X[2]
        out = np.full_like(z, eps_list[-1], dtype=float)
        for lo, hi, val in zip(z_lo, z_hi, eps_list):
            mask = (z >= lo) & (z < hi); out[mask] = val
        out[z == z_edges[-1]] = eps_list[-1]
        return out
    eps.interpolate(_eps); return eps

def gaussian_rho(domain, q, x0, sigma):
    pref = q / (np.pi**1.5 * sigma**3)
    def _rho(X):
        dx = X[0]-x0[0]; dy = X[1]-x0[1]; dz = X[2]-x0[2]
        return pref * np.exp(-(dx*dx+dy*dy+dz*dz)/(sigma*sigma))
    V0 = fem.functionspace(domain, ("Lagrange", 1))
    rho = fem.Function(V0, name="rho"); rho.interpolate(_rho); return rho

def dirichlet_zero_on_boundary(V):
    dom = V.mesh; tdim = dom.topology.dim; fdim = tdim-1
    dom.topology.create_connectivity(fdim, tdim)
    facets = mesh.locate_entities_boundary(dom, fdim, lambda x: np.ones(x.shape[1], bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return [fem.dirichletbc(PETSc.ScalarType(0.0), np.unique(dofs), V)]

def solve_poisson(V, bcs, eps_r_field, rho=None, eps0=EPS0):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_form = (eps_r_field*eps0) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = (rho * v * ufl.dx) if rho is not None else fem.Constant(V.mesh, PETSc.ScalarType(0.0))*v*ufl.dx
    uh = LinearProblem(a_form, L_form, bcs=bcs,
                       petsc_options={"ksp_type":"cg","pc_type":"hypre","pc_hypre_type":"boomeramg",
                                      "ksp_rtol":1e-9,"ksp_error_if_not_converged":True},
                       petsc_options_prefix="lp_").solve()
    uh.name = "phi"
    if V.mesh.comm.rank==0:
        arr = uh.x.array; print(f"[SOLVE] φ: min={arr.min():.6e} max={arr.max():.6e} (n={arr.size})")
    return uh

def project_scalar(domain, expr, name, qdeg=4):
    V1 = fem.functionspace(domain, ("Lagrange", 1))
    uT, v = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    dxm = ufl.dx(domain, metadata={"quadrature_degree": qdeg})
    w = LinearProblem(ufl.inner(uT,v)*dxm, ufl.inner(expr,v)*dxm,
                      petsc_options={"ksp_type":"preonly","pc_type":"lu"},
                      petsc_options_prefix="prj_").solve()
    w.name = name; return w

def project_vector(domain, expr, name, qdeg=4):
    dim = domain.geometry.dim
    Vv = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
    uT, v = ufl.TrialFunction(Vv), ufl.TestFunction(Vv)
    dxm = ufl.dx(domain, metadata={"quadrature_degree": qdeg})
    w = LinearProblem(ufl.inner(uT,v)*dxm, ufl.inner(expr,v)*dxm,
                      petsc_options={"ksp_type":"preonly","pc_type":"lu"},
                      petsc_options_prefix="prjv_").solve()
    w.name = name; return w

def sample_axis(uh, axis="z", tol=None):
    V = uh.function_space
    X = V.tabulate_dof_coordinates().reshape((-1,3))
    U = uh.x.array
    if X.shape[0]>3:
        diffs = np.diff(np.unique(X, axis=0), axis=0)
        h_est = float(np.median(np.linalg.norm(diffs, axis=1))) if diffs.size else 1e-12
    else:
        h_est = 1e-12
    tol = max(h_est*0.5, 1e-12) if tol is None else tol
    if axis=="z":
        mask = (np.abs(X[:,0])<=tol) & (np.abs(X[:,1])<=tol); coord = X[mask,2]
    elif axis=="x":
        mask = (np.abs(X[:,1])<=tol) & (np.abs(X[:,2])<=tol); coord = X[mask,0]
    else:
        raise ValueError("axis must be 'z' or 'x'")
    vals = U[mask]; order = np.argsort(coord)
    return coord[order], vals[order]

def phi_image_axis(z, a, q, eps1_abs, eps2_abs):
    q_img = (eps1_abs - eps2_abs)/(eps1_abs + eps2_abs) * q
    q_dbl = (2*eps2_abs)/(eps1_abs + eps2_abs) * q
    out = np.empty_like(z, float)
    sel1 = z >= 0.0
    if np.any(sel1):
        r1 = np.abs(z[sel1]-a); r2 = np.abs(z[sel1]+a)
        out[sel1] = (q/(4*np.pi*eps1_abs*r1)) + (q_img/(4*np.pi*eps1_abs*r2))
    sel2 = ~sel1
    if np.any(sel2):
        r1 = np.abs(z[sel2]-a)
        out[sel2] = q_dbl/(4*np.pi*eps2_abs*r1)
    return out

def run(scene, Lx, Ly, H, h, eps1, eps2, q, a, sigma, deg, out, probe_axis,
        eps_mode, z0, width, z_edges, eps_list):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    domain = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", deg))
    rho = None
    if scene=="empty":
        eps_field = make_eps_r_constant(domain, eps1)
    elif scene=="uniform":
        eps_field = make_eps_r_constant(domain, eps1)
        rho = gaussian_rho(domain, q, (0.0,0.0,a), sigma)
    elif scene=="two":
        if eps_mode=="jump":
            eps_field = make_eps_r_two_layer(domain, eps1, eps2)
        elif eps_mode=="smooth":
            eps_field = make_eps_r_smooth(domain, eps_top=eps1, eps_bot=eps2, z0=z0, width=width)
        elif eps_mode=="nlayer":
            eps_field = make_eps_r_nlayer(domain, z_edges=z_edges, eps_list=eps_list)
        else:
            raise ValueError("eps_mode must be jump|smooth|nlayer")
        rho = gaussian_rho(domain, q, (0.0,0.0,a), sigma)
    else:
        raise ValueError("scene must be empty|uniform|two")

    uh = solve_poisson(fem.functionspace(domain, ("Lagrange", deg)), dirichlet_zero_on_boundary(V), eps_field, rho)

    with io.XDMFFile(domain.comm, f"{out}.xdmf", "w") as xf:
        xf.write_mesh(domain); xf.write_function(uh); xf.write_function(eps_field)
    if domain.comm.rank==0: print(f"[WRITE] {out}.xdmf (phi, eps_r)")

    grad_phi = ufl.grad(uh)
    E_vec = -grad_phi
    D_vec = (eps_field*EPS0) * E_vec
    Dz    = ufl.inner(D_vec, ufl.as_vector((0.0,0.0,1.0)))

    E_proj  = project_vector(domain, E_vec, "E")
    D_proj  = project_vector(domain, D_vec, "D")
    Dz_proj = project_scalar(domain, Dz,  "D_z")
    with io.XDMFFile(domain.comm, f"{out}.xdmf", "a") as xf:
        xf.write_function(E_proj); xf.write_function(D_proj); xf.write_function(Dz_proj)
    if domain.comm.rank==0: print(f"[WRITE] appended E, D, D_z to {out}.xdmf")

    if probe_axis not in ("z","x"): probe_axis="z"
    coord, phi_fe = sample_axis(uh, axis=probe_axis)
    if domain.comm.rank==0:
        with open(f"{out}_line_{probe_axis}.csv","w") as f:
            f.write(f"{probe_axis},phi\n")
            for c,u in zip(coord, phi_fe): f.write(f"{c:.16e},{u:.16e}\n")
        print(f"[LINE] wrote {out}_line_{probe_axis}.csv")

    if scene=="two" and eps_mode=="jump" and probe_axis=="z" and domain.comm.rank==0:
        phi_ana = phi_image_axis(coord, a=a, q=q, eps1_abs=eps1*EPS0, eps2_abs=eps2*EPS0)
        with open(f"{out}_line_z_analytic.csv","w") as f:
            f.write("z,phi_analytic\n")
            for z,ua in zip(coord, phi_ana): f.write(f"{z:.16e},{ua:.16e}\n")
        print(f"[ANALYTIC] wrote {out}_line_z_analytic.csv")

    if domain.comm.rank==0:
        meta = dict(scene=scene, Lx=Lx, Ly=Ly, H=H, h=h, deg=deg,
                    eps1=eps1, eps2=eps2, q=q, a=a, sigma=sigma, probe=probe_axis,
                    eps_mode=eps_mode, z0=z0, width=width,
                    z_edges=(z_edges.tolist() if isinstance(z_edges,np.ndarray) else z_edges),
                    eps_list=(eps_list.tolist() if isinstance(eps_list,np.ndarray) else eps_list))
        Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[DONE] Outputs under {out.parent}")

def _parse():
    import argparse, json as _json
    p = argparse.ArgumentParser(description="Point charge & dielectric interface verifier (DOLFINx 0.10)")
    p.add_argument("--scene", choices=["empty","uniform","two"], required=True)
    p.add_argument("--eps1",  type=float, default=3.9)
    p.add_argument("--eps2",  type=float, default=11.7)
    p.add_argument("--q",     type=float, default=1.0e-12)
    p.add_argument("--a",     type=float, default=4.0e-8)
    p.add_argument("--sigma", type=float, default=3.0e-8)
    p.add_argument("--L",     type=float, default=1.0e-6)
    p.add_argument("--Ly",    type=float, default=None)
    p.add_argument("--H",     type=float, default=1.0e-6)
    p.add_argument("--h",     type=float, default=2.0e-8)
    p.add_argument("--deg",   type=int,   default=1)
    p.add_argument("--out",   type=str,   required=True)
    p.add_argument("--probe", choices=["z","x"], default="z")
    p.add_argument("--eps_mode", choices=["jump","smooth","nlayer"], default="jump")
    p.add_argument("--z0",    type=float, default=0.0)
    p.add_argument("--width", type=float, default=2.0e-8)
    p.add_argument("--z_edges",  type=str, default=None)
    p.add_argument("--eps_list", type=str, default=None)
    p.add_argument("--stamp", action="store_true", help="append YYYYmmdd_%H%M%S to --out")
    args = p.parse_args()
    if args.eps_mode=="nlayer":
        if args.z_edges is None or args.eps_list is None:
            raise SystemExit("For --eps_mode nlayer pass both --z_edges and --eps_list.")
        try:
            args.z_edges  = np.array(_json.loads(args.z_edges), dtype=float)
            args.eps_list = np.array(_json.loads(args.eps_list), dtype=float)
        except Exception as e:
            raise SystemExit(f"Failed to parse --z_edges/--eps_list JSON arrays: {e}")
    else:
        args.z_edges  = np.array([], dtype=float)
        args.eps_list = np.array([], dtype=float)
    return args

if __name__ == "__main__":
    from datetime import datetime
    args = _parse()
    if args.stamp:
        args.out = f"{args.out}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Ly = args.Ly if args.Ly is not None else args.L
    run(args.scene, args.L, Ly, args.H, args.h,
        args.eps1, args.eps2, args.q, args.a, args.sigma,
        args.deg, args.out, args.probe,
        args.eps_mode, args.z0, args.width, args.z_edges, args.eps_list)
