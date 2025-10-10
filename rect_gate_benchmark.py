#!/usr/bin/env python3
import os, numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

def g_uvz(u, v, z):
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def phi0_rect(x, y, z, a, xs, Vs):
    s = 0.0
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            g_uvz(a - xi + x, a + y, z) +
            g_uvz(a - xi + x, a - y, z) +
            g_uvz(a + xi - x, a + y, z) +
            g_uvz(a + xi - x, a - y, z)
        )
    return -s

def build_box(Lx, Ly, H, h):
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H ], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

# ---- clamp top (z=min) with gates & ground bottom (z=max) ----
def gate_and_bottom_bcs(V, a, xs, Vs, rect_tol=1e-8, ztol=1e-9):
    domain = V.mesh; topo = domain.topology; tdim = topo.dim; fdim = tdim-1
    topo.create_connectivity(fdim, tdim); topo.create_connectivity(fdim, 0)
    z = domain.geometry.x[:,2]
    zmin = domain.comm.allreduce(float(np.min(z)), op=MPI.MIN)
    zmax = domain.comm.allreduce(float(np.max(z)), op=MPI.MAX)
    # top
    top_facets = mesh.locate_entities_boundary(domain, fdim,
                  lambda x: np.isclose(x[2], zmin, atol=ztol))
    dofs_top = np.unique(fem.locate_dofs_topological(V, fdim, top_facets))
    X = V.tabulate_dof_coordinates().reshape((-1,3)); Xtop = X[dofs_top]
    vals_top = np.zeros(dofs_top.shape[0], dtype=PETSc.ScalarType)
    for xi, Vi in zip(xs, Vs):
        in_rect = ((Xtop[:,0] >= (xi-a)-rect_tol) & (Xtop[:,0] <= (xi+a)+rect_tol) &
                   (Xtop[:,1] >= -a-rect_tol) & (Xtop[:,1] <=  a+rect_tol))
        vals_top[in_rect] = PETSc.ScalarType(Vi)
    bc_top_fun = fem.Function(V); bc_top_fun.x.array[:] = 0.0
    bc_top_fun.x.array[dofs_top] = vals_top
    bc_top = fem.dirichletbc(bc_top_fun, dofs_top)
    # bottom
    bot_facets = mesh.locate_entities_boundary(domain, fdim,
                  lambda x: np.isclose(x[2], zmax, atol=ztol))
    dofs_bot = np.unique(fem.locate_dofs_topological(V, fdim, bot_facets))
    bc_bot = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_bot, V)
    if domain.comm.rank==0:
        print(f"[DEBUG] top DOFs (unique): {dofs_top.shape[0]}, gate-DOFs: {int(np.count_nonzero(vals_top))}")
        print(f"[DEBUG] bottom DOFs (unique): {dofs_bot.shape[0]}")
    return [bc_top, bc_bot]

def solve_laplace(domain, bcs, eps_r=11.7, eps0=8.8541878128e-12):
    V = fem.functionspace(domain, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (eps_r*eps0) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx
    uh = LinearProblem(a, L, bcs=bcs).solve(); uh.name = "phi"; return V, uh

def sample_dof_line(uh: fem.Function, zbar: float, h: float, ytol: float=1e-12):
    V = uh.function_space
    X = V.tabulate_dof_coordinates().reshape((-1,3)); U = uh.x.array
    ztol = max(1e-12, 0.5*h)
    mask = (np.abs(X[:,1])<=ytol) & (np.abs(X[:,2]-zbar)<=ztol)
    if not np.any(mask):
        ztol = max(ztol, h)
        mask = (np.abs(X[:,1])<=5e-12) & (np.abs(X[:,2]-zbar)<=ztol)
    xs, us = X[mask,0], U[mask]
    if xs.size==0: raise RuntimeError("No dofs on probe line.")
    order = np.argsort(xs); return xs[order], us[order]

def run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix):
    domain = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", 1))
    bcs = gate_and_bottom_bcs(V, a, xs_gates, Vs_gates)

    V, uh = solve_laplace(domain, bcs)

    # --- print basic voltage stats over the whole field
    uarr = uh.x.array
    n_bad = int(np.sum(~np.isfinite(uarr)))
    if MPI.COMM_WORLD.rank==0:
        print(f"[VOLTAGE] nonfinite={n_bad}, min={np.nanmin(uarr):.4e}, max={np.nanmax(uarr):.4e}")

    os.makedirs(os.path.dirname(outprefix), exist_ok=True)
    with io.XDMFFile(domain.comm, f"{outprefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain); xdmf.write_function(uh)

    # --- compare along line y=0, z=zbar and print a few samples
    xnodes, uh_nodes = sample_dof_line(uh, zbar, h)
    phi0_nodes = np.array([phi0_rect(x, 0.0, zbar, a, xs_gates, Vs_gates) for x in xnodes])

    # print a few example voltages (near center and near ±2a)
    if MPI.COMM_WORLD.rank==0:
        def nearest(x0):
            i = int(np.argmin(np.abs(xnodes-x0))); return xnodes[i], uh_nodes[i], phi0_nodes[i]
        for x0 in [0.0, -2*a, 2*a]:
            xsmp, ufe, uana = nearest(x0)
            print(f"[SAMPLE] x≈{x0:.2e} -> FE={ufe:.4e} V, analytic={uana:.4e} V, Δ={ufe-uana:.2e} V")

    # restrict error to |x|<=2a
    mask = np.abs(xnodes) <= 2*a
    if not np.any(mask):
        if MPI.COMM_WORLD.rank==0:
            print("[WARN] no nodes in |x|<=2a; relaxing to |x|<=3a")
        mask = np.abs(xnodes)<=3*a
    diffs = np.abs(uh_nodes[mask] - phi0_nodes[mask])
    err_max = float(np.max(diffs)) if diffs.size else float('nan')
    err_l2  = float(np.sqrt(np.mean(diffs**2))) if diffs.size else float('nan')

    if MPI.COMM_WORLD.rank==0:
        import csv
        with open(f"{outprefix}_line.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["x_m","phi_FE_V","phi0_V"])
            for x,u,a0 in zip(xnodes, uh_nodes, phi0_nodes): w.writerow([x,u,a0])
        print(f"[{outprefix}] max|Δφ| (|x|≤2a) = {err_max:.4e} V,  L2 = {err_l2:.4e} V")
    return err_max, err_l2

if __name__ == "__main__":
    a_nm=35.0; a=a_nm*1e-9; zbar=a
    xs_gates=np.array([-2*a,0.0,2*a]); Vs_gates=np.array([0.25,0.10,0.25])
    H=200e-9; h=5e-9
    for p in [2.0,3.0,4.0,5.0]:
        Lx=Ly=2*p*a; tag=f"p{int(p)}a"
        run_once(Lx,Ly,H,h,a,xs_gates,Vs_gates,zbar,outprefix=f"results/phi_{tag}")
