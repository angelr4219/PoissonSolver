#!/usr/bin/env python3
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc as fempetsc
from dolfinx.fem.petsc import LinearProblem
import ufl

# ---------- Analytic (sign fixed: positive under +V gates) ----------
def g_uvz(u, v, z):
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def phi0_rect(x, y, z, a, xs, Vs):
    z = z if z != 0.0 else np.finfo(float).eps
    s = 0.0
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            g_uvz(a - xi + x, a + y, z) +
            g_uvz(a - xi + x, a - y, z) +
            g_uvz(a + xi - x, a + y, z) +
            g_uvz(a + xi - x, a - y, z)
        )
    return s

# ---------- Mesh ----------
def build_box(Lx, Ly, H, h):
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H ], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

# ---------- BCs: scalar Dirichlet on explicit DOF sets (top gates + top rest + bottom) ----------
def gate_top_and_bottom_bcs(V, a, xs, Vs, rect_tol=1e-8, ztol=1e-9):
    domain = V.mesh
    topo = domain.topology
    tdim = topo.dim
    fdim = tdim - 1
    topo.create_connectivity(fdim, tdim)
    topo.create_connectivity(fdim, 0)

    Z = domain.geometry.x[:, 2]
    z_min = domain.comm.allreduce(float(np.min(Z)), op=MPI.MIN)
    z_max = domain.comm.allreduce(float(np.max(Z)), op=MPI.MAX)

    # Top facets/DOFs
    top_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_min, atol=ztol)
    )
    dofs_top = np.unique(fem.locate_dofs_topological(V, fdim, top_facets))
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    Xtop = X[dofs_top]

    used = np.zeros(dofs_top.shape[0], dtype=bool)
    bcs = []

    # Gates
    for xi, Vi in zip(xs, Vs):
        in_rect = (
            (Xtop[:, 0] >= (xi - a) - rect_tol) & (Xtop[:, 0] <= (xi + a) + rect_tol) &
            (Xtop[:, 1] >= -a - rect_tol)      & (Xtop[:, 1] <=  a + rect_tol)
        )
        idx = np.where(in_rect)[0]
        if idx.size:
            dofs_i = dofs_top[idx]
            bcs.append(fem.dirichletbc(PETSc.ScalarType(Vi), dofs_i, V))
            used[idx] = True

    # Top remainder = 0 V
    idx0 = np.where(~used)[0]
    if idx0.size:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs_top[idx0], V))

    # Bottom = 0 V
    bot_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_max, atol=ztol)
    )
    dofs_bot = np.unique(fem.locate_dofs_topological(V, fdim, bot_facets))
    if dofs_bot.size:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs_bot, V))

    if domain.comm.rank == 0:
        n_top = dofs_top.shape[0]
        n_gate = int(np.sum(used))
        n_bot = dofs_bot.shape[0]
        print(f"[DEBUG] TOP DOFs total={n_top}, gate-DOFs={n_gate}, top-0V DOFs={n_top-n_gate}")
        print(f"[DEBUG] BOTTOM DOFs total={n_bot}")
    return bcs

# ---------- Solve (IMPORTANT: use the SAME V the BCs were built on) ----------
def solve_laplace(V, bcs, eps_r=11.7, eps0=8.8541878128e-12):
    domain = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    lam = PETSc.ScalarType(eps_r * eps0)
    a = lam * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_error_if_not_converged": True
    }
    problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts)
    uh = problem.solve()
    uh.name = "phi"

    uarr = uh.x.array
    n_bad = int(np.sum(~np.isfinite(uarr)))
    gmin = float(np.nanmin(uarr)) if uarr.size else np.nan
    gmax = float(np.nanmax(uarr)) if uarr.size else np.nan
    if domain.comm.rank == 0:
        print(f"[VOLTAGE] nonfinite={n_bad}, min={gmin:.6e}, max={gmax:.6e}")
    return uh

# ---------- Sampling ----------
def sample_dof_line(uh: fem.Function, zbar: float, h: float, ytol: float = 1e-12):
    V = uh.function_space
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    U = uh.x.array
    ztol = max(1e-12, 0.5*h)
    mask = (np.abs(X[:,1]) <= ytol) & (np.abs(X[:,2] - zbar) <= ztol)
    if not np.any(mask):
        ztol = max(ztol, h)
        mask = (np.abs(X[:,1]) <= 5e-12) & (np.abs(X[:,2] - zbar) <= ztol)
    xs = X[mask, 0]; us = U[mask]
    if xs.size == 0:
        raise RuntimeError("No dofs found on probe line; try slightly adjusting zbar or mesh size.")
    order = np.argsort(xs)
    return xs[order], us[order]

# ---------- One run ----------
def run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix):
    domain = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", 1))
    bcs = gate_top_and_bottom_bcs(V, a, xs_gates, Vs_gates, rect_tol=1e-8, ztol=1e-9)

    uh = solve_laplace(V, bcs)  # <-- pass the SAME V

    os.makedirs(os.path.dirname(outprefix), exist_ok=True)
    with io.XDMFFile(domain.comm, f"{outprefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain); xdmf.write_function(uh)

    xnodes, uh_nodes = sample_dof_line(uh, zbar, h)
    phi0_nodes = np.array([phi0_rect(x, 0.0, zbar, a, xs_gates, Vs_gates) for x in xnodes])

    dx = np.median(np.diff(np.sort(xnodes))) if xnodes.size>1 else a*0.01
    band = 2.0*abs(dx)
    edges = np.concatenate([xs_gates - a, xs_gates + a])
    mask = (np.abs(xnodes) <= 2*a)
    for e in edges:
        mask &= (np.abs(xnodes - e) > band)

    diffs = np.abs(uh_nodes[mask] - phi0_nodes[mask])
    if diffs.size:
        err_max = float(np.max(diffs))
        err_l2  = float(np.sqrt(np.mean(diffs**2)))
    else:
        err_max = float("nan"); err_l2 = float("nan")

    if MPI.COMM_WORLD.rank == 0:
        import csv
        with open(f"{outprefix}_line.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["x_m","phi_FE_V","phi0_V"])
            for x,u,a0 in zip(xnodes, uh_nodes, phi0_nodes):
                w.writerow([x,u,a0])
        print(f"[{outprefix}] max|Δφ| (|x|≤2a, edge-safe) = {err_max:.4e} V,  L2 = {err_l2:.4e} V")
    return err_max, err_l2

# ---------- Main ----------
if __name__ == "__main__":
    a_nm = 35.0
    a = a_nm * 1e-9
    zbar = a
    xs_gates = np.array([-2*a, 0.0,  2*a])
    Vs_gates = np.array([ 0.25, 0.10, 0.25])

    H = 200e-9
    h = 5e-9

    for p in [2.0, 3.0, 4.0, 5.0]:
        Lx = Ly = 2*p*a
        tag = f"p{int(p)}a"
        run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix=f"results/phi_{tag}")
