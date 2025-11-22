#!/usr/bin/env python3
import os, csv, math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# ---------- Analytic rectangular-gate potential ----------
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

# ---------- Meshing ----------
def build_box(Lx, Ly, H, nx, ny, nz):
    p0 = np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H ], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

# ---------- BCs: top gates (Dirichlet), top-rest 0V, bottom 0V ----------
def gate_top_and_bottom_bcs(V, a, xs, Vs, rect_tol=1e-8, ztol=1e-9):
    domain = V.mesh
    topo = domain.topology
    tdim = topo.dim; fdim = tdim - 1
    topo.create_connectivity(fdim, tdim)
    topo.create_connectivity(fdim, 0)

    Z = domain.geometry.x[:, 2]
    z_min = domain.comm.allreduce(float(np.min(Z)), op=MPI.MIN)
    z_max = domain.comm.allreduce(float(np.max(Z)), op=MPI.MAX)

    top_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_min, atol=ztol)
    )
    dofs_top = np.unique(fem.locate_dofs_topological(V, fdim, top_facets))
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    Xtop = X[dofs_top]

    used = np.zeros(dofs_top.shape[0], dtype=bool)
    bcs = []

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

    idx0 = np.where(~used)[0]
    if idx0.size:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs_top[idx0], V))

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

# ---------- Solve ----------
def solve_laplace(V, bcs, eps_r=11.7, eps0=8.8541878128e-12):
    domain = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    lam = PETSc.ScalarType(eps_r * eps0)
    a = lam * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_error_if_not_converged": True
        },
        petsc_options_prefix="phi_"
    )
    uh = problem.solve()
    uh.name = "phi"
    uarr = uh.x.array
    n_bad = int(np.sum(~np.isfinite(uarr)))
    gmin = float(np.nanmin(uarr)) if uarr.size else np.nan
    gmax = float(np.nanmax(uarr)) if uarr.size else np.nan
    if domain.comm.rank == 0:
        print(f"[VOLTAGE] nonfinite={n_bad}, min={gmin:.6e}, max={gmax:.6e}")
    return uh

# ---------- Errors ----------
def compute_errors(domain, V, uh, a, xs_gates, Vs_gates, zbar):
    def exact_xyz(xyz):
        return np.array([phi0_rect(x, y, z, a, xs_gates, Vs_gates) for (x,y,z) in xyz], dtype=float)

    uE = fem.Function(V, name="u_exact")
    dof_xyz = V.tabulate_dof_coordinates().reshape((-1,3))
    uE.x.array[:] = exact_xyz(dof_xyz)

    e = uE - uh
    L2_e_sq  = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx(domain)))
    H1_e_sq  = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx(domain)))
    L2_uE_sq = fem.assemble_scalar(fem.form(ufl.inner(uE, uE) * ufl.dx(domain)))
    comm = domain.comm
    L2_e_sq  = comm.allreduce(L2_e_sq,  op=MPI.SUM)
    H1_e_sq  = comm.allreduce(H1_e_sq,  op=MPI.SUM)
    L2_uE_sq = comm.allreduce(L2_uE_sq, op=MPI.SUM)
    L2_abs = float(np.sqrt(L2_e_sq))
    H1_semi = float(np.sqrt(H1_e_sq))
    L2_rel = float(L2_abs / np.sqrt(L2_uE_sq)) if L2_uE_sq > 0 else np.nan

    uh_arr, uE_arr = uh.x.array, uE.x.array
    rel_denom = np.maximum(np.abs(uE_arr), 1e-12)
    Linf_rel_dof_local = float(np.max(np.abs(uE_arr - uh_arr) / rel_denom)) if uh_arr.size else 0.0
    Linf_rel_dof = comm.allreduce(Linf_rel_dof_local, op=MPI.MAX)

    X = V.tabulate_dof_coordinates().reshape((-1,3))
    h_guess = np.median(np.diff(np.unique(np.sort(X[:,0])))) if X.shape[0] > 8 else (a/4)
    ztol = max(1e-12, 0.5*h_guess)
    on_line = (np.abs(X[:,1]) <= 1e-12) & (np.abs(X[:,2] - zbar) <= ztol)
    xs = X[on_line, 0]
    uh_line = uh_arr[on_line]
    phi_line = np.array([phi0_rect(x, 0.0, zbar, a, xs_gates, Vs_gates) for x in xs])
    band = 2.0*abs(h_guess)
    edges = np.concatenate([xs_gates - a, xs_gates + a])
    mask = (np.abs(xs) <= 2*a)
    for e0 in edges:
        mask &= (np.abs(xs - e0) > band)
    diffs = np.abs(uh_line[mask] - phi_line[mask])
    err_max_line = float(np.max(diffs)) if diffs.size else np.nan
    err_l2_line  = float(np.sqrt(np.mean(diffs**2))) if diffs.size else np.nan

    if comm.rank == 0:
        print(f"[REL]  L2_rel = {L2_rel:.6e}   Linf_rel(dof) = {Linf_rel_dof:.6e}")
        print(f"[GLOBAL] ||e||_L2 = {L2_abs:.6e}   |e|_H1 = {H1_semi:.6e}")
        print(f"[LINE]  max|Δφ| = {err_max_line:.4e} V   L2 ≈ {err_l2_line:.4e} V")

    return dict(
        L2_rel=L2_rel, Linf_rel_dof=Linf_rel_dof,
        L2_abs=L2_abs, H1_semi=H1_semi,
        err_max_line=err_max_line, err_l2_line=err_l2_line
    )

def write_phi_xdmf(domain, uh, outprefix):
    with io.XDMFFile(domain.comm, f"{outprefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        V1 = fem.functionspace(domain, ("Lagrange", 1))
        uh1 = fem.Function(V1, name="phi")
        uh1.interpolate(uh)
        xdmf.write_function(uh1)

# ---------- Sweep ----------
def run_case(a, p, H, deg, nx_hint, xs_gates, Vs_gates, outdir, zbar=None):
    comm = MPI.COMM_WORLD
    Lx = Ly = 2*p*a
    zbar = a if zbar is None else zbar

    nx = ny = int(nx_hint)
    h = Lx / nx
    nz = max(2, int(round(H / h)))

    domain = build_box(Lx, Ly, H, nx, ny, nz)
    V = fem.functionspace(domain, ("Lagrange", deg))
    bcs = gate_top_and_bottom_bcs(V, a, xs_gates, Vs_gates)

    uh = solve_laplace(V, bcs)
    metrics = compute_errors(domain, V, uh, a, xs_gates, Vs_gates, zbar)

    if comm.rank == 0:
        os.makedirs(outdir, exist_ok=True)
        tag = f"h{int(round(h*1e9))}nm_p{p}_d{deg}_nx{nx}"
        outprefix = f"{outdir}/phi_{tag}"
        write_phi_xdmf(domain, uh, outprefix)

        cells = domain.topology.index_map(domain.topology.dim).size_global
        points = domain.topology.index_map(0).size_global
        row = [
            p, H, deg, nx, h, cells, points,
            metrics["L2_rel"], metrics["Linf_rel_dof"],
            metrics["err_max_line"], metrics["err_l2_line"], outprefix
        ]
        with open(f"{outdir}/analytic_sweep.csv", "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["p","H_m","deg","nx","h_m","cells","points",
                            "L2_rel","Linf_rel_dof","line_err_max_V","line_err_L2_V","outprefix"])
            w.writerow(row)
        print(f"[WRITE] {outprefix}.xdmf  (cells={cells:,}, points={points:,})")
    return metrics

def main():
    a_nm   = float(os.getenv("A_NM", "35.0"))
    deg    = int(os.getenv("DEG", "2"))
    H_nm   = float(os.getenv("H_NM", "200.0"))
    pads   = [int(x) for x in os.getenv("PADS", "2,3,4,5").split(",")]
    nxs    = [int(x) for x in os.getenv("NXS",  "32,48,64").split(",")]
    outdir = os.getenv("OUTDIR", "results/analytic_rect_sweep")

    a = a_nm * 1e-9
    H = H_nm * 1e-9
    xs_gates = np.array([-2*a, 0.0, 2*a])
    Vs_gates = np.array([0.25, 0.10, 0.25])
    print(f"[ANALYTIC SWEEP] a={a_nm} nm  deg={deg}  H={H_nm} nm  pads={pads}  nxs={nxs}")

    for p in pads:
        for nx in nxs:
            print(f" -> p{p}: nx={nx}, deg={deg}")
            run_case(a, p, H, deg, nx, xs_gates, Vs_gates, outdir, zbar=a)

if __name__ == "__main__":
    main()
