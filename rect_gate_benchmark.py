#!/usr/bin/env python3
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# ---------- Analytic for line probe ----------
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

# ---------- BCs: top gates (Dirichlet), top-rest 0V, bottom 0V ----------
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

    # top facets at z_min
    top_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_min, atol=ztol)
    )
    dofs_top = np.unique(fem.locate_dofs_topological(V, fdim, top_facets))
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    Xtop = X[dofs_top]

    used = np.zeros(dofs_top.shape[0], dtype=bool)
    bcs = []

    # gate patches
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

    # top remainder 0V
    idx0 = np.where(~used)[0]
    if idx0.size:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs_top[idx0], V))

    # bottom 0V (z_max)
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

    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_error_if_not_converged": True
    }
    problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts)
    uh = problem.solve()
    uh.name = "phi"

    # sanity report
    uarr = uh.x.array
    n_bad = int(np.sum(~np.isfinite(uarr)))
    gmin = float(np.nanmin(uarr)) if uarr.size else np.nan
    gmax = float(np.nanmax(uarr)) if uarr.size else np.nan
    if domain.comm.rank == 0:
        print(f"[VOLTAGE] nonfinite={n_bad}, min={gmin:.6e}, max={gmax:.6e}")
    return uh

# ---------- Sampling along y=0 at z=zbar ----------
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

    uh = solve_laplace(V, bcs)

    # === Error analysis (global + cellwise fields) ===
    from errors import report_errors
    from exact_rect_gates import phi0_rect_three_gates_factory
    os.makedirs("results", exist_ok=True)
    exact_fun = phi0_rect_three_gates_factory(a, zbar, xs_gates, Vs_gates)
    metrics = report_errors(domain, uh, exact_fun, out_prefix=f"{outprefix}_err", qdeg=4)
    # --- Relative error: (exact - solution) / exact ---
    # Pointwise field for ParaView and global relative metrics for CSV
    # Build exact field in same space
    uE = fem.Function(V, name="u_exact_point_rel"); uE.interpolate(exact_fun)
    e = uE - uh
    
    # pointwise relative error at interpolation points: rel_e = |e| / max(|uE|, rel_tol)
    from dolfinx import fem as _fem
    import ufl as _ufl
    import numpy as _np
    from mpi4py import MPI as _MPI
    
    rel_tol_local = 1e-12
    max_uE_local = float(_np.max(_np.abs(uE.x.array))) if uE.x.array.size else 0.0
    max_uE = domain.comm.allreduce(max_uE_local, op=_MPI.MAX)
    rel_tol = max(rel_tol_local, 1e-9*max_uE)  # floor to avoid divide-by-zero where exact≈0
    rel_tol_const = fem.Constant(domain, PETSc.ScalarType(rel_tol))
    expr_rel = _fem.Expression(_ufl.sqrt(e*e) / _ufl.max_value(_ufl.sqrt(uE*uE), rel_tol_const), V.element.interpolation_points())
    rel_e = fem.Function(V, name="rel_e")
    rel_e.interpolate(expr_rel)
    with io.XDMFFile(domain.comm, f"{outprefix}_rel_point.xdmf", "w") as xf:
        xf.write_mesh(domain); xf.write_function(rel_e)
    
    # global relative-L2 and relative Linf(dof)
    # L2_rel = ||e||_L2 / ||uE||_L2
    L2_e_sq  = fem.assemble_scalar(fem.form(_ufl.inner(e, e) * _ufl.dx(domain)))
    L2_uE_sq = fem.assemble_scalar(fem.form(_ufl.inner(uE, uE) * _ufl.dx(domain)))
    L2_e_sq  = domain.comm.allreduce(L2_e_sq,  op=_MPI.SUM)
    L2_uE_sq = domain.comm.allreduce(L2_uE_sq, op=_MPI.SUM)
    L2_rel = float((_np.sqrt(L2_e_sq) / _np.sqrt(L2_uE_sq)) if L2_uE_sq > 0 else _np.nan)
    
    # Linf_rel_dof = max_i |e_i| / max(|uE_i|, rel_tol)
    uh_arr = uh.x.array; uE_arr = uE.x.array
    den = _np.maximum(_np.abs(uE_arr), rel_tol)
    Linf_rel_local = float(_np.max(_np.abs(uE_arr - uh_arr) / den)) if uh_arr.size else 0.0
    Linf_rel_dof = domain.comm.allreduce(Linf_rel_local, op=_MPI.MAX)
    
    if domain.comm.rank == 0:
        print(f"[REL]  L2_rel = {L2_rel:.6e}   Linf_rel(dof) = {Linf_rel_dof:.6e}   (rel_tol={rel_tol:.2e})")
        import csv, os as _os
        _os.makedirs("results", exist_ok=True)
        with open("results/error_summary_rel.csv", "a", newline="") as f:
            csv.writer(f).writerow([outprefix, L2_rel, Linf_rel_dof, rel_tol])

    if domain.comm.rank == 0:
        print(f"[GLOBAL] ||e||_L2      = {metrics['L2']:.6e}")
        print(f"[GLOBAL] |e|_H1        = {metrics['H1_seminorm']:.6e}")
        print(f"[GLOBAL] ||e||_Linf(d) = {metrics['Linf_dof']:.6e}")
        print("[HOTSPOT] local cell id =", metrics["max_cell"]["local_id"])
        print("[HOTSPOT] centroid xyz  =", metrics["max_cell"]["centroid"])
        print("[HOTSPOT] L2(cell)      =", metrics["max_cell"]["L2_cell_norm"])
        # CSV log for Leah
        import csv
        cent = metrics["max_cell"]["centroid"]
        cx, cy, cz = (cent.tolist() if cent is not None else [None, None, None])
        with open("results/error_summary.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                outprefix, metrics["L2"], metrics["H1_seminorm"], metrics["Linf_dof"],
                cx, cy, cz, metrics["max_cell"]["L2_cell_norm"],
            ])

    # write solution once per run
    os.makedirs(os.path.dirname(outprefix), exist_ok=True)
    with io.XDMFFile(domain.comm, f"{outprefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain); xdmf.write_function(uh)

    # probe comparison on line (|x|<=2a, avoid edges)
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
