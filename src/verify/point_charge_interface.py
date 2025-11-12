#!/usr/bin/env python3
# DOLFINx 0.10 verifier: empty box, uniform dielectric + point charge,
# two-layer interface + point charge, with E, D, Dz projections, line probes,
# materials tags, optional asymmetric mesh-by-counts, FEM slices, AND
# Jackson analytic slices written into the SAME XDMF for easy overlay.

import json
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl

COMM = MPI.COMM_WORLD
EPS0 = 8.8541878128e-12

# ---------- Mesh builders ----------
def build_box(Lx, Ly, H, h):
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, -H/2], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H/2], dtype=np.double)
    return mesh.create_box(COMM, [p0, p1], (nx, ny, nz), cell_type=mesh.CellType.tetrahedron)

def build_box_counts(Lx, Ly, Lz_top, Lz_bot, nx, ny, nz_top, nz_bot):
    p0 = np.array([-Lx/2, -Ly/2, -Lz_bot], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  Lz_top], dtype=np.double)
    m_bot = mesh.create_box(COMM, [p0, np.array([ Lx/2,  Ly/2, 0.0], dtype=np.double)],
                            (nx, ny, nz_bot), cell_type=mesh.CellType.tetrahedron)
    m_top = mesh.create_box(COMM, [np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double), p1],
                            (nx, ny, nz_top), cell_type=mesh.CellType.tetrahedron)
    cdim = m_bot.topology.dim
    conn_bot = m_bot.topology.connectivity(cdim, 0).array.reshape(-1, 4)
    conn_top = m_top.topology.connectivity(cdim, 0).array.reshape(-1, 4) + m_bot.geometry.x.shape[0]
    cells = np.vstack((conn_bot, conn_top)).astype(np.int64)
    x = np.vstack((m_bot.geometry.x, m_top.geometry.x))
    from dolfinx.mesh import create_mesh
    return create_mesh(COMM, cells, x, mesh.CellType.tetrahedron)

# ---------- εr(x) constructors ----------
def make_eps_r_constant(domain, epsr):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")
    eps.x.array[:] = epsr
    return eps

def make_eps_r_two_layer(domain, epsr_top, epsr_bot):
    """DG0 εr split at z=0; return (eps, materials MeshTags: 1=top, 2=bottom)."""
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    x = domain.geometry.x
    ncell = domain.topology.index_map(tdim).size_local
    cells = np.arange(ncell, dtype=np.int32)

    from dolfinx.cpp.mesh import cell_num_vertices
    nv = cell_num_vertices(domain.topology.cell_type)
    conn = domain.topology.connectivity(tdim, 0).array.reshape(-1, nv)
    cz = x[conn].mean(axis=1)[:, 2]

    top_mask = cz >= 0.0
    vals = np.empty_like(cz, dtype=float)
    vals[top_mask]  = epsr_top
    vals[~top_mask] = epsr_bot
    eps.x.array[:ncell] = vals  # DG0 one dof per cell

    from dolfinx.mesh import meshtags
    cell_ids = np.concatenate((cells[top_mask], cells[~top_mask]))
    tags     = np.concatenate((np.full(np.count_nonzero(top_mask),  1, dtype=np.int32),
                               np.full(np.count_nonzero(~top_mask), 2, dtype=np.int32)))
    ctags = meshtags(domain, tdim, cell_ids, tags)
    ctags.name = "materials"
    return eps, ctags

def make_eps_r_smooth(domain, eps_top, eps_bot, z0=0.0, width=2e-8):
    if width <= 0:
        raise ValueError("width must be > 0 for a smooth transition")
    W = fem.functionspace(domain, ("Lagrange", 1))
    eps = fem.Function(W, name="eps_r")
    def _eps(X):
        z = X[2]
        S = 0.5*(1.0 + np.tanh((z - z0)/width))
        return eps_bot + (eps_top - eps_bot)*S
    eps.interpolate(_eps)
    return eps

def make_eps_r_nlayer(domain, z_edges, eps_list):
    z_edges = np.asarray(z_edges, dtype=float)
    eps_list = np.asarray(eps_list, dtype=float)
    if z_edges.ndim != 1 or eps_list.ndim != 1:
        raise ValueError("z_edges and eps_list must be 1D sequences")
    if len(z_edges) < 2 or not np.all(np.diff(z_edges) > 0):
        raise ValueError("z_edges must be strictly increasing with length ≥ 2")
    if len(eps_list) != len(z_edges) - 1:
        raise ValueError("eps_list must have length len(z_edges)-1")
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")
    z_lo = z_edges[:-1]
    z_hi = z_edges[1:]
    def _eps(X):
        z = X[2]
        out = np.full_like(z, eps_list[-1], dtype=float)
        for e_lo, e_hi, val in zip(z_lo, z_hi, eps_list):
            mask = (z >= e_lo) & (z < e_hi)
            out[mask] = val
        out[z == z_edges[-1]] = eps_list[-1]
        return out
    eps.interpolate(_eps)
    return eps

# ---------- ρ(x) ----------
def gaussian_rho(domain, q, x0, sigma):
    x0 = np.asarray(x0, dtype=float)
    pref = q / (np.pi**1.5 * sigma**3)
    def _rho(X):
        dx = X[0] - x0[0]; dy = X[1] - x0[1]; dz = X[2] - x0[2]
        r2 = dx*dx + dy*dy + dz*dz
        return pref * np.exp(-r2/(sigma*sigma))
    V0 = fem.functionspace(domain, ("Lagrange", 1))
    rho = fem.Function(V0, name="rho")
    rho.interpolate(_rho)
    return rho

# ---------- Boundary conditions ----------
def dirichlet_zero_top_bottom(V, ztol=1e-14):
    dom = V.mesh
    fdim = dom.topology.dim - 1
    dom.topology.create_connectivity(fdim, dom.topology.dim)

    Z_local = dom.geometry.x[:, 2]
    zmin_local = float(Z_local.min())
    zmax_local = float(Z_local.max())
    zmin = dom.comm.allreduce(zmin_local, op=MPI.MIN)
    zmax = dom.comm.allreduce(zmax_local, op=MPI.MAX)

    def at_zmin(x): return np.isclose(x[2], zmin, atol=ztol)
    def at_zmax(x): return np.isclose(x[2], zmax, atol=ztol)

    fmin = mesh.locate_entities_boundary(dom, fdim, at_zmin)
    fmax = mesh.locate_entities_boundary(dom, fdim, at_zmax)
    dofs_min = fem.locate_dofs_topological(V, fdim, fmin)
    dofs_max = fem.locate_dofs_topological(V, fdim, fmax)
    bc_min = fem.dirichletbc(PETSc.ScalarType(0.0), np.unique(dofs_min), V)
    bc_max = fem.dirichletbc(PETSc.ScalarType(0.0), np.unique(dofs_max), V)
    return [bc_min, bc_max]

# ---------- Solve ∇·(ε∇φ)=ρ ----------
def solve_poisson(V, bcs, eps_r_field, rho=None, eps0=EPS0):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = (eps_r_field*eps0) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = (rho * v * ufl.dx) if rho is not None \
        else fem.Constant(V.mesh, PETSc.ScalarType(0.0)) * v * ufl.dx
    problem = LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-9,
            "ksp_error_if_not_converged": True
        },
        petsc_options_prefix="lp_"
    )
    uh = problem.solve()
    uh.name = "phi"
    if V.mesh.comm.rank == 0:
        arr = uh.x.array
        print(f"[SOLVE] φ: min={arr.min():.6e} max={arr.max():.6e} (n={arr.size})")
    return uh

# ---------- FEM projections ----------
def project_scalar(domain, expr, name, qdeg=4):
    V1 = fem.functionspace(domain, ("Lagrange", 1))
    uT, v = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    dxm = ufl.dx(domain, metadata={"quadrature_degree": qdeg})
    aP = ufl.inner(uT, v) * dxm
    bP = ufl.inner(expr, v) * dxm
    w = LinearProblem(aP, bP,
                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                      petsc_options_prefix="prj_").solve()
    w.name = name
    return w

def project_vector(domain, expr, name, qdeg=4):
    dim = domain.geometry.dim
    Vv = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
    uT, v = ufl.TrialFunction(Vv), ufl.TestFunction(Vv)
    dxm = ufl.dx(domain, metadata={"quadrature_degree": qdeg})
    aP = ufl.inner(uT, v) * dxm
    bP = ufl.inner(expr, v) * dxm
    w = LinearProblem(aP, bP,
                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                      petsc_options_prefix="prjv_").solve()
    w.name = name
    return w

# ---------- Point evaluation + 2D FEM slices ----------
def _eval_func_at_points(u: fem.Function, Xq: np.ndarray):
    msh = u.function_space.mesh
    tdim = msh.topology.dim
    tree = geometry.BoundingBoxTree(msh, tdim)
    cand = geometry.compute_collisions(tree, Xq)
    coll = geometry.compute_colliding_cells(msh, cand, Xq)
    cells = np.array([cc[0] if len(cc) > 0 else -1 for cc in coll], dtype=np.int32)
    bad = cells < 0
    if np.any(bad):
        raise RuntimeError(f"Slice points outside mesh: {np.count_nonzero(bad)}")
    vals = np.zeros((Xq.shape[0],) + u.function_space.element.value_shape(), dtype=float)
    u.eval(vals, Xq, cells)
    return vals

def write_slice_xy(domain, uh, E_proj, z0, out_xdmf, name_suffix, nx2=200, ny2=200):
    X = domain.geometry.x
    xmin = domain.comm.allreduce(float(X[:,0].min()), op=MPI.MIN)
    xmax = domain.comm.allreduce(float(X[:,0].max()), op=MPI.MAX)
    ymin = domain.comm.allreduce(float(X[:,1].min()), op=MPI.MIN)
    ymax = domain.comm.allreduce(float(X[:,1].max()), op=MPI.MAX)
    xs = np.linspace(xmin, xmax, nx2); ys = np.linspace(ymin, ymax, ny2)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = np.full_like(XX, z0)
    Xq = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    phi_vals = _eval_func_at_points(uh, Xq).reshape(ny2, nx2)
    E_vals   = _eval_func_at_points(E_proj, Xq).reshape(ny2, nx2, 3)
    from dolfinx.mesh import create_rectangle
    m2 = create_rectangle(COMM, [np.array([xmin, ymin]), np.array([xmax, ymax])],
                          [nx2-1, ny2-1], mesh.CellType.quadrilateral)
    V2s = fem.functionspace(m2, ("Lagrange", 1))
    V2v = fem.functionspace(m2, ("Lagrange", 1, (3,)))
    phi2 = fem.Function(V2s, name=f"phi_slice_{name_suffix}")
    E2   = fem.Function(V2v, name=f"E_slice_{name_suffix}")
    phi2.x.array[:] = phi_vals.ravel(order="C")
    E2.x.array[:]   = E_vals.reshape(-1, 3, order="C").ravel()
    with io.XDMFFile(m2.comm, out_xdmf, "a") as xf:
        xf.write_mesh(m2); xf.write_function(phi2); xf.write_function(E2)

def write_slice_xz(domain, uh, E_proj, y0, out_xdmf, name_suffix, nx2=200, nz2=200):
    X = domain.geometry.x
    xmin = domain.comm.allreduce(float(X[:,0].min()), op=MPI.MIN)
    xmax = domain.comm.allreduce(float(X[:,0].max()), op=MPI.MAX)
    zmin = domain.comm.allreduce(float(X[:,2].min()), op=MPI.MIN)
    zmax = domain.comm.allreduce(float(X[:,2].max()), op=MPI.MAX)
    xs = np.linspace(xmin, xmax, nx2); zs = np.linspace(zmin, zmax, nz2)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    YY = np.full_like(XX, y0)
    Xq = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    phi_vals = _eval_func_at_points(uh, Xq).reshape(nz2, nx2)
    E_vals   = _eval_func_at_points(E_proj, Xq).reshape(nz2, nx2, 3)
    from dolfinx.mesh import create_rectangle
    m2 = create_rectangle(COMM, [np.array([xmin, zmin]), np.array([xmax, zmax])],
                          [nx2-1, nz2-1], mesh.CellType.quadrilateral)
    V2s = fem.functionspace(m2, ("Lagrange", 1))
    V2v = fem.functionspace(m2, ("Lagrange", 1, (3,)))
    phi2 = fem.Function(V2s, name=f"phi_slice_{name_suffix}")
    E2   = fem.Function(V2v,  name=f"E_slice_{name_suffix}")
    phi2.x.array[:] = phi_vals.ravel(order="C")
    E2.x.array[:]   = E_vals.reshape(-1, 3, order="C").ravel()
    with io.XDMFFile(m2.comm, out_xdmf, "a") as xf:
        xf.write_mesh(m2); xf.write_function(phi2); xf.write_function(E2)

# ---------- Analytic (axis and slices, Jackson method-of-images) ----------
def phi_image_axis(z, a, q, eps1_abs, eps2_abs):
    q_img = (eps1_abs - eps2_abs) / (eps1_abs + eps2_abs) * q
    q_dbl = (2 * eps2_abs) / (eps1_abs + eps2_abs) * q
    out = np.empty_like(z, dtype=float)
    sel1 = z >= 0.0
    if np.any(sel1):
        r1 = np.abs(z[sel1] - a); r2 = np.abs(z[sel1] + a)
        out[sel1] = (q/(4*np.pi*eps1_abs*r1)) + (q_img/(4*np.pi*eps1_abs*r2))
    sel2 = ~sel1
    if np.any(sel2):
        r1 = np.abs(z[sel2] - a)
        out[sel2] = (q_dbl/(4*np.pi*eps2_abs*r1))
    return out

def write_jackson_analytic_slices(eps1_abs, eps2_abs, q, a, out_xdmf,
                                  xlim, ylim, zlim,
                                  nx2=200, ny2=200, nz2=200, eps=1e-12):
    from dolfinx import mesh as _mesh, fem as _fem, io as _io
    import numpy as _np

    def phi_E_top(x, y, z):  # z >= 0 (medium 1)
        rho2 = x*x + y*y
        R1 = _np.sqrt(rho2 + (z - a)**2) + eps
        R2 = _np.sqrt(rho2 + (z + a)**2) + eps
        qp = (eps1_abs - eps2_abs)/(eps1_abs + eps2_abs) * q
        phi = (q /(4*_np.pi*eps1_abs))/R1 + (qp/(4*_np.pi*eps1_abs))/R2
        Ex = -(q /(4*_np.pi*eps1_abs)) * (x)/R1**3 - (qp/(4*_np.pi*eps1_abs)) * (x)/R2**3
        Ey = -(q /(4*_np.pi*eps1_abs)) * (y)/R1**3 - (qp/(4*_np.pi*eps1_abs)) * (y)/R2**3
        Ez = -(q /(4*_np.pi*eps1_abs)) * ((z-a))/R1**3 - (qp/(4*_np.pi*eps1_abs)) * ((z+a))/R2**3
        return phi, _np.stack([Ex, Ey, Ez], axis=-1)

    def phi_E_bot(x, y, z):  # z <= 0 (medium 2)
        rho2 = x*x + y*y
        R1 = _np.sqrt(rho2 + (z - a)**2) + eps
        qpp = (2*eps2_abs)/(eps1_abs + eps2_abs) * q
        phi = (qpp/(4*_np.pi*eps2_abs))/R1
        Ex = -(qpp/(4*_np.pi*eps2_abs)) * (x)/R1**3
        Ey = -(qpp/(4*_np.pi*eps2_abs)) * (y)/R1**3
        Ez = -(qpp/(4*_np.pi*eps2_abs)) * ((z-a))/R1**3
        return phi, _np.stack([Ex, Ey, Ez], axis=-1)

    def write_xy_slice(z0, name):
        xs = _np.linspace(xlim[0], xlim[1], nx2)
        ys = _np.linspace(ylim[0], ylim[1], ny2)
        XX, YY = _np.meshgrid(xs, ys, indexing="xy")
        ZZ = _np.full_like(XX, z0)
        if z0 >= 0.0:
            ph, Ev = phi_E_top(XX, YY, ZZ)
        else:
            ph, Ev = phi_E_bot(XX, YY, ZZ)

        m2 = _mesh.create_rectangle(COMM, [_np.array([xlim[0], ylim[0]]),
                                           _np.array([xlim[1], ylim[1]])],
                                    [nx2-1, ny2-1], _mesh.CellType.quadrilateral)
        V2s = _fem.functionspace(m2, ("Lagrange", 1))
        V2v = _fem.functionspace(m2, ("Lagrange", 1, (3,)))
        phi2 = _fem.Function(V2s, name=f"phi_Jackson_{name}")
        E2   = _fem.Function(V2v, name=f"E_Jackson_{name}")
        phi2.x.array[:] = ph.ravel(order="C")
        E2.x.array[:]   = Ev.reshape(-1, 3, order="C").ravel()
        with _io.XDMFFile(m2.comm, out_xdmf, "a") as xf:
            xf.write_mesh(m2); xf.write_function(phi2); xf.write_function(E2)

    def write_xz_slice(y0, name):
        xs = _np.linspace(xlim[0], xlim[1], nx2)
        zs = _np.linspace(zlim[0], zlim[1], nz2)
        XX, ZZ = _np.meshgrid(xs, zs, indexing="xy")
        YY = _np.full_like(XX, y0)
        phi = _np.zeros_like(XX)
        E   = _np.zeros(XX.shape + (3,))
        mask = ZZ >= 0.0
        if mask.any():
            ph, Ev = phi_E_top(XX[mask], YY[mask], ZZ[mask])
            phi[mask], E[mask] = ph, Ev
        if (~mask).any():
            ph, Ev = phi_E_bot(XX[~mask], YY[~mask], ZZ[~mask])
            phi[~mask], E[~mask] = ph, Ev

        m2 = _mesh.create_rectangle(COMM, [_np.array([xlim[0], zlim[0]]),
                                           _np.array([xlim[1], zlim[1]])],
                                    [nx2-1, nz2-1], _mesh.CellType.quadrilateral)
        V2s = _fem.functionspace(m2, ("Lagrange", 1))
        V2v = _fem.functionspace(m2, ("Lagrange", 1, (3,)))
        phi2 = _fem.Function(V2s, name=f"phi_Jackson_{name}")
        E2   = _fem.Function(V2v, name=f"E_Jackson_{name}")
        phi2.x.array[:] = phi.ravel(order="C")
        E2.x.array[:]   = E.reshape(-1, 3, order="C").ravel()
        with _io.XDMFFile(m2.comm, out_xdmf, "a") as xf:
            xf.write_mesh(m2); xf.write_function(phi2); xf.write_function(E2)

    write_xy_slice(a,      "z_eq_d")
    write_xy_slice(+1e-12, "z_eq_0p")
    write_xz_slice(0.0,    "y_eq_0")

# ---------- Probe along an axis using DOF coordinates ----------
def sample_axis(uh, axis="z", tol=None):
    V = uh.function_space
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    U = uh.x.array
    if X.shape[0] > 3:
        diffs = np.diff(np.unique(X, axis=0), axis=0)
        h_est = float(np.median(np.linalg.norm(diffs, axis=1))) if diffs.size else 1e-12
    else:
        h_est = 1e-12
    tol = max(h_est*0.5, 1e-12) if tol is None else tol
    if axis == "z":
        mask = (np.abs(X[:,0]) <= tol) & (np.abs(X[:,1]) <= tol)
        coord = X[mask, 2]
    elif axis == "x":
        mask = (np.abs(X[:,1]) <= tol) & (np.abs(X[:,2]) <= tol)
        coord = X[mask, 0]
    else:
        raise ValueError("axis must be 'z' or 'x'")
    vals = U[mask]
    order = np.argsort(coord)
    return coord[order], vals[order]

# ---------- Run ----------
def run(scene, Lx, Ly, H, h, eps1, eps2, q, a, sigma, deg, out, probe_axis,
        eps_mode, z0, width, z_edges, eps_list,
        use_counts, Lz_top, Lz_bot, nx, ny, nz_top, nz_bot):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    if use_counts:
        domain = build_box_counts(Lx, Ly, Lz_top, Lz_bot, nx, ny, nz_top, nz_bot)
    else:
        domain = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", deg))

    # physics by scene
    rho = None
    materials = None
    if scene == "empty":
        eps_field = make_eps_r_constant(domain, eps1)
    elif scene == "uniform":
        eps_field = make_eps_r_constant(domain, eps1)
        rho = gaussian_rho(domain, q, (0.0, 0.0, a), sigma)
    elif scene == "two":
        if eps_mode == "jump":
            eps_field, materials = make_eps_r_two_layer(domain, epsr_top=eps1, epsr_bot=eps2)
        elif eps_mode == "smooth":
            eps_field = make_eps_r_smooth(domain, eps_top=eps1, eps_bot=eps2, z0=z0, width=width)
        elif eps_mode == "nlayer":
            eps_field = make_eps_r_nlayer(domain, z_edges=z_edges, eps_list=eps_list)
        else:
            raise ValueError("eps_mode must be one of: jump, smooth, nlayer")
        rho = gaussian_rho(domain, q, (0.0, 0.0, a), sigma)
    else:
        raise ValueError("scene must be one of: empty, uniform, two")

    # solve with Dirichlet on z-faces only
    bcs = dirichlet_zero_top_bottom(V)
    uh = solve_poisson(V, bcs, eps_field, rho)

    # Derived fields (projected for ParaView)
    grad_phi = ufl.grad(uh)
    E_vec = -grad_phi
    D_vec = (eps_field * EPS0) * E_vec
    Dz    = ufl.inner(D_vec, ufl.as_vector((0.0, 0.0, 1.0)))
    E_proj  = project_vector(domain, E_vec, "E")
    D_proj  = project_vector(domain, D_vec, "D")
    Dz_proj = project_scalar(domain, Dz, "D_z")

    # write mesh + tags + fields
    with io.XDMFFile(domain.comm, f"{out}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        if materials is not None:
            xf.write_meshtags(materials)
        xf.write_function(eps_field)   # eps_r
        xf.write_function(uh)          # phi
        xf.write_function(E_proj)      # E
        xf.write_function(D_proj)      # D
        xf.write_function(Dz_proj)     # D_z
    if domain.comm.rank == 0:
        print(f"[WRITE] {out}.xdmf (mesh, materials?, eps_r, phi, E, D, D_z)")

    # 2D FEM diagnostic slices (append to same XDMF)
    if domain.comm.rank == 0:
        print("[SLICE] writing z=d, z=0+, and y=0 FEM slices")
    write_slice_xy(domain, uh, E_proj, z0=a,        out_xdmf=f"{out}.xdmf", name_suffix="z_eq_d")
    write_slice_xy(domain, uh, E_proj, z0=+1e-12,   out_xdmf=f"{out}.xdmf", name_suffix="z_eq_0p")
    write_slice_xz(domain, uh, E_proj, y0=0.0,      out_xdmf=f"{out}.xdmf", name_suffix="y_eq_0")

    # line probe
    if probe_axis not in ("z", "x"):
        probe_axis = "z"
    coord, phi_fe = sample_axis(uh, axis=probe_axis)
    if domain.comm.rank == 0:
        csv_path = f"{out}_line_{probe_axis}.csv"
        with open(csv_path, "w") as f:
            f.write(f"{probe_axis},phi\n")
            for c, u in zip(coord, phi_fe):
                f.write(f"{c:.16e},{u:.16e}\n")
        print(f"[LINE] wrote {csv_path}")

    # analytic overlay for two-layer jump: axis CSV + JACKSON slices
    if scene == "two" and eps_mode == "jump" and domain.comm.rank == 0:
        # Axis (CSV) overlay
        if probe_axis == "z":
            phi_ana = phi_image_axis(coord, a=a, q=q, eps1_abs=eps1*EPS0, eps2_abs=eps2*EPS0)
            with open(f"{out}_line_z_analytic.csv", "w") as f:
                f.write("z,phi_analytic\n")
                for z, ua in zip(coord, phi_ana):
                    f.write(f"{z:.16e},{ua:.16e}\n")
            print(f"[ANALYTIC] wrote {out}_line_z_analytic.csv")

        # Slices (Jackson) overlay
        X = domain.geometry.x
        xmn = domain.comm.allreduce(float(X[:,0].min()), op=MPI.MIN)
        xmx = domain.comm.allreduce(float(X[:,0].max()), op=MPI.MAX)
        ymn = domain.comm.allreduce(float(X[:,1].min()), op=MPI.MIN)
        ymx = domain.comm.allreduce(float(X[:,1].max()), op=MPI.MAX)
        zmn = domain.comm.allreduce(float(X[:,2].min()), op=MPI.MIN)
        zmx = domain.comm.allreduce(float(X[:,2].max()), op=MPI.MAX)
        write_jackson_analytic_slices(
            eps1_abs=eps1*EPS0, eps2_abs=eps2*EPS0, q=q, a=a,
            out_xdmf=f"{out}.xdmf",
            xlim=(xmn, xmx), ylim=(ymn, ymx), zlim=(zmn, zmx),
            nx2=200, ny2=200, nz2=200
        )
        print("[ANALYTIC] Jackson slices appended (phi_Jackson_*, E_Jackson_*)")

    # metadata
    if domain.comm.rank == 0:
        meta = dict(scene=scene,
                    use_counts=use_counts, Lx=Lx, Ly=Ly, H=H, h=h,
                    Lz_top=Lz_top, Lz_bot=Lz_bot, nx=nx, ny=ny, nz_top=nz_top, nz_bot=nz_bot,
                    deg=deg, eps1=eps1, eps2=eps2, q=q, a=a, sigma=sigma, probe=probe_axis,
                    eps_mode=eps_mode, z0=z0, width=width,
                    z_edges=(z_edges.tolist() if isinstance(z_edges, np.ndarray) else z_edges),
                    eps_list=(eps_list.tolist() if isinstance(eps_list, np.ndarray) else eps_list))
        Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[DONE] Outputs under {out.parent}")

# ---------- CLI ----------
def _parse():
    import argparse, json as _json
    p = argparse.ArgumentParser(description="Point charge & dielectric interface verifier (DOLFINx 0.10)")
    p.add_argument("--scene", choices=["empty", "uniform", "two"], required=True)

    # eps with aliases (use either --eps1/--eps2 or --epsr1/--epsr2)
    p.add_argument("--eps1",  type=float, default=None)
    p.add_argument("--eps2",  type=float, default=None)
    p.add_argument("--epsr1", type=float, default=None, help="alias for --eps1")
    p.add_argument("--epsr2", type=float, default=None, help="alias for --eps2")

    p.add_argument("--q",     type=float, default=1.0e-12)
    p.add_argument("--a",     type=float, default=4.0e-8, help="charge height (z=d)")
    p.add_argument("--sigma", type=float, default=3.0e-8)

    # legacy box-by-size
    p.add_argument("--L",     type=float, default=1.0e-6)
    p.add_argument("--Ly",    type=float, default=None)
    p.add_argument("--H",     type=float, default=1.0e-6)
    p.add_argument("--h",     type=float, default=2.0e-8)

    p.add_argument("--deg",   type=int,   default=1)
    p.add_argument("--out",   type=str,   required=True)
    p.add_argument("--probe", choices=["z", "x"], default="z")

    # ε mode + params
    p.add_argument("--eps_mode", choices=["jump", "smooth", "nlayer"], default="jump")
    p.add_argument("--z0",    type=float, default=0.0,   help="center z for smooth tanh")
    p.add_argument("--width", type=float, default=2.0e-8, help="tanh half-width for smooth")
    p.add_argument("--z_edges", type=str, default=None,
                   help="JSON list of z edges for nlayer, e.g. '[-5e-7,0,5e-8,5e-7]'")
    p.add_argument("--eps_list", type=str, default=None,
                   help="JSON list of eps for nlayer, e.g. '[11.7,7.5,3.9]'")

    # Asymmetric mesh-by-counts (optional)
    p.add_argument("--use_counts", action="store_true",
                   help="Use (Lx,Ly,Lz_top,Lz_bot,nx,ny,nz_top,nz_bot) instead of (L,H,h).")
    p.add_argument("--Lx", type=float, default=None)
    p.add_argument("--Ly2", type=float, default=None, help="alias for Ly when using counts")
    p.add_argument("--Lz_top", type=float, default=None)
    p.add_argument("--Lz_bot", type=float, default=None)
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nz_top", type=int, default=None)
    p.add_argument("--nz_bot", type=int, default=None)

    # Timestamping
    p.add_argument("--stamp", action="store_true", help="append YYYYmmdd_%H%M%S to --out")

    args = p.parse_args()

    # counts → require full set; set Ly if missing
    if getattr(args, "use_counts", False):
        for k in ["Lx","Lz_top","Lz_bot","nx","ny","nz_top","nz_bot"]:
            if getattr(args, k) is None:
                raise SystemExit(f"--use_counts requires --{k}")
        if args.Ly2 is not None:
            args.Ly = args.Ly2
        elif args.Ly is None:
            args.Ly = args.Lx

    # Resolve eps aliases and defaults
    if args.eps1 is None and args.epsr1 is not None:
        args.eps1 = args.epsr1
    if args.eps2 is None and args.epsr2 is not None:
        args.eps2 = args.epsr2
    if args.eps1 is None: args.eps1 = 3.9
    if args.eps2 is None: args.eps2 = 11.7

    # Parse nlayer arrays if provided
    if args.eps_mode == "nlayer":
        if args.z_edges is None or args.eps_list is None:
            raise SystemExit("For --eps_mode nlayer you must pass both --z_edges and --eps_list.")
        try:
            args.z_edges = np.array(json.loads(args.z_edges), dtype=float)
            args.eps_list = np.array(json.loads(args.eps_list), dtype=float)
        except Exception as e:
            raise SystemExit(f"Failed to parse --z_edges/--eps_list as JSON arrays: {e}")
    else:
        args.z_edges = np.array([], dtype=float)
        args.eps_list = np.array([], dtype=float)

    # counts: finalize geometry fields for run()
    if args.use_counts:
        # prefer Ly2, else Ly, else Lx
        if args.Ly2 is not None:
            Ly_counts = args.Ly2
        elif args.Ly is not None:
            Ly_counts = args.Ly
        else:
            Ly_counts = args.Lx
        args.Lx, args.Ly, args.H, args.h = float(args.Lx), float(Ly_counts), None, None
    else:
        if args.Ly is None:
            args.Ly = args.L

    return args

# ---------- Entrypoint ----------
if __name__ == "__main__":
    from datetime import datetime
    args = _parse()
    if args.stamp:
        args.out = f"{args.out}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run(args.scene, args.Lx if args.use_counts else args.L,
        args.Ly, args.H, args.h,
        args.eps1, args.eps2, args.q, args.a, args.sigma,
        args.deg, args.out, args.probe,
        args.eps_mode, args.z0, args.width, args.z_edges, args.eps_list,
        args.use_counts,
        args.Lz_top if args.use_counts else None,
        args.Lz_bot if args.use_counts else None,
        args.nx if args.use_counts else None,
        args.ny if args.use_counts else None,
        args.nz_top if args.use_counts else None,
        args.nz_bot if args.use_counts else None)
