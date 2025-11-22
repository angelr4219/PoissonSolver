#!/usr/bin/env python3
"""
Rectangular-gate Laplace benchmark with spatially varying permittivity
- 3D box: x,y in [-L/2, L/2], z in [0, H]
- Top (z≈0): three rectangular Dirichlet patches at voltages Vs; rest of top is 0 V
- Bottom (z≈H): 0 V
- Equation: ∇·(ε∇φ)=0 with ε = ε0 * εr(x) (constant or two-layer)
- Writes:
    <tag>.xdmf           -> CG1 'phi' and CG1 helper 'phi_top_helper' (and optional 'eps_r')
    <tag>_deg0proj.xdmf  -> DG0 'phi_dg0' projection (for quick per-cell viewing)
    <tag>_relerr.xdmf    -> DG0 'err_rel' (if --relerr 1), vs analytic gates (free-space)
    <tag>_relerr_ref.xdmf-> DG0 'err_rel_ref' (if --relerr_ref 1), vs higher-order FEM reference
    <tag>_line.csv       -> 1D probe vs analytic (y=0, z=a)
    <tag>_metrics.json   -> L2/H1-semi/L∞ (vs analytic) and DOFs
    summary_lineprobe.csv (per run)
Outputs go under: results/runs/<YYYYMMDD-HHMMSS>/
"""

import os, json, datetime
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

try:
    from dolfinx_mpc.mpc import MultiPointConstraint
    from dolfinx_mpc.problem import LinearProblem as MPCLinearProblem
    HAS_MPC = True
except Exception:
    try:
        from dolfinx_mpc import MultiPointConstraint, LinearProblem as MPCLinearProblem  # fallback
        HAS_MPC = True
    except Exception as exc:
        print(f"[WARN] dolfinx_mpc not available ({exc}); periodic BCs disabled.")
        HAS_MPC = False



# ---------------------------
# Analytic: rectangular gates (uniform permittivity case)
# ---------------------------
from typing import Sequence, Union

def g_uvz(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    g(u,v,z) = (1/(2π)) * atan2(u*v, z*sqrt(u^2 + v^2 + z^2))
    """
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    denom = z_arr * np.sqrt(u_arr*u_arr + v_arr*v_arr + z_arr*z_arr)
    return (1.0/(2.0*np.pi)) * np.arctan2(u_arr*v_arr, denom)

def phi0_rect(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    a: float,
    xs: Sequence[float],
    Vs: Sequence[float],
) -> Union[float, np.ndarray]:
    """
    φ₀(x,y,z) = − Σ_i V_i [ g(a−x_i+x, a+y, z) + g(a−x_i+x, a−y, z)
                           + g(a+x_i−x, a+y, z) + g(a+x_i−x, a−y, z) ]
    (Eq. 10 form with overall minus sign)
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    # Avoid z=0 singular evaluation by nudging to machine epsilon
    z_arr = np.asarray(z, dtype=float)
    z_arr = np.where(z_arr == 0.0, np.finfo(float).eps, z_arr)

    s = np.zeros_like(x_arr, dtype=float)
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            g_uvz(a - xi + x_arr, a + y_arr, z_arr) +
            g_uvz(a - xi + x_arr, a - y_arr, z_arr) +
            g_uvz(a + xi - x_arr, a + y_arr, z_arr) +
            g_uvz(a + xi - x_arr, a - y_arr, z_arr)
        )
    return -s

# ---------------------------
# Analytic: three rectangular gates (free-space)
# ---------------------------
def _g_uvz(u, v, z):
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def phi0_rect_local(x, y, z, a, xs, Vs):
    z = z if z != 0.0 else np.finfo(float).eps
    s = 0.0
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            _g_uvz(a - xi + x, a + y, z) +
            _g_uvz(a - xi + x, a - y, z) +
            _g_uvz(a + xi - x, a + y, z) +
            _g_uvz(a + xi - x, a - y, z)
        )
    return -s

def get_phi_exact(a, xs_gates, Vs_gates, *, z_fixed=None):
    if z_fixed is None:
        return lambda x, y, z: phi0_rect_local(x, y, z, a, xs_gates, Vs_gates)
    else:
        return lambda x, y, z: phi0_rect_local(x, y, z_fixed, a, xs_gates, Vs_gates)

# ---------------------------
# Mesh/BC helpers
# ---------------------------
def build_box(Lx, Ly, H, h):
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H ], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

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

def build_periodic_mpc(V, bcs, axes: str):
    """
    Create an MPC that enforces periodicity along 'x' and/or 'y' on a box mesh.
    Dirichlet dofs in `bcs` are excluded from constraint creation.
    """
    msh = V.mesh
    tdim = msh.topology.dim
    assert tdim == 3, "This helper assumes a 3D box mesh."

    # Get domain extents from geometry (robust to [-L/2,L/2] coordinates)
    coords = msh.geometry.x
    xmin, xmax = float(coords[:,0].min()), float(coords[:,0].max())
    ymin, ymax = float(coords[:,1].min()), float(coords[:,1].max())
    tol = 250 * np.finfo(PETSc.ScalarType).resolution  # same spirit as demo

    def _make_periodic(axis):
        if axis == "x":
            def on_slave(x):
                return np.isclose(x[0], xmax, atol=tol)
            L = xmax - xmin
            def map_master(x):
                y = np.zeros_like(x)
                y[0] = x[0] - L
                y[1] = x[1]
                y[2] = x[2]
                return y
            return on_slave, map_master
        elif axis == "y":
            def on_slave(x):
                return np.isclose(x[1], ymax, atol=tol)
            L = ymax - ymin
            def map_master(x):
                y = np.zeros_like(x)
                y[0] = x[0]
                y[1] = x[1] - L
                y[2] = x[2]
                return y
            return on_slave, map_master
        else:
            raise ValueError("axis must be 'x' or 'y'")

    mpc = MultiPointConstraint(V)
    if "x" in axes:
        on_slave, map_master = _make_periodic("x")
        mpc.create_periodic_constraint_geometrical(V, on_slave, map_master, bcs)
    if "y" in axes:
        on_slave, map_master = _make_periodic("y")
        mpc.create_periodic_constraint_geometrical(V, on_slave, map_master, bcs)
    mpc.finalize()
    return mpc


# ---------------------------
# εr(x): constant or two-layer
# ---------------------------
def make_eps_r_constant(domain, epsr):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")
    eps.x.array[:] = epsr
    return eps

def make_eps_r_two_layer(domain, t_ox, epsr_top, epsr_bulk):
    """
    DG0 field via interpolation:
      z in [z_min, z_min + t_ox]  -> epsr_top
      z > z_min + t_ox            -> epsr_bulk
    Uses z_min (actual top) to be robust to box placement.
    """
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    eps = fem.Function(W, name="eps_r")

    z_coords = domain.geometry.x[:, 2]
    z_min = domain.comm.allreduce(float(np.min(z_coords)), op=MPI.MIN)
    z_split = z_min + t_ox

    def _eps_callable(X):
        z = X[2]
        return np.where(z <= z_split, float(epsr_top), float(epsr_bulk))

    eps.interpolate(_eps_callable)
    return eps

# ---------------------------
# Solve with ε(x)
# ---------------------------
def solve_laplace(V, bcs, eps_r_field=None, eps0=8.8541878128e-12):
    domain = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    if eps_r_field is None:
        lam = PETSc.ScalarType(11.7 * eps0)
        a_form = lam * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    else:
        a_form = (eps_r_field * eps0) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

<<<<<<< HEAD
    problem = LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10,
                       "ksp_error_if_not_converged": True},
        petsc_options_prefix="lp_"
    )
=======
    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_error_if_not_converged": True
    }
    problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts, petsc_options_prefix="poisson_")
>>>>>>> 7b176cafc436c4f4f7c51f1364ef42c02d769d99
    uh = problem.solve()
    uh.name = "phi"

    uarr = uh.x.array
    if domain.comm.rank == 0:
        n_bad = int(np.sum(~np.isfinite(uarr)))
        print(f"[VOLTAGE] nonfinite={n_bad}, min={np.nanmin(uarr):.6e}, max={np.nanmax(uarr):.6e}")
    return uh

# ---------------------------
# Metrics & utilities
# ---------------------------
def compute_metrics(domain, uh, u_exact, qdeg=4):
    e = uh - u_exact
    dxf = ufl.dx(metadata={"quadrature_degree": qdeg})
    L2_sq  = fem.assemble_scalar(fem.form(e*e*dxf))
    H1s_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e))*dxf))
    L2  = float(np.sqrt(max(L2_sq,  0.0)))
    H1s = float(np.sqrt(max(H1s_sq, 0.0)))
    Linf = float(np.max(np.abs(uh.x.array))) if uh.x.array.size else 0.0
    V = uh.function_space
    dofs = int(V.dofmap.index_map.size_local * V.dofmap.index_map_bs)
    return {"L2": L2, "H1_semi": H1s, "Linf": Linf, "dofs": dofs}

from dolfinx import geometry

def sample_dof_line(uh: fem.Function, zbar: float, a: float, npts: int, h: float):
    """
    Return (x, u(x,0,zbar)) for x in [-2a, 2a].
    First try DOF-coord filtering; if none, evaluate at arbitrary points.
    """
    V = uh.function_space
    msh = V.mesh
    Xdof = V.tabulate_dof_coordinates().reshape((-1, 3))
    U = uh.x.array

    # --- fast path: harvest existing dofs near the plane
    ztol = max(2*h, 1e-12)               # relax tolerance a bit
    mask = (np.abs(Xdof[:, 1]) <= 1e-12) & (np.abs(Xdof[:, 2] - zbar) <= ztol)
    if np.any(mask):
        xs = Xdof[mask, 0]; us = U[mask]
        order = np.argsort(xs)
        return xs[order], us[order]

    # --- robust path: evaluate at arbitrary points (x,0,zbar)
    xs = np.linspace(-2*a, 2*a, npts)
    P = np.stack([xs, np.zeros_like(xs), np.full_like(xs, zbar)], axis=1)  # shape (npts, 3)

    # Build a bounding box tree and find containing cells for each point
    tdim = msh.topology.dim
    tree = geometry.BoundingBoxTree(msh, tdim, msh.geometry.x)
    cells = np.empty(npts, dtype=np.int32)

    for i, p in enumerate(P):
        # Candidates that might contain p
        candidates = geometry.compute_collisions(tree, p)
        colliding = geometry.compute_colliding_cells(msh, candidates, p)
        if len(colliding) == 0:
            # If the exact plane is just outside, nudge zbar slightly toward interior
            p_ = p.copy()
            p_[2] = float(p_[2] + np.sign(p_[2] - msh.geometry.x[:,2].mean()) * 1e-12)
            candidates = geometry.compute_collisions(tree, p_)
            colliding = geometry.compute_colliding_cells(msh, candidates, p_)
            if len(colliding) == 0:
                raise RuntimeError("Probe point not inside the mesh. Try a slightly different --probe_z.")
        cells[i] = colliding[0]

    # Evaluate uh at those points
    vals = uh.eval(P, cells)  # returns shape (npts, 1) for scalar
    return xs, vals[:, 0]


def write_all_in_one(domain, phi, pads_on_top, outprefix, eps_r_field=None):
    V1 = fem.functionspace(domain, ("Lagrange", 1))
    phi1 = fem.Function(V1, name="phi"); phi1.interpolate(phi)
    out = f"{outprefix}.xdmf"
    with io.XDMFFile(domain.comm, out, "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(phi1)
        xf.write_function(pads_on_top)
        if eps_r_field is not None:
            xf.write_function(eps_r_field)
    if domain.comm.rank == 0:
        print(f"[WRITE] {out} (phi, phi_top_helper{', eps_r' if eps_r_field is not None else ''})")

def write_dg0_projection(domain, uh, outprefix):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    uT, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    a_proj = ufl.inner(uT, v) * ufl.dx
    L_proj = ufl.inner(uh, v) * ufl.dx
    w = LinearProblem(
        a_proj, L_proj,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="proj_"
    ).solve()
    w.name = "phi_dg0"
    with io.XDMFFile(domain.comm, f"{outprefix}_deg0proj.xdmf", "w") as xo:
        xo.write_mesh(domain); xo.write_function(w)
    if domain.comm.rank == 0:
        print(f"[WRITE] {outprefix}_deg0proj.xdmf (phi_dg0)")

def write_relerr_dg0(domain, uh, a, xs_gates, Vs_gates, outprefix, qdeg=4, tau=1e-9):
    """
    Build DG0 relative error vs analytic φ₀ under uniform εr.
    err_rel = |uh - φ₀| / (|φ₀| + tau)
    """
    # Evaluate analytic φ₀ at FE dof coordinates and interpolate into same space as uh
    V = uh.function_space
    uE = fem.Function(V, name="phi_exact")

    def _eval_exact(X):
        # X shape: (3, N)
        return phi0_rect(X[0], X[1], X[2], a, xs_gates, Vs_gates)

    uE.interpolate(_eval_exact)

    # Project per-cell (DG0) relative error for a clean heatmap
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    uT, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    dxm = ufl.dx(metadata={"quadrature_degree": qdeg})

    e = uh - uE
    num = ufl.sqrt(ufl.max_value(e*e, 0.0))
    den = ufl.sqrt(ufl.max_value(uE*uE, 0.0)) + tau
    f_rel = num/den

    w_rel = LinearProblem(
        ufl.inner(uT, v)*dxm, ufl.inner(f_rel, v)*dxm,
        petsc_options={"ksp_type":"preonly","pc_type":"lu"},
        petsc_options_prefix="proj_"
    ).solve()
    w_rel.name = "err_rel"

    with io.XDMFFile(domain.comm, f"{outprefix}_relerr.xdmf", "w") as xo:
        xo.write_mesh(domain)
        xo.write_function(w_rel)
    if domain.comm.rank == 0:
        print(f"[WRITE] {outprefix}_relerr.xdmf (err_rel vs analytic)")


# ---------------------------
# Optional: higher-order FEM reference comparator
# ---------------------------
def make_reference_field(domain, a, xs_gates, Vs_gates, eps_r_field, deg_ref=3):
    Vref = fem.functionspace(domain, ("Lagrange", deg_ref))
    # Rebuild BCs on Vref
    bcs_ref = gate_top_and_bottom_bcs(Vref, a, xs_gates, Vs_gates, rect_tol=1e-8, ztol=1e-9)
    uR, vR = ufl.TrialFunction(Vref), ufl.TestFunction(Vref)
    aR = (eps_r_field * 8.8541878128e-12) * ufl.inner(ufl.grad(uR), ufl.grad(vR)) * ufl.dx
    LR = fem.Constant(domain, PETSc.ScalarType(0.0)) * vR * ufl.dx
    uh_ref = LinearProblem(
        aR, LR, bcs=bcs_ref,
        petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-11,
                       "ksp_error_if_not_converged":True},
        petsc_options_prefix="ref_"
    ).solve()
    uh_ref.name = "phi_ref"
    return uh_ref

def write_relerr_ref_dg0(domain, uh, uh_ref, outprefix, qdeg=4, tau=1e-9):
    # Interpolate reference solution into the same space as uh
    V = uh.function_space
    uE = fem.Function(V, name="phi_exact_ref"); uE.interpolate(uh_ref)

    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    uT, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    dxm = ufl.dx(metadata={"quadrature_degree": qdeg})
    e = uh - uE
    num = ufl.sqrt(ufl.max_value(e*e, 0.0))
    den = ufl.sqrt(ufl.max_value(uE*uE, 0.0)) + tau
    f_rel = num/den
    aP = ufl.inner(uT, v)*dxm
    bP = ufl.inner(f_rel, v)*dxm
    w_rel = LinearProblem(
        aP, bP,
        petsc_options={"ksp_type":"preonly","pc_type":"lu"},
        petsc_options_prefix="proj_"
    ).solve()
    w_rel.name = "err_rel_ref"
    with io.XDMFFile(domain.comm, f"{outprefix}_relerr_ref.xdmf", "w") as xo:
        xo.write_mesh(domain); xo.write_function(w_rel)
    if domain.comm.rank == 0:
        print(f"[WRITE] {outprefix}_relerr_ref.xdmf (err_rel vs reference FEM)")

# ---------------------------
# One case
# ---------------------------
def run_once(RUN_ROOT, Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, nprobe,
             tag, deg, eps_mode, eps_const, eps_tox, eps_top, eps_bulk,
             write_relerr, write_relerr_ref, write_eps, pbc_axes: str = ""):

    outprefix = str(Path(RUN_ROOT) / tag)
    Path(outprefix).parent.mkdir(parents=True, exist_ok=True)

    domain = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", deg))

    # Build εr(x)
    if eps_mode == "two":
        eps_r_field = make_eps_r_two_layer(domain, eps_tox, eps_top, eps_bulk)
    else:
        eps_r_field = make_eps_r_constant(domain, eps_const)

    # BCs and solve
    # Build εr(x)
    # BCs (Dirichlet on top gates + bottom ground)
    bcs = gate_top_and_bottom_bcs(V, a, xs_gates, Vs_gates, rect_tol=1e-8, ztol=1e-9)

    # Assemble & solve (MPC if periodic)
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a_form = ((eps_r_field * 8.8541878128e-12) * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L_form = fem.Constant(V.mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

<<<<<<< HEAD
    if pbc_axes:
        mpc = build_periodic_mpc(V, bcs, pbc_axes)
        problem = MPCLinearProblem(a_form, L_form, mpc, bcs=bcs,
                                    petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10,
                                                "ksp_error_if_not_converged":True},
                                    petsc_options_prefix="lp_")
        uh = problem.solve()
    else:
        problem = LinearProblem(a_form, L_form, bcs=bcs,
                                petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10,
                                                "ksp_error_if_not_converged":True},
                                petsc_options_prefix="lp_")
        uh = problem.solve()
=======
    # === Error analysis (global + cellwise fields) ===
    from errors import report_errors
    from exact_rect_gates import phi0_rect_three_gates_factory
    os.makedirs("results", exist_ok=True)
    exact_fun = phi0_rect_three_gates_factory(a, zbar, xs_gates, Vs_gates)
    # ---- Metrics vs analytic (safe for 0.10) ----
    from dolfinx import fem, io
    import ufl, numpy as np

    def _phi0_on_dofs(X):
        # X has shape (3, N) in dolfinx 0.10
        vals = np.empty(X.shape[1], dtype=float)
        for i in range(X.shape[1]):
            vals[i] = phi0_rect_local(X[0, i], X[1, i], X[2, i], a, xs_gates, Vs_gates)
        return vals

    uE = fem.Function(uh.function_space, name="phi_exact")
    uE.interpolate(_phi0_on_dofs)

    # L2 and H1-seminorm of error (assemble_scalar)
    e   = uh - uE
    dxf = ufl.dx(metadata={"quadrature_degree": 4})
    L2_sq  = fem.assemble_scalar(fem.form(e*e*dxf))
    H1s_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e))*dxf))
    L2  = float(np.sqrt(max(L2_sq,  0.0)))
    H1s = float(np.sqrt(max(H1s_sq, 0.0)))
    Linf = float(np.max(np.abs(uh.x.array))) if uh.x.array.size else 0.0

    # Persist a tiny JSON summary
    from pathlib import Path, PurePath
    met_json = f"{outprefix}_metrics.json"
    Path(PurePath(met_json)).write_text(json.dumps(
        {"tag": tag, "degree": deg,
        "dofs": int(uh.function_space.dofmap.index_map.size_local
                    * uh.function_space.dofmap.index_map_bs),
        "l2": L2, "h1s": H1s, "linf": Linf}, indent=2))
    print(f"[METRICS] wrote {met_json}")
    # ---- Optional DG0 relative-error heatmap (err_rel) ----
    from dolfinx.fem.petsc import LinearProblem
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    uT, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    dxm = ufl.dx(metadata={"quadrature_degree": 4})

    tau = 1e-9
    num = ufl.sqrt(ufl.max_value(e*e, 0.0))
    den = ufl.sqrt(ufl.max_value(uE*uE, 0.0)) + tau
    f_rel = num/den

    aP = ufl.inner(uT, v) * dxm
    bP = ufl.inner(f_rel, v) * dxm
    w_rel = LinearProblem(aP, bP,
                        petsc_options={"ksp_type":"preonly","pc_type":"lu"},
                        petsc_options_prefix="proj_").solve()
    w_rel.name = "err_rel"
    with io.XDMFFile(domain.comm, f"{outprefix}_relerr.xdmf", "w") as xo:
        xo.write_mesh(domain); xo.write_function(w_rel)
    print(f"[WRITE] {outprefix}_relerr.xdmf (err_rel vs analytic)")


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
>>>>>>> 7b176cafc436c4f4f7c51f1364ef42c02d769d99

    uh.name = "phi"


    # Helper field with pad voltages at top (for visualization)
    V1 = fem.functionspace(domain, ("Lagrange", 1))
    pads_on_top = fem.Function(V1, name="phi_top_helper")
    X1 = V1.tabulate_dof_coordinates().reshape((-1, 3))
    Z = domain.geometry.x[:, 2]
    z_min = domain.comm.allreduce(float(Z.min()), op=MPI.MIN)
    vals3d = np.full(X1.shape[0], np.nan, dtype=float)
    on_top = np.isclose(X1[:, 2], z_min, atol=1e-9)
    for xi, Vi in zip(xs_gates, Vs_gates):
        in_rect = (
            on_top &
            (X1[:, 0] >= (xi - a)) & (X1[:, 0] <= (xi + a)) &
            (X1[:, 1] >= -a)       & (X1[:, 1] <=  a)
        )
        vals3d[in_rect] = Vi
    pads_on_top.x.array[:] = vals3d

    # Metrics vs analytic (full-field)
    uE = fem.Function(V, name="phi_exact")
    def _eval_exact(X):
        return np.array([phi0_rect_local(X[0,i], X[1,i], X[2,i], a, xs_gates, Vs_gates)
                         for i in range(X.shape[1])], dtype=float)
    uE.interpolate(_eval_exact)
    mets = compute_metrics(domain, uh, uE, qdeg=4)
    if MPI.COMM_WORLD.rank == 0:
        (Path(f"{outprefix}_metrics.json")).write_text(json.dumps({
            "tag": tag, "degree": deg, "dofs": mets["dofs"],
            "l2": mets["L2"], "h1s": mets["H1_semi"], "linf": mets["Linf"]
        }, indent=2))
        print(f"[METRICS] wrote {outprefix}_metrics.json")

    # Writes
    write_all_in_one(domain, uh, pads_on_top, outprefix, eps_r_field if write_eps else None)
    write_dg0_projection(domain, uh, outprefix)

    # --- Part C: only do analytic relative error when εr is uniform ---
    if write_relerr:
        if eps_mode == "const":
            write_relerr_dg0(domain, uh, a, xs_gates, Vs_gates, outprefix, qdeg=4, tau=1e-9)
        else:
            if domain.comm.rank == 0:
                print("[RELERR] Skipping analytic comparison under two-layer εr "
                      "(analytic assumes uniform medium).")

    # Reference-FEM comparison (same physics as current εr/BCs) is always valid
    if write_relerr_ref:
        uh_ref = make_reference_field(domain, a, xs_gates, Vs_gates, eps_r_field, deg_ref=max(2, deg+1))
        write_relerr_ref_dg0(domain, uh, uh_ref, outprefix, qdeg=4, tau=1e-9)

        # --- Line probe (y=0, z=zbar) and edge-safe error vs analytic ---
    xs_line, uh_line = sample_dof_line(uh, zbar, a, nprobe, h)

    # If uniform εr, compare to analytic φ0; otherwise just warn (analytic is free-space)
    if eps_mode == "const":
        phi_exact = get_phi_exact(a, xs_gates, Vs_gates, z_fixed=zbar)
        phi0_line = np.array([phi_exact(x, 0.0, zbar) for x in xs_line])
    else:
        if domain.comm.rank == 0:
            print("[LINE] Two-layer εr: analytic φ0 is not physically matched (free-space formula). "
                  "CSV will still include φ0 for visual reference.")
        phi_exact = get_phi_exact(a, xs_gates, Vs_gates, z_fixed=zbar)
        phi0_line = np.array([phi_exact(x, 0.0, zbar) for x in xs_line])

    # Edge-safe mask to avoid corner singularities
    dx = np.median(np.diff(xs_line)) if xs_line.size > 1 else a*0.01
    band = 2.0*abs(dx)
    edges = np.concatenate([xs_gates - a, xs_gates + a])
    mask = (np.abs(xs_line) <= 2*a)
    for e_edge in edges:
        mask &= (np.abs(xs_line - e_edge) > band)

    diffs = np.abs(uh_line[mask] - phi0_line[mask])
    err_max = float(np.max(diffs)) if diffs.size else float("nan")
    err_l2  = float(np.sqrt(np.mean(diffs**2))) if diffs.size else float("nan")

    if MPI.COMM_WORLD.rank == 0:
        import csv
        csv_path = f"{outprefix}_line.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_m", "phi_FE_V", "phi0_V"])
            for x, uval, aval in zip(xs_line, uh_line, phi0_line):
                w.writerow([x, uval, aval])
        print(f"[LINE] {csv_path}")
        print(f"[{tag}] max|Δφ| (|x|≤2a, edge-safe) = {err_max:.4e} V,  L2 = {err_l2:.4e} V")

    return err_max, err_l2


# ---------------------------
# CLI + main
# ---------------------------
def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(description="Rectangular-gate benchmark with spatial εr")
    p.add_argument("--deg", type=int, default=1, help="Lagrange degree (default 1)")
    p.add_argument("--a", type=float, required=True, help="Gate half-size a (m)")
    p.add_argument("--zfill", type=float, required=True, help="Device thickness H (m)")
    p.add_argument("--xs", type=str, required=True, help='JSON list of gate centers, e.g. "[-7e-8,0,7e-8]"')
    p.add_argument("--Vs", type=str, required=True, help='JSON list of gate voltages, e.g. "[0.25,0.10,0.25]"')
    p.add_argument("--h", type=float, default=5e-9, help="Target element size h (m)")
    p.add_argument("--pad", type=float, default=None, help="Padding p (if omitted: runs p=2..5)")
    p.add_argument("--run-root", type=str, default=None, help="Override run root (default: results/runs/<STAMP>)")
    p.add_argument("--relerr", type=int, default=0, help="Write DG0 relative-error heatmap (0/1) vs analytic")
    p.add_argument("--relerr_ref", type=int, default=0, help="Write DG0 relative-error vs higher-order FEM (0/1)")
    p.add_argument("--write-eps", type=int, default=0, help="Also write eps_r into the XDMF (0/1)")
    
    # in _parse_cli()
    p.add_argument("--probe_z", type=float, default=None, help="Override probe height zbar (m). Default: a")
    p.add_argument("--probe_n", type=int, default=401, help="Number of samples along the probe line (default 401)")

    # εr options
    p.add_argument("--epsr", type=float, default=11.7, help="Uniform relative permittivity (default 11.7)")
    p.add_argument("--two-layer", action="store_true", help="Enable two-layer εr split in z")
    p.add_argument("--t_ox", type=float, default=5e-9, help="Top layer thickness if --two-layer")
    p.add_argument("--epsr_top", type=float, default=3.9, help="Top layer εr (default 3.9)")
    p.add_argument("--epsr_bulk", type=float, default=11.7, help="Bottom layer εr (default 11.7)")
    p.add_argument("--pbc", type=str, default="",
               help='Periodic axes in-plane: "", "x", "y", or "xy" (default "")')

    return p.parse_args()

if __name__ == "__main__":
    args = _parse_cli()
    a = args.a
    zbar = a
    xs_gates = np.array(json.loads(args.xs), dtype=float)
    Vs_gates = np.array(json.loads(args.Vs), dtype=float)
    H = args.zfill
    h = args.h
    deg = args.deg
    write_relerr = bool(args.relerr)
    write_relerr_ref = bool(args.relerr_ref)
    write_eps = bool(args.write_eps)
    eps_mode = "two" if args.two_layer else "const"
    zbar = args.probe_z if args.probe_z is not None else a
    nprobe = args.probe_n
    # --- periodic BC flag ---
    PBC = (args.pbc or "").lower()
    if PBC not in {"", "x", "y", "xy"}:
        raise SystemExit("--pbc must be one of: '', x, y, xy")
    if PBC and not HAS_MPC:
        print("[WARN] --pbc requested but dolfinx_mpc not available; continuing without periodicity.")
        PBC = ""


<<<<<<< HEAD
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_ROOT = Path(args.run_root) if args.run_root else Path("results")/"runs"/stamp
    if MPI.COMM_WORLD.rank == 0:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"==> Run root: {RUN_ROOT}")

    pads = [args.pad] if args.pad is not None else [2.0, 3.0, 4.0, 5.0]

    errs = []
    for p in pads:
        Lx = Ly = 2.0 * p * a
        tag = f"p{int(p)}a_deg{deg}"
        if MPI.COMM_WORLD.rank == 0:
            print(f"---- solving: {tag}  (Lx=Ly={Lx:.3e} m, H={H:.3e} m, h≈{h:.1e}, deg={deg})")
        em, el2 = run_once(
                RUN_ROOT, Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, nprobe,
                tag, deg,
                eps_mode, args.epsr, args.t_ox, args.epsr_top, args.epsr_bulk,
                write_relerr, write_relerr_ref, write_eps,
                pbc_axes=PBC
            )


        errs.append((tag, em, el2))

    if MPI.COMM_WORLD.rank == 0:
        summary_csv = RUN_ROOT / "summary_lineprobe.csv"
        with summary_csv.open("w") as f:
            f.write("tag,err_max,err_l2\n")
            for tag, em, el2 in errs:
                f.write(f"{tag},{em:.6e},{el2:.6e}\n")
        print(f"[SUMMARY] {summary_csv}")
        print(f"==> Finished. Outputs in: {RUN_ROOT}")
=======
    for p in [2.0, 3.0, 4.0, 5.0]:
        Lx = Ly = 2*p*a
        tag = f"p{int(p)}a"
        run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix=f"results/phi_{tag}")
# === appended: write error fields to XDMF ==============================
try:
    import os, ufl
    from src.error_fields import write_error_fields_xdmf
    try:
        # Adjust if your function lives elsewhere
        from exact_rect_gates import rect_gates_phi_ufl
    except Exception:
        # Fallback: assume it's already imported in this script
        pass

    # Only run if main-level variables exist
    _have = all(name in globals() for name in ["domain", "V", "uh", "args"])
    if _have:
        x = ufl.SpatialCoordinate(domain)
        # Build analytic UFL expression — adjust args/params as needed to match your function signature
        if "rect_gates_phi_ufl" in globals():
            phi_exact_ufl = rect_gates_phi_ufl(
                x[0], x[1], x[2],
                a=getattr(args, "a", 70e-9),
                vp1=getattr(args, "vp1", 0.25),
                vx=getattr(args, "vx", 0.10),
                vp2=getattr(args, "vp2", 0.25),
            )
        else:
            raise NameError("rect_gates_phi_ufl not found. Import it or change call here.")

        out_xdmf = os.path.join(getattr(args, "outdir", "results"), "error_fields.xdmf")
        write_error_fields_xdmf(domain, V, uh, phi_exact_ufl, out_xdmf, tau=1e-9)
    else:
        print("[ERROR-FIELDS] Skipped writing errors (domain/V/uh/args not in scope).")
except Exception as _e:
    print("[ERROR-FIELDS] Failed to write error fields:", _e)
# ======================================================================
# === appended: write error fields to XDMF ==============================
try:
    import os, ufl
    from src.error_fields import write_error_fields_xdmf
    try:
        # Adjust if your function lives elsewhere
        from exact_rect_gates import rect_gates_phi_ufl
    except Exception:
        # Fallback: assume it's already imported in this script
        pass

    # Only run if main-level variables exist
    _have = all(name in globals() for name in ["domain", "V", "uh", "args"])
    if _have:
        x = ufl.SpatialCoordinate(domain)
        # Build analytic UFL expression — adjust args/params as needed to match your function signature
        if "rect_gates_phi_ufl" in globals():
            phi_exact_ufl = rect_gates_phi_ufl(
                x[0], x[1], x[2],
                a=getattr(args, "a", 70e-9),
                vp1=getattr(args, "vp1", 0.25),
                vx=getattr(args, "vx", 0.10),
                vp2=getattr(args, "vp2", 0.25),
            )
        else:
            raise NameError("rect_gates_phi_ufl not found. Import it or change call here.")

        out_xdmf = os.path.join(getattr(args, "outdir", "results"), "error_fields.xdmf")
        write_error_fields_xdmf(domain, V, uh, phi_exact_ufl, out_xdmf, tau=1e-9)
    else:
        print("[ERROR-FIELDS] Skipped writing errors (domain/V/uh/args not in scope).")
except Exception as _e:
    print("[ERROR-FIELDS] Failed to write error fields:", _e)
# ======================================================================
# ==== AUTO ERROR-MAP HOOK (appended) =======================================
# When phi (or uh) is written to XDMF, also write error fields to <outdir>/phi_error.xdmf
# Pulls a, vp1, vx, vp2, outdir from sys.argv (the same flags you already pass).
import sys, os, re
import ufl
from dolfinx import io as _io, fem as _fem

# optional dependency: exact_rect_gates.rect_gates_phi(x,y,z, a=..., vp1=..., vx=..., vp2=...)
try:
    from exact_rect_gates import rect_gates_phi as _rect_phi
except Exception as _e:
    _rect_phi = None

# helper provided earlier (make sure src/error_fields.py exists)
try:
    from src.error_fields import write_error_fields_xdmf as _write_err
except Exception as _e:
    _write_err = None

def _parse_flag(name, default=None, cast=float):
    """Parse --name <value> from sys.argv."""
    if isinstance(name, str) and not name.startswith("--"):
        name = f"--{name}"
    try:
        i = sys.argv.index(name)
        return cast(sys.argv[i+1])
    except Exception:
        return default

def _outdir_default():
    try:
        i = sys.argv.index("--outdir")
        return sys.argv[i+1]
    except Exception:
        return "results"

# Guard to avoid double-writing
_set_hook = getattr(sys.modules[__name__], "_err_hook_set", False)

if not _set_hook and _rect_phi is not None and _write_err is not None:
    _orig_write_fn = _io.XDMFFile.write_function

    def _write_function_with_error(self, u, *args, **kwargs):
        # call original writer first
        _orig_write_fn(self, u, *args, **kwargs)
        try:
            name = getattr(u, "name", "")
            if name not in ("phi", "uh"):
                return
            # gather params from argv
            a   = _parse_flag("a",   70e-9)
            vp1 = _parse_flag("vp1", 0.25)
            vx  = _parse_flag("vx",  0.10)
            vp2 = _parse_flag("vp2", 0.25)
            outdir = _outdir_default()
            # build analytic UFL on the same mesh/space
            V = u.function_space
            mesh = V.mesh
            x = ufl.SpatialCoordinate(mesh)
            phi_exact = _rect_phi(x[0], x[1], x[2], a=a, vp1=vp1, vx=vx, vp2=vp2)
            # choose outfile path next to your main phi output
            out_xdmf = os.path.join(outdir, "phi_error.xdmf")
            _write_err(mesh, V, u, phi_exact, out_xdmf, tau=1e-9)
        except Exception as _ee:
            # keep the main solve robust; only warn
            try:
                print("[ERROR-FIELDS] Skipped writing error fields:", _ee)
            except Exception:
                pass

    _io.XDMFFile.write_function = _write_function_with_error
    _err_hook_set = True
# ==========================================================================

>>>>>>> 7b176cafc436c4f4f7c51f1364ef42c02d769d99
