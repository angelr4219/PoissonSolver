from __future__ import annotations
import numpy as np
from dolfinx import fem, io, geometry
from petsc4py import PETSc
import ufl
from dolfinx.fem.petsc import LinearProblem

# ---- analytic gates (free-space) ----
def _g_uvz(u, v, z):
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def phi0_rect_local(x, y, z, a, xs, Vs):
    z = z if z != 0.0 else np.finfo(float).eps
    s = 0.0
    for xi, Vi in zip(xs, Vs):
        s += Vi * (_g_uvz(a - xi + x, a + y, z)
                 + _g_uvz(a - xi + x, a - y, z)
                 + _g_uvz(a + xi - x, a + y, z)
                 + _g_uvz(a + xi - x, a - y, z))
    return -s

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

def sample_dof_line(uh: fem.Function, zbar: float, a: float, npts: int, h: float):
    V = uh.function_space; msh = V.mesh
    Xdof = V.tabulate_dof_coordinates().reshape((-1, 3))
    U = uh.x.array
    ztol = max(2*h, 1e-12)
    mask = (np.abs(Xdof[:, 1]) <= 1e-12) & (np.abs(Xdof[:, 2] - zbar) <= ztol)
    if np.any(mask):
        xs = Xdof[mask, 0]; us = U[mask]
        order = np.argsort(xs)
        return xs[order], us[order]

    xs = np.linspace(-2*a, 2*a, npts)
    P = np.stack([xs, np.zeros_like(xs), np.full_like(xs, zbar)], axis=1)
    tdim = msh.topology.dim
    tree = geometry.BoundingBoxTree(msh, tdim, msh.geometry.x)
    cells = np.empty(npts, dtype=np.int32)
    for i, p in enumerate(P):
        cand = geometry.compute_collisions(tree, p)
        coll = geometry.compute_colliding_cells(msh, cand, p)
        if len(coll) == 0:
            raise RuntimeError("Probe point not inside mesh.")
        cells[i] = coll[0]
    vals = uh.eval(P, cells)
    return xs, vals[:, 0]

# ---- projections & relerr writers (kept here to stay near verification) ----
def write_dg0_projection(domain, uh, outprefix):
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    uT, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    a_proj = ufl.inner(uT, v) * ufl.dx
    L_proj = ufl.inner(uh, v) * ufl.dx
    w = LinearProblem(a_proj, L_proj,
                      petsc_options={"ksp_type":"preonly","pc_type":"lu"},
                      petsc_options_prefix="proj_").solve()
    w.name = "phi_dg0"
    with io.XDMFFile(domain.comm, f"{outprefix}_deg0proj.xdmf", "w") as xo:
        xo.write_mesh(domain); xo.write_function(w)

def write_relerr_dg0(domain, uh, a, xs_gates, Vs_gates, outprefix, qdeg=4, tau=1e-9):
    V = uh.function_space
    uE = fem.Function(V, name="phi_exact")
    def _eval_exact(X):
        return np.array([phi0_rect_local(X[0,i], X[1,i], X[2,i], a, xs_gates, Vs_gates)
                         for i in range(X.shape[1])], dtype=float)
    uE.interpolate(_eval_exact)
    W = fem.functionspace(domain, ("Discontinuous Lagrange", 0))
    uT, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    dxm = ufl.dx(metadata={"quadrature_degree": qdeg})
    e = uh - uE
    num = ufl.sqrt(ufl.max_value(e*e, 0.0))
    den = ufl.sqrt(ufl.max_value(uE*uE, 0.0)) + tau
    f_rel = num/den
    w_rel = LinearProblem(ufl.inner(uT, v)*dxm, ufl.inner(f_rel, v)*dxm,
                          petsc_options={"ksp_type":"preonly","pc_type":"lu"},
                          petsc_options_prefix="proj_").solve()
    w_rel.name = "err_rel"
    with io.XDMFFile(domain.comm, f"{outprefix}_relerr.xdmf", "w") as xo:
        xo.write_mesh(domain); xo.write_function(w_rel)
