#!/usr/bin/env python3
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# ---------- helper: robustly count Dirichlet dofs across dolfinx versions ----------
def _count_bc_dofs(bcs):
    n = 0
    for bc in bcs:
        idx = bc.dof_indices()  # may be numpy array or tuple/list
        try:
            n += idx.size
        except AttributeError:
            n += len(idx)
    return n

# ---------------- Analytic rectangular-gate potential (Eq. 10–11 style) ----------------
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

# ---------------- Build 3D box (no gmsh; avoids MPI partition quirks) ----------------
def build_box(Lx, Ly, H, h):
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H ], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

# ---------------- Dirichlet BCs on top z=0: Vi on gate squares, 0 elsewhere ----------------
def gate_dofs(V, a, xs, Vs, tol=1e-8):
    """
    Returns list of fem.DirichletBC objects that fix:
      - Gate squares centered at (xi, 0) of size 2a x 2a to Vi
      - All other dofs on the top plane to 0 V
    """
    def on_top(x): return np.abs(x[2]) <= tol  # robust top plane detection

    dofs_top = fem.locate_dofs_geometrical(V, on_top)
    x_all = V.tabulate_dof_coordinates().reshape((-1, 3))
    x_top = x_all[dofs_top]

    used = np.zeros(len(dofs_top), dtype=bool)
    bcs = []

    for xi, Vi in zip(xs, Vs):
        in_rect = (
            (x_top[:, 0] >= (xi - a) - tol) & (x_top[:, 0] <= (xi + a) + tol) &
            (x_top[:, 1] >= -a - tol)      & (x_top[:, 1] <=  a + tol)
        )
        ii = np.where(in_rect)[0]
        if ii.size:
            dofs_i = dofs_top[ii]
            bcs.append(fem.dirichletbc(PETSc.ScalarType(Vi), dofs_i, V))
            used[ii] = True

    # remainder of the top is ground
    ii0 = np.where(~used)[0]
    if ii0.size:
        dofs0 = dofs_top[ii0]
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs0, V))

    # sanity check
    ncon = _count_bc_dofs(bcs)
    if ncon == 0 and V.mesh.comm.rank == 0:
        raise RuntimeError("No top-surface DOFs were constrained; Dirichlet BCs did not apply.")
    return bcs

# ---------------- Solve −∇·(ε∇φ)=0 with constant ε ----------------
def solve_laplace(domain, bcs, eps_r=11.7, eps0=8.8541878128e-12):
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    lam = eps_r * eps0
    a = lam * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx
    uh = LinearProblem(a, L, bcs=bcs).solve()
    uh.name = "phi"
    return V, uh

# ---------------- Sample along nodal line: y=0, z≈zbar (tolerances tied to h) ----------------
def sample_dof_line(uh: fem.Function, zbar: float, h: float, ytol: float = 1e-12):
    """
    Get (x_sorted, u_sorted) by selecting nodal dofs with |y|<=ytol and |z-zbar|<=ztol.
    If nothing is found at first, relax ztol up to one cell.
    """
    V = uh.function_space
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    U = uh.x.array

    ztol = max(1e-12, 0.5 * h)
    mask = (np.abs(X[:,1]) <= ytol) & (np.abs(X[:,2] - zbar) <= ztol)
    if not np.any(mask):
        ztol = max(ztol, h)
        mask = (np.abs(X[:,1]) <= 5e-12) & (np.abs(X[:,2] - zbar) <= ztol)

    xs = X[mask, 0]
    us = U[mask]
    if xs.size == 0:
        raise RuntimeError("No dofs found on probe line; try slightly adjusting zbar or mesh size.")
    order = np.argsort(xs)
    return xs[order], us[order]

# ---------------- One run (build, solve, sample, compare) ----------------
def run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix):
    domain = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", 1))

    bcs = gate_dofs(V, a, xs_gates, Vs_gates)
    if MPI.COMM_WORLD.rank == 0:
        print("Dirichlet DOFs on top:", _count_bc_dofs(bcs))

    V, uh = solve_laplace(domain, bcs)

    os.makedirs(os.path.dirname(outprefix), exist_ok=True)
    with io.XDMFFile(domain.comm, f"{outprefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)

    # Compare to analytic along nodal line
    xnodes, uh_nodes = sample_dof_line(uh, zbar, h)
    phi0_nodes = np.array([phi0_rect(x, 0.0, zbar, a, xs_gates, Vs_gates) for x in xnodes])

    # Error inside |x| ≤ 2a
    mask = np.abs(xnodes) <= 2*a
    err_max = float(np.max(np.abs(uh_nodes[mask] - phi0_nodes[mask])))
    err_l2  = float(np.sqrt(np.mean((uh_nodes[mask] - phi0_nodes[mask])**2)))

    if MPI.COMM_WORLD.rank == 0:
        import csv
        with open(f"{outprefix}_line.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["x_m", "phi_FE_V", "phi0_V"])
            for x, uval, aval in zip(xnodes, uh_nodes, phi0_nodes):
                w.writerow([x, uval, aval])
        print(f"[{outprefix}] max|Δφ| (|x|≤2a) = {err_max:.4e} V,  L2 = {err_l2:.4e} V")
    return err_max, err_l2

# ---------------- Main sweep: domain/edge effects vs lateral padding ----------------
if __name__ == "__main__":
    a_nm = 35.0
    a = a_nm * 1e-9
    zbar = a
    xs_gates = np.array([-2*a, 0.0, 2*a])   # centers
    Vs_gates = np.array([0.25, 0.10, 0.25]) # volts

    H = 200e-9   # vertical extent
    h = 5e-9     # target cell size

    paddings = [2.0, 3.0, 4.0, 5.0]
    for p in paddings:
        Lx = Ly = 2*p*a
        tag = f"p{int(p)}a"
        run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix=f"results/phi_{tag}")
