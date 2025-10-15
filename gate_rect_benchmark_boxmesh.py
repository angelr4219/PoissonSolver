#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl, os

# ---------- Analytic rectangular-gate potential (paper Eq. (10)-(11)) ----------
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
    return -s  # minus sign per Eq. (10)

# ---------- Build box with native FEniCSx (no Gmsh) ----------
def build_box_boxmesh(Lx, Ly, H, h):
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H ], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)

# ---------- Locate dofs on gate rectangles on z=0 plane ----------
def gate_dofs(V, a, xs, Vs, tol=1e-10):
    def on_top(x): return np.isclose(x[2], 0.0, atol=1e-12)
    dofs_top = fem.locate_dofs_geometrical(V, on_top)
    x_top = V.tabulate_dof_coordinates()[dofs_top].reshape((-1, 3))

    dofs_gate, vals_gate = [], []
    used = np.zeros(len(dofs_top), dtype=bool)
    for xi, Vi in zip(xs, Vs):
        in_rect = (
            (x_top[:, 0] >= (xi - a) - tol) & (x_top[:, 0] <= (xi + a) + tol) &
            (x_top[:, 1] >= -a - tol)      & (x_top[:, 1] <=  a + tol)
        )
        idx = np.where(in_rect)[0]
        if idx.size:
            dofs_gate.append(dofs_top[idx])
            vals_gate.append(np.full(idx.size, Vi, dtype=float))
            used[idx] = True

    idx0 = np.where(~used)[0]
    if idx0.size:
        dofs_gate.append(dofs_top[idx0])
        vals_gate.append(np.zeros(idx0.size, dtype=float))

    bcs = []
    for dof_array, values in zip(dofs_gate, vals_gate):
        bc_fun = fem.Function(V)
        bc_fun.x.array[:] = 0.0
        bc_fun.x.array[dof_array] = values
        bcs.append(fem.dirichletbc(bc_fun, dof_array))
    return bcs

# ---------- Solve Laplace with constant permittivity ----------
def solve_laplace(domain, bcs, eps_r=11.7, eps0=8.8541878128e-12):
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    lam = eps_r * eps0
    a = lam * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0)) * v * ufl.dx
    uh = LinearProblem(a, L, bcs=bcs).solve()
    uh.name = "phi"
    return V, uh

# ---------- Version-agnostic sampler: use nodal dofs on the line y=0, z≈zbar ----------
def sample_dof_line(uh: fem.Function, zbar: float, h: float, ytol: float = 1e-12):
    """
    Returns (x_sorted, u_sorted) by selecting nodal dofs with |y|<=ytol and |z-zbar|<=ztol,
    where ztol is tied to mesh size h. Works without geometry API differences.
    """
    V = uh.function_space
    X = V.tabulate_dof_coordinates().reshape((-1, 3))
    U = uh.x.array

    ztol = max(1e-12, 0.5 * h)  # half a cell in z is a good tolerance
    mask = (np.abs(X[:,1]) <= ytol) & (np.abs(X[:,2] - zbar) <= ztol)

    if not np.any(mask):
        # relax tolerances slightly if we missed due to grid parity
        ztol = max(ztol, h)
        mask = (np.abs(X[:,1]) <= 5e-12) & (np.abs(X[:,2] - zbar) <= ztol)

    xs = X[mask, 0]
    us = U[mask]
    if xs.size == 0:
        raise RuntimeError("No dofs found on the probe line; try a slightly different zbar or larger ztol.")

    order = np.argsort(xs)
    return xs[order], us[order]

# ---------- One run ----------
def run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix):
    domain = build_box_boxmesh(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", 1))
    bcs = gate_dofs(V, a, xs_gates, Vs_gates)

    V, uh = solve_laplace(domain, bcs)

    os.makedirs(os.path.dirname(outprefix), exist_ok=True)
    with io.XDMFFile(domain.comm, f"{outprefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)

    # --- Compare to analytic along the nodal line (y=0, z≈zbar)
    xnodes, uh_nodes = sample_dof_line(uh, zbar, h)  # robust to dolfinx version
    phi0_nodes = np.array([phi0_rect(x, 0.0, zbar, a, xs_gates, Vs_gates) for x in xnodes])

    # Error inside |x| ≤ 2a
    mask = np.abs(xnodes) <= 2*a
    err_max = float(np.max(np.abs(uh_nodes[mask] - phi0_nodes[mask])))
    err_l2  = float(np.sqrt(np.mean((uh_nodes[mask] - phi0_nodes[mask])**2)))

    if MPI.COMM_WORLD.rank == 0:
        import csv
        with open(f"{outprefix}_line.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_m", "phi_FE_V", "phi0_V"])
            for x, uval, aval in zip(xnodes, uh_nodes, phi0_nodes):
                w.writerow([x, uval, aval])
        print(f"[{outprefix}] max|Δφ| (|x|≤2a) = {err_max:.4e} V,  L2 = {err_l2:.4e} V")
    return err_max, err_l2

# ---------- Sweep box size ----------
if __name__ == "__main__":
    a_nm = 35.0
    a = a_nm * 1e-9
    zbar = a
    xs_gates = np.array([-2*a, 0.0, 2*a])
    Vs_gates = np.array([0.25, 0.10, 0.25])

    H = 200e-9
    h = 5e-9

    paddings = [2.0, 3.0, 4.0, 5.0]
    for p in paddings:
        Lx = Ly = 2*p*a
        tag = f"p{int(p)}a"
        run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar,
                 outprefix=f"results/phi_{tag}")
