#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem import Function, FunctionSpace
import ufl
import gmsh

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

# ---------- Build mesh with top surface z=0 and box extending to z=H ----------
def build_box(Lx, Ly, H, h):
    gmsh.initialize()
    gmsh.model.add("rect_gate_box")
    vol = gmsh.model.occ.addBox(-Lx/2, -Ly/2, 0.0, Lx, Ly, H)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(3)

    # Tag top surface (z=0) as physical group 1
    top_faces = []
    for dim, tag in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[2] - 0.0) < 1e-12:
            top_faces.append(tag)
    if top_faces:
        gmsh.model.addPhysicalGroup(2, top_faces, 1)

    from dolfinx.io import gmshio
    domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return domain, facet_tags

# ---------- Locate dofs on gate rectangles on z=0 plane ----------
def gate_dofs(V, a, xs, Vs, facet_tags, top_tag=1, tol=1e-10):
    def on_top(x):
        return np.isclose(x[2], 0.0, atol=1e-12)

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
        bc_fun = Function(V)
        bc_fun.x.array[:] = 0.0
        bc_fun.x.array[dof_array] = values
        bcs.append(fem.dirichletbc(bc_fun, dof_array))
    return bcs

# ---------- Solve Laplace with constant permittivity ----------
def solve_laplace(domain, bcs, eps_r=11.7, eps0=8.8541878128e-12):
    V = FunctionSpace(domain, ("Lagrange", 1))
    u = Function(V, name="phi")
    v = ufl.TestFunction(V)
    lam = eps_r * eps0
    a = lam * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.Constant(domain, 0.0) * v * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
    uh = problem.solve()
    return V, uh

# ---------- Sample along y=0 at depth z̄ ----------
def sample_line(uh, xs, zbar):
    pts = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, zbar)])
    vals = uh.eval(pts)
    return vals

# ---------- One run ----------
def run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix):
    domain, facet_tags = build_box(Lx, Ly, H, h)
    V = FunctionSpace(domain, ("Lagrange", 1))
    bcs = gate_dofs(V, a, xs_gates, Vs_gates, facet_tags)
    V, uh = solve_laplace(domain, bcs)

    # Save field
    with io.XDMFFile(MPI.COMM_WORLD, f"{outprefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)

    # Compare to analytic along a line
    xline = np.linspace(-3*a, 3*a, 601)
    uh_line = sample_line(uh, xline, zbar).ravel()
    phi0_line = np.array([phi0_rect(x, 0.0, zbar, a, xs_gates, Vs_gates) for x in xline])

    # Error inside |x| ≤ 2a
    mask = np.abs(xline) <= 2*a
    err_max = float(np.max(np.abs(uh_line[mask] - phi0_line[mask])))
    err_l2 = float(np.sqrt(np.mean((uh_line[mask] - phi0_line[mask])**2)))

    if MPI.COMM_WORLD.rank == 0:
        import csv, os
        os.makedirs(os.path.dirname(outprefix), exist_ok=True)
        with open(f"{outprefix}_line.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_m", "phi_FE_V", "phi0_V"])
            for x, uval, aval in zip(xline, uh_line, phi0_line):
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
        run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix=f"results/phi_{tag}")
