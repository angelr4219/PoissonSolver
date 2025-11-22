try:
    from dolfinx.io import gmshio as dgmsh
except ImportError:
    from dolfinx.io import gmsh as dgmsh
#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from dolfinx.fem import Function, FunctionSpace
import ufl
import gmsh
try:
    default_scalar_type
except NameError:
    import numpy as _np
    default_scalar_type = _np.float64


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
    """
    Build box [-Lx/2,Lx/2] x [-Ly/2,Ly/2] x [0,H] with uniform size h.
    Returns (domain, facet_tags). facet_tags: 1=top, 2=bottom, 3=x-sides, 4=y-sides.
    """
    import gmsh
    from mpi4py import MPI

    gmsh.initialize()
    gmsh.model.add("box")

    x0, y0, z0 = -Lx/2.0, -Ly/2.0, 0.0
    gmsh.model.occ.addBox(x0, y0, z0, Lx, Ly, H)
    gmsh.model.occ.synchronize()

    # Characteristic length
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    # Keep meshing fast
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

    # Physical groups
    surf = gmsh.model.getEntities(dim=2)
    vol  = gmsh.model.getEntities(dim=3)
    if vol:
        gmsh.model.addPhysicalGroup(3, [vol[0][1]], tag=10)

    # Classify surfaces by z to tag top/bottom; others by bbox extent
    top, bot, xsides, ysides = [], [], [], []
    for (dim, tag) in surf:
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(cz - H) < 1e-12*max(1.0, H):
            top.append(tag)
        elif abs(cz - 0.0) < 1e-12:
            bot.append(tag)
        else:
            bx0, by0, bz0, bx1, by1, bz1 = gmsh.model.getBoundingBox(dim, tag)
            dx, dy = bx1 - bx0, by1 - by0
            if dx > dy:
                ysides.append(tag)
            else:
                xsides.append(tag)

    if top:    gmsh.model.addPhysicalGroup(2, top, tag=1)
    if bot:    gmsh.model.addPhysicalGroup(2, bot, tag=2)
    if xsides: gmsh.model.addPhysicalGroup(2, xsides, tag=3)
    if ysides: gmsh.model.addPhysicalGroup(2, ysides, tag=4)

    gmsh.model.mesh.generate(3)

    # Convert to dolfinx mesh
    out = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0)
    if isinstance(out, tuple):
        if len(out) == 3:
            domain, _, facet_tags = out
        elif len(out) == 2:
            domain, _ = out
            facet_tags = None
        else:
            domain = out[0]
            facet_tags = None
    else:
        domain = out
        facet_tags = None

    gmsh.finalize()
    return domain, facet_tags


def gate_dofs(V, a, xs, Vs, facet_tags, top_tag=1, tol=1e-10):
    z_top = float(V.mesh.geometry.x[:,2].max())
    def on_top(x):
        return np.isclose(x[2], z_top, atol=1e-12)

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
def solve_laplace(domain, bcs, deg=1, eps_r=11.7, eps0=8.8541878128e-12):
    import ufl
    # Function space
    V = fem.functionspace(domain, ("Lagrange", deg))

    # Variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps = default_scalar_type(eps_r * eps0)
    a_ufl = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_ufl = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

    # Wrap as dolfinx forms for PETSc assembly
    a = fem.form(a_ufl)
    L = fem.form(L_ufl)

    # Assemble linear system A u = b with Dirichlet BCs
    A = assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = assemble_vector(L)

    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve with CG + Hypre (good default for SPD Laplace)
    uh = fem.Function(V, name="phi")
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    ksp.setTolerances(rtol=1e-10, max_it=5000)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    uh.name = "phi"
    uh.name = "phi"
    return V, uh

def sample_line(uh, xs, zbar):
    """
    Sample the FE function `uh` along the line y=0, z=zbar at x in `xs`.
    Returns a 1D NumPy array (np.nan where the point is outside the mesh).
    """
    import numpy as np
    from dolfinx import geometry

    V = uh.function_space
    mesh = V.mesh

    # Build query points
    xs_arr = np.asarray(xs, dtype=mesh.geometry.x.dtype)
    pts = np.zeros((xs_arr.size, 3), dtype=mesh.geometry.x.dtype)
    pts[:, 0] = xs_arr
    pts[:, 1] = 0.0
    pts[:, 2] = zbar

    # Spatial search
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    cells = geometry.compute_colliding_cells(mesh, candidates, pts)

    # In some dolfinx versions, 'cells' is an AdjacencyList; flatten to array
    if hasattr(cells, "links"):
        arr = np.full(xs_arr.size, -1, dtype=np.int32)
        for i in range(xs_arr.size):
            li = cells.links(i)
            if len(li) > 0:
                arr[i] = int(li[0])
        cells = arr

    vals = np.full(xs_arr.size, np.nan, dtype=float)
    mask = (cells >= 0)
    if mask.any():
        vals[mask] = uh.eval(pts[mask], cells[mask]).reshape(-1)
    return vals


def run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar, outprefix, deg=1):
    domain, facet_tags = build_box(Lx, Ly, H, h)
    V = fem.functionspace(domain, ("Lagrange", deg))
    bcs = gate_dofs(V, a, xs_gates, Vs_gates, facet_tags)
    V, uh = solve_laplace(domain, bcs, deg=deg)

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
