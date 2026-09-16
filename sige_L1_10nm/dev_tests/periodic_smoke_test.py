"""DEVELOPMENT SMOKE TEST ONLY - not a scientific result.

Minimal x/y-periodic Laplace problem on a box with CG1, top/bottom Dirichlet,
using exactly the same MPC construction strategy as the periodic AFM solver.
"""
import numpy as np, ufl
from mpi4py import MPI
from dolfinx import default_scalar_type, fem, mesh as dmesh
import dolfinx_mpc
from dolfinx_mpc import LinearProblem as MPCLinearProblem

comm = MPI.COMM_WORLD
LX = LY = 1.0
N = 12
msh = dmesh.create_box(
    comm, [np.array([-LX/2, -LY/2, 0.0]), np.array([LX/2, LY/2, 1.0])],
    [N, N, N], dmesh.CellType.tetrahedron,
    ghost_mode=dmesh.GhostMode.shared_facet,
)
V = fem.functionspace(msh, ("Lagrange", 1))
tol = 1e-8

def top(x):    return np.isclose(x[2], 1.0, atol=tol)
def bottom(x): return np.isclose(x[2], 0.0, atol=tol)

bcs = []
for locator, val in ((top, 1.0), (bottom, 0.0)):
    dofs = fem.locate_dofs_geometrical(V, locator)
    bcs.append(fem.dirichletbc(default_scalar_type(val), dofs, V))

at_xmax = lambda x: np.isclose(x[0], LX/2, atol=tol)
at_ymax = lambda x: np.isclose(x[1], LY/2, atol=tol)

mpc = dolfinx_mpc.MultiPointConstraint(V)
counts = {}
before = 0
def add(name, ind, rel):
    global before
    mpc.create_periodic_constraint_geometrical(V, ind, rel, bcs, default_scalar_type(1.0))
    now = len(mpc._slaves); counts[name] = now - before; before = now

# same 3-way split as the AFM solver: no chained or duplicated constraints
add("x", lambda x: at_xmax(x) & ~at_ymax(x), lambda x: np.vstack([x[0]-LX, x[1], x[2]]))
add("y", lambda x: at_ymax(x) & ~at_xmax(x), lambda x: np.vstack([x[0], x[1]-LY, x[2]]))
add("corner", lambda x: at_xmax(x) & at_ymax(x), lambda x: np.vstack([x[0]-LX, x[1]-LY, x[2]]))
mpc.finalize()
print(f"slave DOFs  x={counts['x']}  y={counts['y']}  corner={counts['corner']}  "
      f"total={mpc.num_local_slaves}")
assert counts["x"] > 0 and counts["y"] > 0 and counts["corner"] > 0, "MPC produced zero slaves"

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(msh, default_scalar_type(0.0)) * v * ufl.dx
prob = MPCLinearProblem(a, L, mpc, bcs=bcs, petsc_options_prefix="smoke_",
    petsc_options={"ksp_type": "gmres", "pc_type": "gamg", "ksp_rtol": 1e-12,
                   "ksp_error_if_not_converged": True})
uh = prob.solve(); uh.x.scatter_forward()
print(f"matrix assembled and solver converged: KSP its={prob.solver.getIterationNumber()} "
      f"reason={prob.solver.getConvergedReason()}")

# numerical periodicity check on the solution
X = V.tabulate_dof_coordinates(); vals = uh.x.array.real
def residual(axis, lo, hi):
    a_ = np.flatnonzero(np.isclose(X[:, axis], lo, atol=1e-7))
    b_ = np.flatnonzero(np.isclose(X[:, axis], hi, atol=1e-7))
    other = [k for k in range(3) if k != axis]
    key = lambda idx: [tuple(np.round(X[i, other], 7)) for i in idx]
    mb = dict(zip(key(b_), b_))
    d = [abs(vals[i] - vals[mb[k]]) for i, k in zip(a_, key(a_)) if k in mb]
    return (max(d) if d else float("nan")), len(d)

rx, nx = residual(0, -LX/2, LX/2)
ry, ny = residual(1, -LY/2, LY/2)
print(f"max |u(xmin,y,z)-u(xmax,y,z)| = {rx:.3e}   ({nx} matched pairs)")
print(f"max |u(x,ymin,z)-u(x,ymax,z)| = {ry:.3e}   ({ny} matched pairs)")
ok = (rx < 1e-10) and (ry < 1e-10)
print("SMOKE TEST:", "PASS" if ok else "FAIL")
assert ok, "periodicity not enforced"
