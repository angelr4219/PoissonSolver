#!/usr/bin/env bash
set -euo pipefail
RUNROOT="results/runs/20251029-071922"   # change to your deg3 folder
A="3.5e-8"
XS_JSON='[-7.0e-8, 0.0, 7.0e-8]'
VS_JSON='[0.25, 0.10, 0.25]'
ZCONST="3.5e-8"
TAU="1e-9"

for TAG in p2a p3a p4a p5a; do
  CASE="${RUNROOT}/${TAG}_deg3.xdmf"
  [ -f "$CASE" ] || { echo "skip (missing): $CASE"; continue; }
  echo "---- verify: $CASE"
  docker run --rm \
    -e CASE="$CASE" -e A="$A" -e XS_JSON="$XS_JSON" -e VS_JSON="$VS_JSON" \
    -e ZCONST="$ZCONST" -e TAU="$TAU" \
    -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 - <<'PY'
# (same Python block as above)
import os, json, numpy as np
from mpi4py import MPI
from dolfinx import fem, io
import ufl
case   = os.environ["CASE"]
a      = float(os.environ["A"])
xs     = np.array(json.loads(os.environ["XS_JSON"]), dtype=float)
Vs     = np.array(json.loads(os.environ["VS_JSON"]), dtype=float)
zbar   = float(os.environ["ZCONST"])
tau    = float(os.environ["TAU"])
def g_uvz(u, v, z):
    import numpy as np
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))
def phi0_rect(x, y, z, a, xs, Vs):
    import numpy as np
    z = np.where(np.isclose(z, 0.0), np.finfo(float).eps, z)
    s = np.zeros_like(x, dtype=float)
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            g_uvz(a - xi + x, a + y, z) +
            g_uvz(a - xi + x, a - y, z) +
            g_uvz(a + xi - x, a + y, z) +
            g_uvz(a + xi - x, a - y, z)
        )
    return -s
comm = MPI.COMM_WORLD
rank = comm.rank
with io.XDMFFile(comm, case, "r") as xdmf:
    mesh = xdmf.read_mesh()
    V1 = fem.functionspace(mesh, ("Lagrange", 1))
    phi_h = fem.Function(V1, name="phi")
    try:
        xdmf.read_function(phi_h, name="phi")
    except Exception:
        phi_h.name = "phi_top_helper"
        xdmf.read_function(phi_h, name="phi_top_helper")
V = phi_h.function_space
coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
xx, yy = coords[:,0], coords[:,1]
zz     = np.full_like(xx, zbar)
phi_ref = fem.Function(V, name="phi_ref")
phi_ref.x.array[:] = phi0_rect(xx, yy, zz, a, xs, Vs)
arr_h   = phi_h.x.array
arr_ref = phi_ref.x.array
den     = np.maximum(np.abs(arr_ref), tau)
r_dof   = np.abs(arr_h - arr_ref) / den
rel_dof = fem.Function(V, name="rel_error_nodal")
rel_dof.x.array[:] = r_dof
Linf_dof = comm.allreduce(float(r_dof.max(initial=0.0) if r_dof.size>0 else 0.0), op=MPI.MAX)
L2_dof   = comm.allreduce(float(np.sum(r_dof*r_dof)), op=MPI.SUM)**0.5
V0     = fem.functionspace(mesh, ("DG", 0))
u0,v0  = ufl.TrialFunction(V0), ufl.TestFunction(V0)
r_expr = ufl.abs(phi_h - phi_ref) / ufl.max_value(ufl.abs(phi_ref), tau)
aP     = fem.form(u0*v0*ufl.dx)
LP     = fem.form(r_expr*v0*ufl.dx)
rel_cell = fem.Function(V0, name="rel_error_cellwise")
prob   = fem.petsc.LinearProblem(aP, LP, u=rel_cell,
         petsc_options={"ksp_type":"preonly","pc_type":"jacobi"})
prob.solve()
cell_arr = rel_cell.x.array
max_cell_local = float(cell_arr.max(initial=0.0) if cell_arr.size>0 else 0.0)
sum_cell_local = float(np.sum(cell_arr))
cnt_cell_local = float(cell_arr.size)
from mpi4py import MPI
max_cell = MPI.COMM_WORLD.allreduce(max_cell_local, op=MPI.MAX)
sum_cell = MPI.COMM_WORLD.allreduce(sum_cell_local, op=MPI.SUM)
cnt_cell = MPI.COMM_WORLD.allreduce(cnt_cell_local, op=MPI.SUM)
mean_cell = (sum_cell/cnt_cell) if cnt_cell>0 else 0.0
stem = os.path.splitext(case)[0]
out_nodal = f"{stem}_rel_error_nodal.xdmf"
out_cell  = f"{stem}_rel_error_cellwise.xdmf"
if rank==0:
    print(f"[WRITE] {out_nodal}")
with io.XDMFFile(comm, out_nodal, "w") as xf:
    xf.write_mesh(mesh); xf.write_function(rel_dof)
if rank==0:
    print(f"[WRITE] {out_cell}")
with io.XDMFFile(comm, out_cell, "w") as xf:
    xf.write_mesh(mesh); xf.write_function(rel_cell)
if rank==0:
    summ = f"{stem}_summary.csv"
    new  = not os.path.exists(summ)
    import csv
    with open(summ, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["case","a","xs","Vs","tau","REL_CELL_max","REL_CELL_mean","REL_DOF_Linf","REL_DOF_L2"])
        w.writerow([case, a,
                    ";".join(f"{v:.6g}" for v in xs),
                    ";".join(f"{v:.6g}" for v in Vs),
                    tau,
                    f"{max_cell:.6e}", f"{mean_cell:.6e}",
                    f"{Linf_dof:.6e}", f"{L2_dof:.6e}"])
    print(f"[SUMMARY] {summ}")
    print(f"[REL_DOF]  Linf={Linf_dof:.6e}  L2={L2_dof:.6e}")
    print(f"[REL_CELL] max={max_cell:.6e}  mean={mean_cell:.6e}")
PY
done
