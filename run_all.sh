#!/usr/bin/env bash
set -euo pipefail

# --- Timestamped folders ---
STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="results/runs/${STAMP}"
RECT_DIR="${ROOT}/rect"
EXACT_DIR="${ROOT}/exact"
mkdir -p "${RECT_DIR}" "${EXACT_DIR}"

echo "==> Run stamp: ${STAMP}"
echo "==> Rect outputs -> ${RECT_DIR}"
echo "==> Exact sweep outputs -> ${EXACT_DIR}"

# --- 1) Rectangular-gate benchmark (same flags as before) ---
docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
 /dolfinx-env/bin/python3 rect_gate_benchmark.py \
  --deg 3 --a 3.5e-8 --zfill 3.5e-8 \
  --xs "[-7.0e-8,0,7.0e-8]" --Vs "[0.25,0.10,0.25]"

# Move all produced phi files into the rect folder
shopt -s nullglob
for f in results/phi_p*a*; do mv -f "$f" "${RECT_DIR}/"; done
shopt -u nullglob

# --- 2) exact_rect_sweep (your env-driven script) ---
mkdir -p scripts
cat > scripts/exact_rect_sweep.py <<'PY'
import os, time, csv, json, sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

a      = float(os.environ.get("A",      3.5e-8))
pad    = float(os.environ.get("PAD",    3.0))
H      = float(os.environ.get("H",      3.5e-8))
xs     = [float(x) for x in os.environ.get("XS", "-7.0e-8,0.0,7.0e-8").split(",")]
Vs     = [float(v) for v in os.environ.get("VS",  "0.25,0.10,0.25").split(",")]
deg    = int(os.environ.get("DEG", "2"))
NXS    = os.environ.get("NXS", "32,48,64,96,128")
nxs    = [int(s) for s in NXS.split(",") if s.strip()]

out_csv = Path(f"results/exact_rect_sweep_deg{deg}.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

sys.path.append("/app")
import exact_rect_gates as ERG
try:
    phi_factory = ERG.phi0_rect_three_gates_factory(a, xs, Vs, Vs)
except TypeError:
    phi_factory = ERG.phi0_rect_three_gates_factory(a, xs, Vs)

def phi_exact_eval(xyz: np.ndarray) -> np.ndarray:
    vals = np.empty(xyz.shape[0], dtype=float)
    for i,(x,y,z) in enumerate(xyz):
        vals[i] = float(phi_factory(x, y, z))
    return vals

def run_once(nx: int) -> dict:
    L = pad * 2.0 * a
    bounds = np.array([[-L, -L, -H], [L, L, 0.0]], dtype=float)
    nz = max(4, int(nx * (H / (2.0*L)) + 0.5))
    n = [nx, nx, nz]

    domain = mesh.create_box(MPI.COMM_WORLD, [bounds[0], bounds[1]],
                             n, cell_type=mesh.CellType.tetrahedron)

    V = fem.functionspace(domain, ("Lagrange", deg))
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = fem.Constant(domain, 0.0) * v * ufl.dx

    uD = fem.Function(V, name="phi_D")
    xall = V.tabulate_dof_coordinates().reshape((-1, 3))
    uD.x.array[:] = phi_exact_eval(xall)

    x0,y0,z0 = bounds[0]; x1,y1,z1 = bounds[1]
    def on_boundary(x):
        return np.isclose(x[0], x0) | np.isclose(x[0], x1) | \
               np.isclose(x[1], y0) | np.isclose(x[1], y1) | \
               np.isclose(x[2], z0) | np.isclose(x[2], z1)
    bdofs = fem.locate_dofs_geometrical(V, on_boundary)
    bcs = [fem.dirichletbc(uD, bdofs)]

    t0 = time.perf_counter()
    problem = LinearProblem(a_form, L_form, bcs=bcs)
    uh = problem.solve()
    t_solve = time.perf_counter() - t0

    uE = fem.Function(V, name="phi_exact"); uE.x.array[:] = uD.x.array
    e = uh - uE
    L2 = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))))
    H1 = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))))

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
    ncells = int(domain.topology.index_map(domain.topology.dim).size_global)
    npts   = int(domain.topology.index_map(0).size_global)
    dof    = int(V.dofmap.index_map.size_global)
    h_est  = (2.0*L)/nx

    return dict(nx=nx, nz=nz, dof=dof, cells=ncells, points=npts,
                h=h_est, L2=L2, H1=H1, t_total=t_solve)

def main():
    rows = []
    print(f"[exact_rect_sweep] deg={deg}  nxs={nxs}")
    for nx in nxs:
        r = run_once(nx)
        print(json.dumps(r))
        rows.append(r)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deg","nx","nz","dof","cells","points","h","L2","H1","t_total"])
        for r in rows:
            w.writerow([deg, r["nx"], r["nz"], r["dof"], r["cells"], r["points"],
                        f"{r['h']:.6e}", f"{r['L2']:.6e}", f"{r['H1']:.6e}",
                        f"{r['t_total']:.3f}"])
    print("Wrote", out_csv)

if __name__ == "__main__":
    main()
PY

docker run --rm -v "$PWD":/app -w /app \
  -e A=3.5e-8 -e PAD=3.0 -e H=3.5e-8 \
  -e XS="-7.0e-8,0.0,7.0e-8" -e VS="0.25,0.10,0.25" \
  -e DEG=3 -e NXS="32,48,64,96,128" \
  dolfinx/dolfinx:stable /dolfinx-env/bin/python3 scripts/exact_rect_sweep.py

# Move CSV to the stamped folder
if [[ -f results/exact_rect_sweep_deg3.csv ]]; then
  mv results/exact_rect_sweep_deg3.csv "${EXACT_DIR}/exact_rect_sweep_deg3_${STAMP}.csv"
fi

# Show results tree
if command -v tree >/dev/null 2>&1; then
  echo; tree -a "${ROOT}"
else
  echo; find "${ROOT}" -print
fi

echo
echo "Done. Outputs saved under ${ROOT}"
