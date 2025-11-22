#!/usr/bin/env bash
set -euo pipefail

# ---------- EDIT THESE TO MATCH THE PAPER ----------
A=3.5e-8                            # gate half-width "a"
XS_JSON='[-7.0e-8, 0.0, 7.0e-8]'    # 3 gate centers
VS_JSON='[0.25, 0.10, 0.25]'        # gate voltages
ZFIL=3.5e-8                         # "zfill" thickness (your cushion above device)
PAD=3.0                              # lateral padding multiple
DEG=2                                # FE degree used in the paper figure (change if needed)
NX_LIST=("64" "96" "128" "192")      # refinement list for the paper plot
TAG="paper"                          # label for outputs (change if you want)

# Add PETSc verbosity if you like seeing solver progress
export PETSC_OPTIONS='-ksp_type cg -pc_type hypre -ksp_rtol 1e-10'

run_case () {
  local nx="$1"
  local outdir="results/cases/${TAG}_deg${DEG}_nx${nx}"
  local outprefix="${outdir}/phi_p${DEG}a"
  mkdir -p "${outdir}"
  echo "==> Running nx_hint=${nx}, deg=${DEG}  ->  ${outprefix}.*"

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 verify/gates.py \
      --nx_hint "${nx}" --deg "${DEG}" \
      --a "${A}" --zfill "${ZFIL}" --pad "${PAD}" \
      --xs "${XS_JSON}" --Vs "${VS_JSON}" \
      --outprefix "${outprefix}" \
  | tee "${outdir}/summary.json"

  # Basic check: mesh size & phi min/max
  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 - <<PY
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import json, os, numpy as np

xdmf = "${outprefix}.xdmf"
with XDMFFile(MPI.COMM_WORLD, xdmf, "r") as f:
    mesh = f.read_mesh()
    V = fem.functionspace(mesh, ("Lagrange", ${DEG}))
    u = fem.Function(V, name="phi")
    f.read_function(u)

mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
nc = mesh.topology.index_map(mesh.topology.dim).size_global
np_ = mesh.topology.index_map(0).size_global
arr = u.x.array
print(f"[check] {xdmf}: cells={nc:,} points={np_:,}  phi[min,max]=({float(arr.min()):.5g},{float(arr.max()):.5g})")
PY
}

# ---- RUN ALL MESHES ----
for nx in "${NX_LIST[@]}"; do
  run_case "${nx}"
done

# ---- POSTPROCESS: RMSE on midline vs finest mesh ----
echo "==> Postprocess: compute RMSE of midline vs finest mesh"
docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
  /dolfinx-env/bin/python3 - <<'PY'
import json, glob, os, csv
from pathlib import Path
import numpy as np

rows=[]
case_dirs = sorted(Path("results/cases").glob("paper_deg*_nx*"), key=lambda p:(int(str(p).split("_deg")[1].split("_")[0]), int(str(p).split("nx")[1])))
# Collect summaries + find finest trace per degree
by_deg = {}
for d in case_dirs:
    # try read summary
    summ = {}
    sfile = d/"summary.json"
    if sfile.exists():
        try:
            summ = json.loads(sfile.read_text())
        except Exception:
            pass
    # read line csv if exists
    line_glob = list(d.glob("phi_p*a_line.csv"))
    line = None
    if line_glob:
        arr = np.loadtxt(line_glob[0], delimiter=",", comments="#")
        if arr.ndim==1 and arr.size>=2:
            arr = arr.reshape(-1,2)
        if arr.shape[1]>=2:
            line = arr
    # parse nx, deg
    name = d.name  # e.g. paper_deg2_nx128
    try:
        deg = int(name.split("deg")[1].split("_")[0])
        nx  = int(name.split("nx")[1])
    except Exception:
        continue
    by_deg.setdefault(deg, []).append({"dir":d, "nx":nx, "summary":summ, "line":line})

# For each degree, compute RMSE to finest nx that has a line
out_tab=[]
for deg,items in by_deg.items():
    items = sorted(items, key=lambda x:x["nx"])
    finest = None
    for it in reversed(items):
        if it["line"] is not None:
            finest = it
            break
    if finest is None:
        # nothing to compare
        continue
    x_f, u_f = finest["line"][:,0], finest["line"][:,1]
    for it in items:
        rmse = np.nan
        if it["line"] is not None:
            x, u = it["line"][:,0], it["line"][:,1]
            # interpolate to finest grid for fair compare
            if len(x)>=2 and len(x_f)>=2:
                u_on_f = np.interp(x_f, x, u)
                diff = u_on_f - u_f
                rmse = float(np.sqrt(np.nanmean(diff**2)))
        # extract a few timing/size fields if present
        s = it["summary"]
        t_total = s.get("t_total", np.nan)
        dof     = s.get("dof", np.nan)
        h       = s.get("h", np.nan)
        out_tab.append([deg, it["nx"], dof, h, t_total, rmse, str(it["dir"])])

# write CSV
out_path = Path("results")/"paper_case_rmse.csv"
with out_path.open("w", newline="") as f:
    w=csv.writer(f)
    w.writerow(["deg","nx_hint","dof","h","t_total","RMSE_vs_finest","folder"])
    for r in sorted(out_tab, key=lambda r:(r[0], r[1])):
        w.writerow(r)
print("Wrote", out_path)
PY

echo "Done. Inspect results/paper_case_rmse.csv and the per-run folders under results/cases/."
