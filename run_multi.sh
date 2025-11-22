#!/usr/bin/env bash
set -euo pipefail

# PETSc log/monitoring for all runs
export PETSC_OPTIONS='-ksp_monitor -ksp_type cg -pc_type hypre -ksp_rtol 1e-10'

run_case () {
  # usage: run_case <case_tag> <deg> <nx_hint> <label> <phi_base>
  local case_tag="$1"; local deg="$2"; local nx="$3"; local label="$4"; local phi="$5"
  local outdir="results/cases/${case_tag}_${label}"
  mkdir -p "$outdir"
  echo "==> Running ${case_tag}: deg=${deg}, nx_hint=${nx}  ->  ${outdir}/${phi}.*"
  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 verify/gates.py \
      --nx_hint "${nx}" --deg "${deg}" \
      --a 3.5e-8 --zfill 3.5e-8 --pad 3.0 \
      --xs "[-7.0e-8,0,7.0e-8]" --Vs "[0.25,0.10,0.25]" \
      --outprefix "${outdir}/${phi}" \
  | tee "${outdir}/summary.json"
}

check_mesh () {
  local xdmf="$1"
  [ -f "$xdmf" ] || { echo "skip (missing): $xdmf"; return 0; }
  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
    /dolfinx-env/bin/python3 - <<PY
from mpi4py import MPI
from dolfinx.io import XDMFFile
p="${xdmf}"
with XDMFFile(MPI.COMM_WORLD, p, "r") as f:
    mesh=f.read_mesh()
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
nc=mesh.topology.index_map(mesh.topology.dim).size_global
np=mesh.topology.index_map(0).size_global
print(f"{p}: cells={nc:,}  points={np:,}")
PY
}

# ------- Matrix of runs -------

# p2a (deg=2)
run_case p2a 2  64  h90x90x48     phi_p2a
run_case p2a 2 120  h120x120x64   phi_p2a
run_case p2a 2 180  h180x180x96   phi_p2a

# p3a (deg=3)
run_case p3a 3  64  h90x90x48     phi_p3a
run_case p3a 3 120  h120x120x64   phi_p3a
run_case p3a 3 180  h180x180x96   phi_p3a

# p4a (deg=4)
run_case p4a 4  64  h90x90x48     phi_p4a
run_case p4a 4 120  h120x120x64   phi_p4a
run_case p4a 4 180  h180x180x96   phi_p4a

# p5a (deg=5)
run_case p5a 5  64  h90x90x48     phi_p5a
run_case p5a 5 120  h120x120x64   phi_p5a
run_case p5a 5 180  h180x180x96   phi_p5a

# tiny sanity case
run_case p5a 5   4  2a_nx4        phi_p5a

# ------- Mesh size checks (best-effort) -------
check_mesh results/cases/p2a_h120x120x64/phi_p2a.xdmf || true
check_mesh results/cases/p3a_h120x120x64/phi_p3a.xdmf || true
check_mesh results/cases/p4a_h120x120x64/phi_p4a.xdmf || true
check_mesh results/cases/p5a_h120x120x64/phi_p5a.xdmf || true
check_mesh results/cases/p5a_2a_nx4/phi_p5a.xdmf      || true

echo "All requested runs attempted."
