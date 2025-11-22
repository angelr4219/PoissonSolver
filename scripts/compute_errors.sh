#!/usr/bin/env bash
# Compute analytic vs FEM error for each (deg, nx) subfolder in a given run folder.
# Writes per-subfolder summary.csv and aggregates summary_all.csv at run root.

set -euo pipefail
shopt -s nullglob dotglob

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 results/runs/<STAMP>"
  exit 2
fi

RUN_ROOT="$1"
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Not a directory: ${RUN_ROOT}"
  exit 2
fi

STAMP="$(basename "${RUN_ROOT}")"
TOP_SUMMARY="${RUN_ROOT}/summary_all.csv"
: > "${TOP_SUMMARY}"
echo "run_stamp,deg,nx_hint,pad,label,L2,Linfty,max_abs_delta,num_points" >> "${TOP_SUMMARY}"

# Process each setting folder (degX or degX_nxY)
for setting_dir in "${RUN_ROOT}"/deg*; do
  [[ -d "${setting_dir}" ]] || continue

  # Parse deg and nx from folder name
  base="$(basename "${setting_dir}")"
  if [[ "${base}" =~ ^deg([0-9]+)_nx([0-9]+)$ ]]; then
    DEG="${BASH_REMATCH[1]}"
    NX="${BASH_REMATCH[2]}"
  elif [[ "${base}" =~ ^deg([0-9]+)$ ]]; then
    DEG="${BASH_REMATCH[1]}"
    NX=""
  else
    DEG=""
    NX=""
  fi

  SUB_SUMMARY="${setting_dir}/summary.csv"
  : > "${SUB_SUMMARY}"
  echo "run_stamp,deg,nx_hint,pad,label,L2,Linfty,max_abs_delta,num_points" >> "${SUB_SUMMARY}"

  # For each pad file present
  for xdmf in "${setting_dir}"/phi_p[2345]a*.xdmf; do
    [[ -f "${xdmf}" ]] || continue

    PAD=""; LABEL="$(basename "${xdmf}")"
    if [[ "${LABEL}" =~ phi_(p[2345]a).*\.xdmf$ ]]; then
      PAD="${BASH_REMATCH[1]}"
    fi

    echo "Computing error for: ${xdmf}"

    docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable \
      /dolfinx-env/bin/python3 - <<PY
import os, re, math, csv, sys
import numpy as np
from mpi4py import MPI
from dolfinx import io, fem

# Constants (your physical setup)
a    = 3.5e-8
xs   = np.array([-7.0e-8, 0.0, 7.0e-8], dtype=float)
Vs   = np.array([0.25, 0.10, 0.25], dtype=float)

xdmf_path = r"${xdmf}"
run_stamp = r"${STAMP}"
deg       = r"${DEG}"
nx_hint   = r"${NX}"
pad       = r"${PAD}"
label     = os.path.basename(xdmf_path)

# Try to import your analytic; fallback to a local copy
def _g_uvz(u, v, z):
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def _phi0_rect(x, y, z, a, xs, Vs):
    # tiny z lift to avoid z=0 singular eval
    z = z if z != 0.0 else np.finfo(float).eps
    s = 0.0
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            _g_uvz(a - xi + x, a + y, z) +
            _g_uvz(a - xi + x, a - y, z) +
            _g_uvz(a + xi - x, a + y, z) +
            _g_uvz(a + xi - x, a - y, z)
        )
    return s

try:
    import exact_rect_gates as ERG
    # Prefer a 3-arg callable if present; else use fallback
    try:
        _ = ERG.phi0_rect(0.0, 0.0, 1e-9, a, xs, Vs)
        phi_exact = lambda X: ERG.phi0_rect(X[0], X[1], X[2], a, xs, Vs)
    except TypeError:
        phi_exact = lambda X: ERG.phi0_rect(X, a, xs, Vs)
except Exception:
    phi_exact = lambda X: _phi0_rect(X[0], X[1], X[2], a, xs, Vs)

# Load solution
comm = MPI.COMM_WORLD
with io.XDMFFile(comm, xdmf_path, "r") as f:
    mesh = f.read_mesh(name=None)
    f.read_information(mesh)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    uh = fem.Function(V, name="phi")
    try:
        f.read_function(uh)
    except Exception:
        # If file stores higher degree: build CG1 and read into that
        V1 = fem.functionspace(mesh, ("Lagrange", 1))
        uh = fem.Function(V1, name="phi")
        f.read_function(uh)

# Compare on nodal coordinates (quick, consistent across runs)
X = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
U = uh.x.array
n = U.size

phi0 = np.empty(n, dtype=float)
for i in range(n):
    phi0[i] = float(phi_exact(X[i]))

diff = U - phi0
L2 = float(np.sqrt(np.mean(diff**2))) if n > 0 else float("nan")
Linfty = float(np.max(np.abs(diff))) if n > 0 else float("nan")
max_abs_delta = Linfty
row = [run_stamp, deg, nx_hint, pad, label, L2, Linfty, max_abs_delta, n]

# Write per-subfolder CSV
sub_csv = r"${SUB_SUMMARY}"
top_csv = r"${TOP_SUMMARY}"
for target in (sub_csv, top_csv):
    header_needed = not os.path.exists(target) or os.path.getsize(target) == 0
    with open(target, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["run_stamp","deg","nx_hint","pad","label","L2","Linfty","max_abs_delta","num_points"])
        w.writerow(row)
print(f"Wrote: {sub_csv} and appended {top_csv}")
PY

  done
done

echo
echo "==> Aggregated summary written to: ${TOP_SUMMARY}"
echo
command -v column >/dev/null 2>&1 && column -t -s, "${TOP_SUMMARY}" | sed 's/^/  /'
