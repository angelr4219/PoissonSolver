#!/usr/bin/env bash
set -euo pipefail

# ---------- user knobs ----------
PAD=${PAD:-4}                  # fixed padding for pure p-refinement (e.g., 4a)
DEG_LIST=${DEG_LIST:-"1 2 3 4 5"}
A=${A:-70e-9}
ZFILL=${ZFILL:-2.0e-7}
XS=${XS:-'[-7.0e-8,0.0,7.0e-8]'}
VS=${VS:-'[0.25,0.10,0.25]'}
H_ELEM=${H_ELEM:-5e-9}
IMAGE=${IMAGE:-dolfinx/dolfinx:stable}
# --------------------------------

STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_ROOT="results/runs/${STAMP}"
CODE_SNAP="${RUN_ROOT}/code"
META_DIR="${RUN_ROOT}/meta"
OUT_DIR="${RUN_ROOT}/out_p${PAD}"
mkdir -p "${CODE_SNAP}" "${META_DIR}" "${OUT_DIR}"

echo "==> Run root: ${RUN_ROOT}"

# 1) snapshot code (no .git, no results)
# rsync keeps perms and is fast
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude=".git" \
    --exclude="results" \
    --exclude="__pycache__" \
    --exclude=".mypy_cache" \
    --exclude=".pytest_cache" \
    ./ "${CODE_SNAP}/"
else
  tar --exclude=.git --exclude=results --exclude=__pycache__ \
      --exclude=.mypy_cache --exclude=.pytest_cache \
      -cf - . | (cd "${CODE_SNAP}" && tar xf -)
fi

# 2) record git metadata (if in a repo)
{
  echo "timestamp: ${STAMP}"
  (git rev-parse --short HEAD 2>/dev/null && git status -sb && echo && git diff) || echo "not a git repo"
} > "${META_DIR}/git_snapshot.txt"

# 3) record environment from container
docker run --rm -v "$PWD":/app -w /app ${IMAGE} bash -lc '
python - <<PY
import sys, platform
import numpy as np
import ufl, petsc4py
try:
    import dolfinx
    from dolfinx import __version__ as dxxv
except Exception as e:
    dxxv=f"unavailable: {e}"
print("python:", platform.python_version())
print("platform:", platform.platform())
print("dolfinx:", dxxv)
print("ufl:", ufl.__version__)
print("petsc4py:", petsc4py.__version__)
print("numpy:", np.__version__)
PY
' | tee "${META_DIR}/env.txt" >/dev/null

# 4) run pure p-refinement (fixed geometry & mesh)
for k in ${DEG_LIST}; do
  echo "---- degree ${k} (pad=${PAD}) ----"
  docker run --rm -v "$PWD":/app -w /app ${IMAGE} bash -lc "\
/dolfinx-env/bin/python3 rect_gate_benchmark.py \
  --deg ${k} \
  --a ${A} \
  --zfill ${ZFILL} \
  --xs '${XS}' \
  --Vs '${VS}' \
  --h ${H_ELEM} \
  --pad ${PAD} \
  --run-root '${OUT_DIR}'"
done

# 5) aggregate per-run CSVs into one summary for this timestamp
python - <<'PY'
import csv, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
out  = root / "summary_prefine.csv"
rows = []
for f in sorted(root.glob("p*a_deg*_line.csv")):
    m = re.search(r"p(\d+)a_deg(\d+)_line\.csv$", f.name)
    if not m: 
        continue
    pad, deg = int(m.group(1)), int(m.group(2))
    # take the printed summary from the sibling summary file if present
    # otherwise compute L2 from the line file quickly
    with open(f, newline="") as fh:
        r = csv.DictReader(fh)
        xs, fe, an = [], [], []
        for row in r:
            xs.append(float(row["x_m"])); fe.append(float(row["phi_FE_V"])); an.append(float(row["phi0_V"]))
    import numpy as np
    diff = np.asarray(fe) - np.asarray(an)
    if diff.size:
        l2 = float(np.sqrt(np.mean(diff**2)))
        mmax = float(np.max(np.abs(diff)))
    else:
        l2 = float("nan"); mmax = float("nan")
    rows.append({"pad_a": pad, "degree": deg, "err_max": mmax, "err_l2": l2})
rows.sort(key=lambda r:(r["pad_a"], r["degree"]))
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["pad_a","degree","err_max","err_l2"])
    w.writeheader(); w.writerows(rows)
print("Wrote", out)
PY "${OUT_DIR}"

echo "==> Done. Snapshot: ${CODE_SNAP}"
echo "    Git/env meta: ${META_DIR}"
echo "    Results:      ${OUT_DIR}"
