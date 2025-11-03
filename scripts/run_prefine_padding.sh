#!/usr/bin/env bash
set -euo pipefail

# --- user knobs ---
IMAGE="dolfinx/dolfinx:stable"      # or :nightly if you want
SOLVER="rect_gate_benchmark.py"     # your driver script (path inside repo)
PAD_LIST=(2 3 4 5)                  # p = {2a,3a,4a,5a}
DEG_LIST=(1 2 3 4 5)                # polynomial degrees for p-refinement
NX=${NX:-160}                       # base mesh density knobs (override via env)
NY=${NY:-160}
NZ=${NZ:-80}
A_NM=${A_NM:-35}                    # 'a' in nm (only for labeling if your script needs it)
EXTRA_ARGS=${EXTRA_ARGS:-""}        # any extra flags you want to pass through
# -------------------

STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="results/prefine_cube_${STAMP}"
mkdir -p "${OUTDIR}/logs" "${OUTDIR}/json" "${OUTDIR}/fields"

echo "Writing outputs to ${OUTDIR}"

# We’ll ask the solver to write a JSON metrics file; if it can't,
# we'll still log stdout and parse later.
for p in "${PAD_LIST[@]}"; do
  for k in "${DEG_LIST[@]}"; do
    tag="cube_p${p}a_deg${k}"
    log="${OUTDIR}/logs/${tag}.log"
    json="${OUTDIR}/json/${tag}.json"
    field="${OUTDIR}/fields/${tag}.xdmf"

    echo ">>> Running ${tag} ..."
    docker run --rm \
      -v "$PWD":/app -w /app ${IMAGE} \
      /dolfinx-env/bin/python3 "${SOLVER}" \
        --shape cube \
        --pad-a ${p} \
        --degree ${k} \
        --nx ${NX} --ny ${NY} --nz ${NZ} \
        --write-xdmf "${field}" \
        --write-json "${json}" \
        --a-nm ${A_NM} \
        ${EXTRA_ARGS} \
      2>&1 | tee "${log}"

    # If your solver doesn’t support --write-json, drop that flag above
    # and we’ll still have the logs to parse.
  done
done

echo "All runs finished. Aggregating..."
/usr/bin/env python3 scripts/summarize_prefine_padding.py "${OUTDIR}"
echo "Done. See:"
echo "  - ${OUTDIR}/summary.csv"
echo "  - ${OUTDIR}/summary.md"

