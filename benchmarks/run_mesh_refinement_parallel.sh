#!/usr/bin/env bash
# Run the 3 mesh_refinement_tradeoff.py configs (uniform_coarse, uniform_fine,
# quadrant_mixed) CONCURRENTLY as 3 separate ./run_dolfinx.sh (docker run)
# processes, each writing one config's result, then merge them into the same
# tradeoff_results.csv + report.txt format the sequential run produces.
#
# This is the "concurrent independent configs" parallelism path: each config
# is an independent gmsh + PETSc solve with non-thread-safe global C state,
# so the safe way to run them side by side is as separate OS processes/
# containers, not threads within one process. Each config can additionally
# use multiple MPI ranks internally via DOLFINX_NP (see run_dolfinx.sh);
# combine both with e.g. DOLFINX_NP=2 ./run_mesh_refinement_parallel.sh.
#
# Usage:
#   ./benchmarks/run_mesh_refinement_parallel.sh [extra args passed through
#                                                  to every config's solve]
#
# Output lands in results/mesh_refinement_parallel/ (created automatically),
# with one subdir per config plus the merged tradeoff_results.csv/report.txt
# at the top level.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RUN="$REPO_ROOT/run_dolfinx.sh"
if [[ ! -x "$RUN" ]]; then
    echo "run_dolfinx.sh not found or not executable at $RUN" >&2
    exit 1
fi

# run_dolfinx.sh mounts "$PWD" (not REPO_ROOT) as /app, so cd there first --
# otherwise relative --outdir paths below would land outside the mount.
cd "$REPO_ROOT"

# run_dolfinx.sh mounts $PWD (REPO_ROOT) as /app inside the container and
# runs with cwd=/app, so --outdir must be a path RELATIVE to REPO_ROOT --
# an absolute host path like "$REPO_ROOT/results/..." would resolve inside
# the container's own (ephemeral, discarded-on-exit) filesystem instead of
# the mounted volume.
REL_OUTDIR="results/mesh_refinement_parallel"
OUTDIR="$REPO_ROOT/$REL_OUTDIR"
mkdir -p "$OUTDIR"

echo "========================================"
echo "  Mesh-refinement tradeoff -- 3 configs in parallel"
echo "  $(date)"
echo "========================================"

T0=$(date +%s)

pids=()
names=(uniform_coarse uniform_fine quadrant_mixed)
for name in "${names[@]}"; do
    "$RUN" benchmarks/mesh_refinement_tradeoff.py --only "$name" --outdir "$REL_OUTDIR" "$@" \
        > "$OUTDIR/${name}.log" 2>&1 &
    pids+=($!)
    echo "launched $name (pid $!), log -> $OUTDIR/${name}.log"
done

status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "config '${names[$i]}' FAILED -- see $OUTDIR/${names[$i]}.log" >&2
        status=1
    fi
done

T1=$(date +%s)
echo "all configs finished in $((T1 - T0))s wall time (concurrent)"

if [[ $status -ne 0 ]]; then
    echo "one or more configs failed; not merging" >&2
    exit $status
fi

# merge_tradeoff_results.py only needs stdlib (json/csv) -- no dolfinx/docker
# needed for this step.
python3 "$SCRIPT_DIR/merge_tradeoff_results.py" "$OUTDIR"
