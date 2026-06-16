#!/usr/bin/env bash
# Run the 4-quadrant mesh-density demo inside the dolfinx Docker container.
#
# Usage:
#   ./benchmarks/run_quadrant_demo.sh [extra args passed to quadrant_mesh_density_demo.py]
#
# Examples:
#   ./benchmarks/run_quadrant_demo.sh
#   ./benchmarks/run_quadrant_demo.sh --L-nm 400
#
# Prerequisites:
#   docker or podman available; run_dolfinx.sh present in repo root.
#
# Output lands in  results/quadrant_density/  (created automatically).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RUN="$REPO_ROOT/run_dolfinx.sh"
if [[ ! -x "$RUN" ]]; then
    echo "run_dolfinx.sh not found or not executable at $RUN" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/results/quadrant_density"

echo "========================================"
echo "  4-quadrant mesh-density demo"
echo "  $(date)"
echo "========================================"

"$RUN" python3 benchmarks/quadrant_mesh_density_demo.py "$@"
