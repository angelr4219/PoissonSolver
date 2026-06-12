#!/usr/bin/env bash
# Run the triple-method sphere comparison inside the dolfinx Docker container.
#
# Usage:
#   ./benchmarks/run_sphere_benchmark.sh [extra args passed to sphere_triple_comparison.py]
#
# Examples:
#   ./benchmarks/run_sphere_benchmark.sh
#   ./benchmarks/run_sphere_benchmark.sh --skip-per          # skip FEM-Periodic (no dolfinx_mpc)
#   ./benchmarks/run_sphere_benchmark.sh --fft-max-n 200     # stricter memory cap
#   ./benchmarks/run_sphere_benchmark.sh --skip-dir --skip-per  # FFT only, fast smoke-test
#
# Prerequisites:
#   docker or podman available; run_dolfinx.sh present in repo root.
#
# Output lands in  results/sphere_triple/  (created automatically).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RUN="$REPO_ROOT/run_dolfinx.sh"
if [[ ! -x "$RUN" ]]; then
    echo "run_dolfinx.sh not found or not executable at $RUN" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/results/sphere_triple"

echo "========================================"
echo "  Sphere triple-comparison benchmark"
echo "  $(date)"
echo "========================================"

"$RUN" python3 benchmarks/sphere_triple_comparison.py "$@"
