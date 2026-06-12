#!/usr/bin/env bash
# Run the sphere triple comparison benchmark inside the Docker container.
# Usage: ./benchmarks/run_sphere_benchmark.sh [--skip-dir] [--skip-per] [--skip-fft]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="dolfinx/dolfinx:v0.7.2"

echo "=== Sphere Triple Comparison Benchmark ==="
echo "Project root : $PROJECT_ROOT"
echo "Docker image : $IMAGE"
echo "Args         : $*"
echo

docker run --rm \
  -v "$PROJECT_ROOT":/workspace \
  -w /workspace \
  "$IMAGE" \
  bash -c "
    pip install --quiet dolfinx-mpc 2>/dev/null || true
    python benchmarks/sphere_triple_comparison.py $*
  "
