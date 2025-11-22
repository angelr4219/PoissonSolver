#!/usr/bin/env bash
# Run commands inside a DOLFINx Docker container with your repo mounted at /app.
# Usage examples (from repo root):
#   ./scripts/run_dolfinx.sh -m poisson.cli --deg 1 ...
#   ./scripts/run_dolfinx.sh mpirun -n 4 python -m poisson.cli --deg 1 ...

set -euo pipefail
export DOCKER_PLATFORM="--platform=linux/amd64"

# Auto-force x86_64 on Apple Silicon unless already set
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -z "${DOCKER_PLATFORM:-}" ]]; then
  DOCKER_PLATFORM="--platform=linux/amd64"
fi
DOCKER_PLATFORM=${DOCKER_PLATFORM:-}

# Choose image (Docker Hub by default; override with: export DOLFINX_IMAGE=...)
IMAGE="${DOLFINX_IMAGE:-dolfinx/dolfinx:stable}"

# Resolve repo root even if called from a subdir
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
UIDGID="$(id -u):$(id -g)"

# Decide which Python inside the image has dolfinx
if docker run --rm ${DOCKER_PLATFORM} -v "${REPO_ROOT}":/app -w /app "$IMAGE" \
     /dolfinx-env/bin/python3 -c "import dolfinx" >/dev/null 2>&1; then
  PY="/dolfinx-env/bin/python3"
elif docker run --rm ${DOCKER_PLATFORM} -v "${REPO_ROOT}":/app -w /app "$IMAGE" \
     python3 -c "import dolfinx" >/dev/null 2>&1; then
  PY="python3"
else
  echo "ERROR: Could not import dolfinx in image: $IMAGE" >&2
  echo "Tip: try a different image, e.g. export DOLFINX_IMAGE=dolfinx/dolfinx:nightly" >&2
  exit 1
fi

# If first arg is 'mpirun', run via the image's mpirun (fallback to system if needed)
if [[ "${1:-}" == "mpirun" ]]; then
  shift
  MPIRUN="/dolfinx-env/bin/mpirun"
  if ! docker run --rm ${DOCKER_PLATFORM} -v "${REPO_ROOT}":/app -w /app "$IMAGE" \
       $MPIRUN -n 1 $PY -c "import dolfinx" >/dev/null 2>&1; then
    MPIRUN="mpirun"
  fi
  exec docker run --rm ${DOCKER_PLATFORM} \
    -u "${UIDGID}" -v "${REPO_ROOT}":/app -w /app \
    -e PYTHONPATH=/app/src \
    "$IMAGE" $MPIRUN "$@"
fi

# Otherwise, run Python directly (module or script)
exec docker run --rm ${DOCKER_PLATFORM} \
  -u "${UIDGID}" -v "${REPO_ROOT}":/app -w /app \
  -e PYTHONPATH=/app/src \
  "$IMAGE" $PY "$@"
