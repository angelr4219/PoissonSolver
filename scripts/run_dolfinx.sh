#!/usr/bin/env bash
# Run commands inside the dolfinx Docker container with your repo mounted at /app

set -euo pipefail

# Auto-force x86_64 on Apple Silicon unless already set
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -z "${DOCKER_PLATFORM:-}" ]]; then
  DOCKER_PLATFORM="--platform=linux/amd64"
fi

# If you want to force manually, you can: export DOCKER_PLATFORM="--platform=linux/amd64"
DOCKER_PLATFORM=${DOCKER_PLATFORM:-}

# Resolve repo root even if called from a subdir
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Map user so outputs aren't root-owned
UIDGID="$(id -u):$(id -g)"

# Default: use the conda Python inside the image
# If first arg is "mpirun", switch to the image's mpirun
if [[ "${1:-}" == "mpirun" ]]; then
  shift
  CMD=(/dolfinx-env/bin/mpirun "$@")
else
  CMD=(/dolfinx-env/bin/python3 "$@")
fi

exec docker run --rm \
  ${DOCKER_PLATFORM} \
  -u "${UIDGID}" \
  -v "${REPO_ROOT}":/app \
  -w /app \
  -e PYTHONPATH=/app/src \
  dolfinx/dolfinx:stable \
  "${CMD[@]}"
