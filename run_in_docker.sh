#!/usr/bin/env bash
set -euo pipefail

IMAGE="${DOLFINX_IMAGE:-ghcr.io/fenics/dolfinx/dolfinx:stable}"
WORKDIR_IN_CONTAINER="/work"

if [ "$#" -eq 0 ]; then
  echo "Usage:"
  echo "  $0 python3 main.py <case> [args...]"
  echo
  echo "Example:"
  echo "  $0 python3 main.py square_gate_stack --outdir results/test --basename test"
  exit 1
fi

docker run --rm -it \
  --shm-size=1g \
  -v "$PWD":"$WORKDIR_IN_CONTAINER" \
  -w "$WORKDIR_IN_CONTAINER" \
  "$IMAGE" \
  "$@"
