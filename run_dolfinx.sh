#!/usr/bin/env bash
set -euo pipefail

# Only use -t if we're attached to a real TTY
TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app -w /app \
  dolfinx/dolfinx:nightly \
  sh -lc 'export PYTHONPATH="/app/src:${PYTHONPATH}"; /dolfinx-env/bin/python3 -u "$@"' -- "$@"
