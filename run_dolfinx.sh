#!/usr/bin/env bash
set -euo pipefail

# Only use -t if we're attached to a real TTY
TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

# Number of MPI ranks to run the script under. Defaults to 1 (no mpirun
# wrapper, same behavior as before). Set DOLFINX_NP=N to run with N ranks,
# e.g. `DOLFINX_NP=4 ./run_dolfinx.sh benchmarks/foo.py`. The image ships
# MPICH, whose mpirun runs as root inside the container without needing
# any extra flag (unlike OpenMPI's --allow-run-as-root).
DOLFINX_NP="${DOLFINX_NP:-1}"

docker run --rm ${TTY_FLAGS} \
  -v "$PWD":/app -w /app \
  -e DOLFINX_NP="${DOLFINX_NP}" \
  dolfinx/dolfinx:nightly \
  sh -lc '
    export PYTHONPATH="/app/src:${PYTHONPATH}"
    if [ "${DOLFINX_NP}" -gt 1 ]; then
      exec mpirun -n "${DOLFINX_NP}" /dolfinx-env/bin/python3 -u "$@"
    else
      exec /dolfinx-env/bin/python3 -u "$@"
    fi
  ' -- "$@"
