#!/usr/bin/env bash
set -euo pipefail

SCRIPT="disk_gate_poisson.py"
DOCKER_IMAGE="dolfinx/dolfinx:stable"
ROOT="disk_runs"

ts() { date +"%Y%m%d_%H%M%S"; }

run_case () {
  local label="$1"; shift
  local stamp
  stamp="$(ts)"
  local rundir="${ROOT}/${stamp}_${label}"
  mkdir -p "${rundir}/out"

  # record provenance
  cp -f "${SCRIPT}" "${rundir}/${SCRIPT}"
  {
    echo "timestamp: ${stamp}"
    echo "label: ${label}"
    echo "cwd: $(pwd)"
    echo "script: ${SCRIPT}"
    echo "docker_image: ${DOCKER_IMAGE}"
    echo -n "command: "
    echo docker run --rm -it -v "$(pwd)":/app -w /app "${DOCKER_IMAGE}" python3 -u "${SCRIPT}" "$@"
  } > "${rundir}/cmd.txt"

  # run and tee output (portable: redirect stderr to stdout)
  docker run --rm -it -v "$(pwd)":/app -w /app "${DOCKER_IMAGE}" \
    python3 -u "${SCRIPT}" \
      --outdir "${rundir}/out" \
      --basename "${label}" \
      "$@" 2>&1 | tee "${rundir}/run.log"
}

mkdir -p "${ROOT}"

# ----- CASES -----
run_case "A_eps1_laplace" \
  --eps_uniform 1.0 \
  --rho0 0.0

run_case "B_eps11p7_laplace" \
  --eps_uniform 11.7 \
  --rho0 0.0

run_case "C_eps12_laplace" \
  --eps_uniform 12.0 \
  --rho0 0.0

run_case "D_stack_nocharge" \
  --tox 10e-9 --eps_ox 3.9 --eps_si 11.7 \
  --rho0 0.0
# Stack + Gaussian volumetric charge
run_case "E_stack_withcharge" \
  --tox 10e-9 --eps_ox 3.9 --eps_si 11.7 \
  --rho0 1e5 --xq 0 --yq 0 --zq=-60e-9 --sigma 15e-9

