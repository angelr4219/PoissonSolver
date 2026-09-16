#!/usr/bin/env bash

set -u
set -o pipefail

###############################################################################
# AFM + Si/SiGe floating-gate verification campaign
#
# Run from:
#   /Users/angelramirez/Desktop/poisson_solver
#
# Uses:
#   sige_L1_10nm/afm_sige_floating_gates.py
###############################################################################

IMAGE="dolfinx/dolfinx:stable"
MPI_N=4

SOLVER="sige_L1_10nm/afm_sige_floating_gates.py"

OUTROOT="sige_L1_10nm/results/verification_campaign"

MANIFEST="${OUTROOT}/campaign_manifest.csv"

mkdir -p "${OUTROOT}"

if [[ ! -f "${SOLVER}" ]]; then
    echo "ERROR: solver not found:"
    echo "  ${SOLVER}"
    exit 1
fi


###############################################################################
# Manifest
###############################################################################

cat > "${MANIFEST}" <<'EOF'
case_name,gap_nm,bc,lx_nm,ly_nm,h_fine_nm,h_coarse_nm,return_code
EOF


###############################################################################
# Run one physical case
###############################################################################

run_case()
{
    local GAP="$1"
    local BC="$2"
    local H_FINE="$3"
    local LX="$4"
    local LY="$5"
    local CASE_NAME="$6"

    local HOST_OUT="${OUTROOT}/${CASE_NAME}"
    local CONTAINER_OUT="/work/${HOST_OUT}"

    mkdir -p "${HOST_OUT}"

    echo
    echo "======================================================================"
    echo "RUNNING"
    echo "case      : ${CASE_NAME}"
    echo "gap       : ${GAP} nm"
    echo "BC        : ${BC}"
    echo "box       : ${LX} x ${LY} nm"
    echo "h_fine    : ${H_FINE} nm"
    echo "======================================================================"
    echo

    BC_ARGS=()

    if [[ "${BC}" == "dirichlet" ]]; then
        BC_ARGS+=(
            --outer-voltage 0.0
        )

    elif [[ "${BC}" == "neumann" ]]; then
        BC_ARGS+=(
            --outer-neumann
        )

    else
        echo "ERROR: unknown BC '${BC}'"
        return 99
    fi

    docker run --rm -i \
      -v "$PWD":/work \
      -w /work \
      "${IMAGE}" \
      mpiexec -n "${MPI_N}" \
      python3 "/work/${SOLVER}" \
        --lx "${LX}" \
        --ly "${LY}" \
        --air-height 320 \
        --buffer-thickness 2000 \
        --gap "${GAP}" \
        --tip-radius 10 \
        --shank-height 160 \
        --shank-top-radius 120 \
        --shaft-radius 120 \
        --shaft-height 100 \
        --gate-radius 10 \
        --gate-thickness 2 \
        --gate1-z 35 \
        --gate2-z 48 \
        --eps-air 1.0 \
        --eps-si 11.7 \
        --eps-sige 12.0 \
        --back-voltage -4.4 \
        "${BC_ARGS[@]}" \
        --tip-voltages 0 1 -1 \
        --h-fine "${H_FINE}" \
        --h-coarse 20 \
        --refine-dist-min 10 \
        --refine-dist-max 80 \
        --probe-z 35 \
        --probe-half-width 100 \
        --probe-points 201 \
        --degree 1 \
        --output "${CONTAINER_OUT}" \
      2>&1 | tee "${HOST_OUT}/run.log"

    local RC=${PIPESTATUS[0]}

    echo \
      "${CASE_NAME},${GAP},${BC},${LX},${LY},${H_FINE},20,${RC}" \
      >> "${MANIFEST}"

    if [[ ${RC} -eq 0 ]]; then
        echo
        echo "SUCCESS: ${CASE_NAME}"
    else
        echo
        echo "FAILED: ${CASE_NAME}"
        echo "return code = ${RC}"
        echo "Continuing with remaining cases."
    fi

    echo
}


###############################################################################
# PART A
#
# Gap sweep
#
# SAME nominal 5 nm FEM resolution:
#
#   1 nm
#   10 nm
#   20 nm
#   30 nm
#   50 nm
#
# For both:
#
#   grounded Dirichlet outer boundaries
#   natural Neumann outer boundaries
#
###############################################################################

echo
echo "######################################################################"
echo "PART A: GAP SWEEP"
echo "######################################################################"

for GAP in 1 10 20 30 50
do
    GAP_TAG=$(printf "%03d" "${GAP}")

    run_case \
        "${GAP}" \
        "dirichlet" \
        "5" \
        "300" \
        "300" \
        "gap_${GAP_TAG}nm/bc_dirichlet/h5"

    run_case \
        "${GAP}" \
        "neumann" \
        "5" \
        "300" \
        "300" \
        "gap_${GAP_TAG}nm/bc_neumann/h5"
done


###############################################################################
# PART B
#
# Mesh convergence
#
# Focus on current 10 nm baseline:
#
#   h = 5 nm
#   h = 3 nm
#   h = 2 nm
#
# h=5 already exists from Part A.
###############################################################################

echo
echo "######################################################################"
echo "PART B: MESH CONVERGENCE AT 10 nm GAP"
echo "######################################################################"

for H in 3 2
do

    run_case \
        "10" \
        "dirichlet" \
        "${H}" \
        "300" \
        "300" \
        "gap_010nm/bc_dirichlet/h${H}"

    run_case \
        "10" \
        "neumann" \
        "${H}" \
        "300" \
        "300" \
        "gap_010nm/bc_neumann/h${H}"

done


###############################################################################
# PART C
#
# 1 nm gap refinement
#
# h=5 is NOT enough to consider the 1 nm gap numerically converged.
# This gives us a much more meaningful secondary check.
###############################################################################

echo
echo "######################################################################"
echo "PART C: REFINED 1 nm GAP CHECK"
echo "######################################################################"

run_case \
    "1" \
    "dirichlet" \
    "2" \
    "300" \
    "300" \
    "gap_001nm/bc_dirichlet/h2"

run_case \
    "1" \
    "neumann" \
    "2" \
    "300" \
    "300" \
    "gap_001nm/bc_neumann/h2"


###############################################################################
# PART D
#
# Lateral domain sensitivity
#
# Current shaft:
#
#   radius = 120 nm
#   diameter = 240 nm
#
# In a 300 nm box only 30 nm exists between shaft edge and side wall.
#
# Compare:
#
#   300 nm baseline
#   400 nm
#   500 nm
#
# h=5 for screening.
###############################################################################

echo
echo "######################################################################"
echo "PART D: DOMAIN SIZE SENSITIVITY"
echo "######################################################################"

for L in 400 500
do

    run_case \
        "10" \
        "dirichlet" \
        "5" \
        "${L}" \
        "${L}" \
        "domain_${L}nm/gap_010nm/bc_dirichlet/h5"

    run_case \
        "10" \
        "neumann" \
        "5" \
        "${L}" \
        "${L}" \
        "domain_${L}nm/gap_010nm/bc_neumann/h5"

done


###############################################################################
# DONE
###############################################################################

echo
echo "======================================================================"
echo "CAMPAIGN COMPLETE"
echo "======================================================================"
echo
echo "Manifest:"
echo "  ${MANIFEST}"
echo
echo "Results:"
echo "  ${OUTROOT}"
echo
echo "Inspect failed cases with:"
echo
echo "  grep ',[1-9][0-9]*$' ${MANIFEST}"
echo
echo "Successful cases:"
echo
echo "  grep ',0$' ${MANIFEST}"
echo
