#!/usr/bin/env bash
# run_all_verification.sh
#
# Run the full verification ladder in the correct order.
# Saves timestamped logs to results/verification_<TIMESTAMP>/.
#
# Usage:
#   bash tests/run_all_verification.sh [--quick|--full]
#
# Make this script executable:
#   chmod +x tests/run_all_verification.sh

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="results/verification_${TIMESTAMP}"
mkdir -p "$LOGDIR"

RUN="./run_dolfinx.sh"

# ---------------------------------------------------------------------------
# Helper: run a test, tee output to log, print pass/fail
# ---------------------------------------------------------------------------
run_test() {
  local name=$1; shift
  local log="$LOGDIR/${name}.log"
  echo "=== $name ===" | tee -a "$LOGDIR/summary.txt"
  if $RUN "tests/${name}.py" "$@" 2>&1 | tee "$log"; then
    echo "  PASS" | tee -a "$LOGDIR/summary.txt"
  else
    echo "  FAIL (see $log)" | tee -a "$LOGDIR/summary.txt"
  fi
}

MODE="${1:---quick}"   # --quick or --full

echo "Verification run: $TIMESTAMP  mode: $MODE" | tee "$LOGDIR/summary.txt"
echo "================================================" | tee -a "$LOGDIR/summary.txt"

# ---------------------------------------------------------------------------
# Tier 0: infrastructure (always run, no Docker needed for feasibility)
# ---------------------------------------------------------------------------
python3 tests/test_mesh_feasibility.py --save-csv --outdir "$LOGDIR/mesh_feasibility" \
  2>&1 | tee "$LOGDIR/test_mesh_feasibility.log"

# ---------------------------------------------------------------------------
# Tier 1: verification ladder
# ---------------------------------------------------------------------------
run_test test_tag_integrity             $MODE --write-xdmf --outdir "$LOGDIR/tag_integrity"
run_test test_h_convergence_manufactured $MODE --write-xdmf --save-csv --outdir "$LOGDIR/h_convergence"
run_test test_p_convergence_manufactured $MODE --write-xdmf --save-csv --outdir "$LOGDIR/p_convergence"
run_test test_bc_comparison_point_charge $MODE --write-xdmf --save-csv --outdir "$LOGDIR/bc_point_charge"
run_test test_bc_comparison_neutral_periodic $MODE --write-xdmf --save-csv --outdir "$LOGDIR/bc_neutral_periodic"
run_test test_fft_vs_fem               $MODE --write-xdmf --save-csv --outdir "$LOGDIR/fft_vs_fem"
run_test test_square_gate_h_convergence $MODE --write-xdmf --save-csv --outdir "$LOGDIR/square_gate"
run_test test_four_square_refinement_box $MODE --write-xdmf --save-csv --outdir "$LOGDIR/refinement_box"
run_test test_probe_stability          $MODE --write-xdmf --save-csv --outdir "$LOGDIR/probe_stability"

echo ""
echo "=== Summary ==="
cat "$LOGDIR/summary.txt"
echo ""
echo "Logs saved to: $LOGDIR"
