#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/verify
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 --phi-xdmf results/phi_p3a.xdmf --a 70e-9 [--xs -1.4e-7,0,1.4e-7 --Vs 0.25,0.10,0.25 ...]"
  exit 1
fi
python3 -m verify.compare_rect_vs_analytic "$@"
