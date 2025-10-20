#!/usr/bin/env bash
set -euo pipefail

# params (can override via env)
A=${A:-70e-9}
XS=${XS:--1.4e-7,0,1.4e-7}
VS=${VS:-0.25,0.10,0.25}
TAU=${TAU:-1e-9}
FIELD=${FIELD:-phi}

# first try cases/, else fall back to top-level Results/
FILES="$(find Results/cases -type f -name 'phi_*.xdmf' 2>/dev/null | sort || true)"
if [ -z "${FILES}" ]; then
  FILES="$(find Results -maxdepth 1 -type f -name 'phi_*.xdmf' ! -name '*rel_error*' | sort || true)"
fi

if [ -z "${FILES}" ]; then
  echo "[verify] No phi_*.xdmf files found under Results/ or Results/cases/" >&2
  exit 1
fi

while IFS= read -r x; do
  [ -z "$x" ] && continue
  outstem="${x%.xdmf}"
  echo "[verify] $x -> ${outstem}_*.{xdmf,csv}"

  docker run --rm -v "$PWD":/app -w /app dolfinx/dolfinx:stable bash -lc "
    set -e
    python -m pip install -q meshio h5py
    export PYTHONPATH=/app:\$PYTHONPATH
    python -m verify.compare_rect_vs_analytic \
      --phi-xdmf='$x' \
      --a='$A' \
      --xs='$XS' \
      --Vs='$VS' \
      --tau='$TAU' \
      --field-name='$FIELD' \
      --out-stem='$outstem'
  "
done <<< "${FILES}"
