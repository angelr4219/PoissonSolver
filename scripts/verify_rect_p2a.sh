#!/usr/bin/env bash
set -euo pipefail
: "${A:=70e-9}"
: "${XS:=-1.4e-7,0,1.4e-7}"
: "${VS:=0.25,0.10,0.25}"
: "${TAU:=1e-9}"
: "${FIELD:=phi}"
STEM=phi_p2a

[ -f "Results/${STEM}.xdmf" ] || { echo "[error] Results/${STEM}.xdmf not found" >&2; exit 1; }
mkdir -p Results/verify
echo "[host] running ${STEM} (A=${A} XS=${XS} VS=${VS} TAU=${TAU} FIELD=${FIELD})"

docker run --rm -v "$PWD":/app -w /app \
  -e A="${A}" -e XS="${XS}" -e VS="${VS}" -e TAU="${TAU}" -e FIELD="${FIELD}" \
  -e PYTHONUNBUFFERED=1 \
  dolfinx/dolfinx:stable bash -lc '
set -exo pipefail                            # <- no -u here
: "${A:=70e-9}"; : "${XS:=-1.4e-7,0,1.4e-7}"; : "${VS:=0.25,0.10,0.25}"; : "${TAU:=1e-9}"; : "${FIELD:=phi}"
python -m pip install -q meshio h5py
export PYTHONPATH=/app:$PYTHONPATH
python -m verify.compare_rect_vs_analytic \
  --phi-xdmf="Results/'"$STEM"'.xdmf" \
  --a="$A" --xs="$XS" --Vs="$VS" --tau="$TAU" \
  --field-name="$FIELD" \
  --out-stem="Results/verify/'"$STEM"'"
'
