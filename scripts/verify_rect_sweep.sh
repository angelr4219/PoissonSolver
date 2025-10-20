#!/usr/bin/env bash
set -euo pipefail

# analytic params (env-overridable)
A="${A:-70e-9}"
XS="${XS:--1.4e-7,0,1.4e-7}"
VS="${VS:-0.25,0.10,0.25}"
TAU="${TAU:-1e-9}"
FIELD="${FIELD:-phi}"

mkdir -p Results/verify

docker run --rm -it -v "$PWD":/app -w /app dolfinx/dolfinx:stable bash -lc "
set -euo pipefail
python -m pip install -q meshio h5py
export PYTHONPATH=/app:\$PYTHONPATH

# loop through every phi_*.xdmf under Results/cases/**/
shopt -s nullglob
for x in Results/cases/*/phi_*.xdmf; do
  stem=\${x%.xdmf}
  tag=\$(basename \"\$stem\")     # e.g., phi_p3a_h120x120x64
  out_stem=\"Results/verify/\$tag\"
  echo \"[verify] \$x -> \$out_stem\"
  python -m verify.compare_rect_vs_analytic \
    --phi-xdmf=\"\$x\" \
    --a=\"${A}\" \
    --xs=\"${XS}\" \
    --Vs=\"${VS}\" \
    --tau=\"${TAU}\" \
    --field-name=\"${FIELD}\" \
    --out-stem=\"\$out_stem\"
done
echo \"[verify] done -> Results/verify/*_summary.csv, *_rel_error_*.xdmf\"
"
