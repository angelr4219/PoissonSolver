#!/usr/bin/env bash
set -euo pipefail

ROOT="Results/10-50scases"
mkdir -p "$ROOT"

for RNM in 10 20 30 40 50; do
  echo "=============================================="
  echo "Running disk radius ${RNM} nm"
  echo "=============================================="

  OUTDIR="${ROOT}/disk_R${RNM}nm_neg1_neg12"
  BASENAME="diskR${RNM}nm"

  docker run --rm -it \
    -v "$PWD":/work -w /work \
    ghcr.io/fenics/dolfinx/dolfinx:stable \
    python3 leah_like_disk_gate_benchmark.py \
      --R "${RNM}e-9" \
      --Vgate -1.0 \
      --Vbottom -12.0 \
      --Lx 500e-9 --Ly 500e-9 --Lz 255e-9 \
      --Nx 84 --Ny 84 --Nz 43 \
      --celltype hex \
      --z_sige_top 50e-9 \
      --z_si_top 55e-9 \
      --epsr_sige 12.0 \
      --epsr_si 11.7 \
      --sigma_top_cm2="-2.0e11" \
      --out "$OUTDIR"

  if [[ -f "${OUTDIR}/fields.xdmf" ]]; then
    mv "${OUTDIR}/fields.xdmf" "${OUTDIR}/${BASENAME}.xdmf"
  fi

  if [[ -f "${OUTDIR}/fields.h5" ]]; then
    mv "${OUTDIR}/fields.h5" "${OUTDIR}/${BASENAME}.h5"
  fi

  python3 - <<PY
from pathlib import Path
p = Path("${OUTDIR}/${BASENAME}.xdmf")
if p.exists():
    txt = p.read_text()
    txt = txt.replace("fields.h5", "${BASENAME}.h5")
    p.write_text(txt)
    print(f"Updated XDMF reference: {p}")
else:
    print(f"Warning: missing {p}")
PY

  echo "Finished ${RNM} nm"
  echo
done

echo "All runs finished."
echo "Outputs are under: ${ROOT}"
