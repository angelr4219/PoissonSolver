#!/usr/bin/env bash
set -euo pipefail

PYFILE="leah_like_disk_gate_benchmark.py"
IMAGE="ghcr.io/fenics/dolfinx/dolfinx:stable"

# Physical case
R="10e-9"
VGATE="-12.0"
VBOTTOM="12.0"

LX_NM=500
LY_NM=500
LZ_NM=255

LX="500e-9"
LY="500e-9"
LZ="255e-9"

Z_SIGE_TOP="50e-9"
Z_SI_TOP="55e-9"
EPSR_SIGE="12.0"
EPSR_SI="11.7"
SIGMA_TOP_CM2="-2.0e11"

# 4 practical cases
HS=(9 6)

if [[ ! -f "$PYFILE" ]]; then
  echo "ERROR: $PYFILE not found in current directory."
  exit 1
fi

mkdir -p Results

for HNM in "${HS[@]}"; do
  for CELLTYPE in hex tet; do
    NX=$(( (LX_NM + HNM - 1) / HNM ))
    NY=$(( (LY_NM + HNM - 1) / HNM ))
    NZ=$(( (LZ_NM + HNM - 1) / HNM ))

    OUTDIR="Results/${CELLTYPE}_h${HNM}nm_neg12v"
    BASENAME="fields_${CELLTYPE}_h${HNM}nm_neg12v"

    echo "=============================================="
    echo "Running case:"
    echo "  target h = ${HNM} nm"
    echo "  celltype = ${CELLTYPE}"
    echo "  Vgate    = ${VGATE} V"
    echo "  Vbottom  = ${VBOTTOM} V"
    echo "  Nx,Ny,Nz = ${NX},${NY},${NZ}"
    echo "  outdir   = ${OUTDIR}"
    echo "=============================================="

    docker run --rm -it \
      -v "$PWD":/work -w /work \
      "$IMAGE" \
      python3 "$PYFILE" \
        --R "$R" \
        --Vgate "$VGATE" \
        --Vbottom "$VBOTTOM" \
        --Lx "$LX" --Ly "$LY" --Lz "$LZ" \
        --Nx "$NX" --Ny "$NY" --Nz "$NZ" \
        --celltype "$CELLTYPE" \
        --z_sige_top "$Z_SIGE_TOP" \
        --z_si_top "$Z_SI_TOP" \
        --epsr_sige "$EPSR_SIGE" \
        --epsr_si "$EPSR_SI" \
        --sigma_top_cm2="$SIGMA_TOP_CM2" \
        --out "$OUTDIR"

    if [[ -f "${OUTDIR}/fields.h5" ]]; then
      mv "${OUTDIR}/fields.h5" "${OUTDIR}/${BASENAME}.h5"
    fi

    if [[ -f "${OUTDIR}/fields.xdmf" ]]; then
      mv "${OUTDIR}/fields.xdmf" "${OUTDIR}/${BASENAME}.xdmf"
    fi

    python3 - <<PY
from pathlib import Path
p = Path("${OUTDIR}/${BASENAME}.xdmf")
if p.exists():
    txt = p.read_text()
    txt = txt.replace("fields.h5", "${BASENAME}.h5")
    p.write_text(txt)
    print(f"Updated XDMF reference in {p}")
PY

    echo
    echo "Finished: ${OUTDIR}/${BASENAME}.xdmf"
    echo "Finished: ${OUTDIR}/${BASENAME}.h5"
    echo
  done
done

echo "All 4 negative-gate cases completed."
