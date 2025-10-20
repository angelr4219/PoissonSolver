#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden with export FOO=... before running)
A=${A:-70e-9}
VP1=${VP1:-0.25}
VX=${VX:-0.10}
VP2=${VP2:-0.25}
DEGREE=${DEGREE:-1}

NX_BASE=${NX_BASE:-120}
NY_BASE=${NY_BASE:-120}
NZ_BASE=${NZ_BASE:-64}

PADS=${PADS:-"2 3 4 5"}
SCALES=${SCALES:-"0.75 1.0 1.5"}

echo "[host] A=$A VP1=$VP1 VX=$VX VP2=$VP2 degree=$DEGREE"
echo "[host] base=($NX_BASE,$NY_BASE,$NZ_BASE) pads=($PADS) scales=($SCALES)"

mkdir -p Results/cases

round_int () {
  # usage: round_int FLOAT min_value
  python - "$1" "$2" <<'PY'
import sys, math
x = float(sys.argv[1]); m = int(sys.argv[2])
n = int(round(x))
print(max(m, n))
PY
}

for p in $PADS; do
  for s in $SCALES; do
    nx=$(round_int "$(python - <<PY
print(float("$NX_BASE")*float("$s"))
PY
)" 8)
    ny=$(round_int "$(python - <<PY
print(float("$NY_BASE")*float("$s"))
PY
)" 8)
    nz=$(round_int "$(python - <<PY
print(float("$NZ_BASE")*float("$s"))
PY
)" 4)

    tag="p${p}a_h${nx}x${ny}x${nz}"
    outdir="Results/cases/${tag}"
    echo "[solve] ${tag} (nx,ny,nz)=(${nx},${ny},${nz})"

    docker run --rm -e A="$A" -e VP1="$VP1" -e VX="$VX" -e VP2="$VP2" -e DEGREE="$DEGREE" \
      -e PADDING="$p" -e NX="$nx" -e NY="$ny" -e NZ="$nz" \
      -v "$PWD":/app -w /app dolfinx/dolfinx:stable bash -lc '
set -e
mkdir -p "'"$outdir"'"
python rect_gate_benchmark.py \
  --a "$A" --padding-multiple "$PADDING" \
  --vp1 "$VP1" --vx "$VX" --vp2 "$VP2" \
  --degree "$DEGREE" --nx "$NX" --ny "$NY" --nz "$NZ" \
  --outdir "'"$outdir"'"

# Normalize filenames for consistency
if [ -f "'"$outdir"'/phi.xdmf" ]; then
  cp -f "'"$outdir"'/phi.xdmf" "'"$outdir"'/phi_'"$tag"'.xdmf"
  cp -f "'"$outdir"'/phi.h5"    "'"$outdir"'/phi_'"$tag"'.h5"
elif ls "'"$outdir"'/phi_p*.xdmf" >/dev/null 2>&1; then
  xdmf=$(ls "'"$outdir"'/phi_p*.xdmf" | head -n1)
  h5="${xdmf%.xdmf}.h5"
  cp -f "$xdmf" "'"$outdir"'/phi_'"$tag"'.xdmf"
  cp -f "$h5"   "'"$outdir"'/phi_'"$tag"'.h5"
fi
'
  done
done

echo "[regen] done -> Results/cases/*/phi_*.xdmf"
