#!/usr/bin/env bash
set -euo pipefail

BASE=poisson_point_imprint.py

run_case () {
  local Q="$1" sigma="$2" zsem="$3" x0="$4" y0="$5" zq="$6" h="$7" holeR="$8" tag="$9"
  local TMP=poisson_point_imprint_tmp.py
  cp "$BASE" "$TMP"

  # Patch parameters in a temp copy (portable BSD/GNU sed)
  sed -i.bak \
    -e "s/^Q = .*/Q = ${Q}/" \
    -e "s/^sigma = .*/sigma = ${sigma}/" \
    -e "s/^z_sem = .*/z_sem = ${zsem}/" \
    -e "s/^x0, y0, zq = .*/x0, y0, zq = ${x0}, ${y0}, ${zq}/" \
    -e "s/^h = .*/h = ${h}/" \
    -e "s/^hole_R = .*/hole_R = ${holeR}/" \
    "$TMP"

  echo "==> Running: tag=${tag}, Q=${Q}, sigma=${sigma}, z_sem=${zsem}, x0,y0,zq=${x0},${y0},${zq}, h=${h}, hole_R=${holeR}"
  ./run_dolfinx.sh "$TMP"

  # Stash outputs with unique names
  mv results3d.xdmf "results3d_${tag}.xdmf" 2>/dev/null || true
  [ -f results3d.h5 ] && mv results3d.h5 "results3d_${tag}.h5"
  for f in imprint_z*.csv; do
    [ -f "$f" ] && mv "$f" "imprint_${tag}.csv"
  done

  rm -f "$TMP" "$TMP.bak"
  echo "==> Saved: results3d_${tag}.xdmf (+ .h5 if present), imprint_${tag}.csv"
}

# --------------------
# Example test cases
# --------------------

# 1) Smaller Q, closer plane (stronger imprint), charge slightly below center
run_case 1e-6 0.04 0.15 0.50 0.50 0.35 0.03 0.10 "nearplane_smallQ"

# 2) Lateral offset in x (see asymmetric imprint), plane at your default z=0.20
run_case 1e-6 0.04 0.20 0.60 0.50 0.40 0.03 0.10 "offsetx_centerZ"

# 3) Wider Gaussian (smoother potential), charge higher
run_case 1e-6 0.08 0.20 0.50 0.50 0.45 0.03 0.10 "wider_sigma_higherZ"

# 4) Further plane (weaker imprint) and larger hole radius
run_case 1e-6 0.04 0.25 0.70 0.50 0.40 0.03 0.14 "farplane_bigger_hole"

# 5) Finer mesh (smaller h) to sharpen features at same configuration
run_case 1e-6 0.04 0.20 0.50 0.50 0.40 0.02 0.10 "finer_mesh"

# Uncomment to switch to quick-units mode on the fly (eps0=1.0)
# sed -i.bak -e 's/^eps0 = .*/eps0 = 1.0/' "$BASE"

