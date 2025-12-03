#!/usr/bin/env bash
set -euo pipefail

IMG="dolfinx/dolfinx:stable"

# Distances from the relevant "border" (in nm)
dists_nm=(10 20 30)

L_main=1e-7   # 100 nm cubic box for point_charge_box
H_main=1e-7   # 100 nm tall 3D box for multi_point_charge_box
H_jack=8e-8   # 80 nm tall box for Jackson interface case

for d_nm in "${dists_nm[@]}"; do
    # Convert nm → meters
    d_m="${d_nm}e-9"  # e.g. 10 nm → 10e-9 = 1e-8 m

    # ----- For 3D boxes: distance d from top boundary z = +L/2 or +H/2 -----
    z0_main=$(python3 - <<PY
H = ${H_main}
d = ${d_m}
z0 = 0.5*H - d      # top boundary at +H/2
print(f"{z0:.6e}")
PY
)

    # ----- Jackson interface: distance d above interface at z=0 -----
    z0_jack=$(python3 - <<PY
d = ${d_m}
print(f"{d:.6e}")
PY
)

    echo "=== Running distance ${d_nm} nm ==="
    echo "    -> point_charge_box: x0 = (0, 0, ${z0_main}) in cube of side L=${L_main}"
    echo "    -> multi_point_charge_box: z0 = ${z0_main} m inside 100 nm box"
    echo "    -> Jackson interface      : z0 = ${z0_jack} m above interface"

    # ---------- 1) 3D point_charge_box (single smeared charge in a cube) ----------
    docker run --rm -v "$PWD":/app -w /app "$IMG" bash -lc "
      /dolfinx-env/bin/python3 PC/point_charge_box.py \
        --L ${L_main} \
        --h 5e-9 \
        --epsr 11.7 \
        --deg 1 \
        --q 1.602176634e-19 \
        --x0 0.0,0.0,${z0_main} \
        --sigma 5e-9 \
        --bc dirichlet \
        --outdir results/point_single_d${d_nm}nm
    "

    # ---------- 2) 3D multi-point charge box (single charge) ----------
    docker run --rm -v "$PWD":/app -w /app "$IMG" bash -lc "
      export PETSC_OPTIONS='-ksp_type cg -pc_type gamg';
      /dolfinx-env/bin/python3 PC/multi_point_charge_box.py \
        --Lx 1e-7 --Ly 1e-7 --H 1e-7 \
        --h 5e-9 \
        --epsr 11.7 \
        --deg 1 \
        --q '[1.602176634e-19]' \
        --x0 '[0.0]' --y0 '[0.0]' --z0 '[${z0_main}]' \
        --sigma 5e-9 \
        --x-probe 0.0 --y-probe 0.0 \
        --npts 401 \
        --run-root results/multi_point_d${d_nm}nm
    "

    # ---------- 3) Jackson dielectric interface ----------
    docker run --rm -v "$PWD":/app -w /app "$IMG" bash -lc "
      export PETSC_OPTIONS='-ksp_type cg -pc_type gamg';
      /dolfinx-env/bin/python3 PC/jackson_interface_point_charge_box.py \
        --Lx 8e-8 --Ly 8e-8 --H 8e-8 --h 5e-9 \
        --epsr-top 3.9 --epsr-bot 11.7 \
        --q '[1.602176634e-19]' \
        --x0 '[0.0]' --y0 '[0.0]' --z0 '[${z0_jack}]' \
        --sigma 5e-9 \
        --deg 1 \
        --x-probe 0.0 --y-probe 0.0 \
        --npts 401 \
        --run-root results/jackson_interface_d${d_nm}nm \
        --basename jackson_interface
    "

done
