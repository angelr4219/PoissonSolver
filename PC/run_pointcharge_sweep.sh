set -euo pipefail

IMG="dolfinx/dolfinx:stable"
DOCKER="docker run --rm -v \"$PWD\":/app -w /app ${IMG} bash -lc"

# Distances from the relevant "border"
# - For point_charge_box & multi_point_charge_box:
#     distance from the TOP Dirichlet boundary (z = +H/2) in a 100 nm box.
# - For jackson_interface_point_charge_box:
#     distance above the dielectric interface at z = 0.
dists_nm=(10 20 30)

H_main=1e-7   # 100 nm, symmetric box centered at z=0
H_jack=8e-8   # 80 nm tall box for the Jackson interface case

for d_nm in "${dists_nm[@]}"; do
    # Convert nm → meters
    d_m="${d_nm}e-9"  # e.g. 10 nm → 10e-9 = 1e-8 m

    # For the homogeneous box tests, place the charge d away from the *top* boundary:
    # top boundary at +H_main/2, so z0 = H_main/2 - d
    z0_main=$(python3 - <<PY
H = ${H_main}
d = ${d_m}
z0 = 0.5*H - d
print(f"{z0:.6e}")
PY
)

    # For the Jackson interface: distance above the interface at z = 0
    z0_jack=$(python3 - <<PY
d = ${d_m}
print(f"{d:.6e}")
PY
)

    echo "=== Running distance ${d_nm} nm ==="
    echo "    -> point/multipoint z0 = ${z0_main} m (inside 100 nm box)"
    echo "    -> Jackson interface  z0 = ${z0_jack} m above interface"

    # ---------- 1) Single point charge box ----------
    ${DOCKER} "
      export PETSC_OPTIONS='-ksp_type cg -pc_type gamg';
      /dolfinx-env/bin/python3 PC/point_charge_box.py \
        --Lx 1e-7 --Ly 1e-7 --H 1e-7 \
        --h 5e-9 \
        --epsr 11.7 \
        --deg 1 \
        --q 1.602176634e-19 \
        --x0 0.0 --y0 0.0 --z0 ${z0_main} \
        --sigma 5e-9 \
        --x-probe 0.0 --y-probe 0.0 \
        --npts 401 \
        --run-root results/point_single_d${d_nm}nm
    "

    # ---------- 2) Multi-point charge box (single charge) ----------
    ${DOCKER} "
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
    ${DOCKER} "
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