#!/usr/bin/env bash
set -euo pipefail

# ---------- knobs you can edit ----------
A=${A:-70e-9}
ZFILL=${ZFILL:-2.0e-7}
XS=${XS:-'[-7.0e-8,0.0,7.0e-8]'}
VS=${VS:-'[0.25,0.10,0.25]'}
H_ELEM=${H_ELEM:-5e-9}
PADS=${PADS:-"2 3 4 5"}
DEGREES=${DEGREES:-"1 2 3"}      # plus a DG0 projection file for each run
IMAGE=${IMAGE:-dolfinx/dolfinx:stable}
# ---------------------------------------

STAMP=$(date +"%Y%m%d-%H%M%S")
RUN_ROOT="results/runs/${STAMP}"
STAGEA="${RUN_ROOT}/stageA_phi_pXa"
STAGEB="${RUN_ROOT}/stageB_prefine"
CMPDIR="${RUN_ROOT}/compare_json"
PLOTDIR="${RUN_ROOT}/plots"
mkdir -p "${STAGEA}" "${STAGEB}" "${CMPDIR}" "${PLOTDIR}"

echo "==> Run root: ${RUN_ROOT}"

# ------------- STAGE A: original phi_pXa (deg=1) -------------
for p in ${PADS}; do
  tag="p${p}a_deg1"
  echo "---- Stage A: ${tag} ----"
  docker run --rm -v "$PWD":/app -w /app "${IMAGE}" bash -lc "
/dolfinx-env/bin/python3 rect_gate_benchmark.py \
  --deg 1 --a ${A} --zfill ${ZFILL} --xs '${XS}' --Vs '${VS}' \
  --h ${H_ELEM} --pad ${p} --run-root '${STAGEA}'"
  # optional legacy copies
  cp -f "${STAGEA}/${tag}.xdmf" "results/phi_p${p}a.xdmf" || true
  cp -f "${STAGEA}/${tag}.h5"   "results/phi_p${p}a.h5"   2>/dev/null || true
done

# ------------- STAGE C: compare vs analytic by reading XDMFs with meshio -------------
# Computes volume-weighted L2 and Linf. Supports CG1 (point_data "phi") and DG0 (cell_data "phi_dg0").

  for f in ${STAGEA}/p*a_deg1.xdmf ${STAGEB}/p*a_deg*.xdmf; do
  base=$(basename "$f" .xdmf)
  out="${CMPDIR}/${base}.json"

  docker run --rm \
    -e A="${A}" -e XS="${XS}" -e VS="${VS}" -e IN="$f" -e OUT="$out" \
    -v "$PWD":/app -w /app "$IMAGE" bash -lc 'python -m pip install --quiet meshio && python - <<PY


# -------- inputs ----------
infile  = os.environ["IN"]
outfile = os.environ["OUT"]
a       = float(os.environ["A"])
XS      = np.array(json.loads(os.environ["XS"]), dtype=float)
VS      = np.array(json.loads(os.environ["VS"]), dtype=float)

def g_uvz(u, v, z):
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def phi0_rect_local(x, y, z, a, xs, Vs):
    z = z if z != 0.0 else np.finfo(float).eps
    s = 0.0
    for xi, Vi in zip(xs, Vs):
        s += Vi * ( g_uvz(a - xi + x, a + y, z) +
                    g_uvz(a - xi + x, a - y, z) +
                    g_uvz(a + xi - x, a + y, z) +
                    g_uvz(a + xi - x, a - y, z) )
    return -s

def tet_volume(pts):
    # pts: (4,3) vertices
    v0 = pts[1]-pts[0]; v1 = pts[2]-pts[0]; v2 = pts[3]-pts[0]
    return abs(np.linalg.det(np.stack([v0,v1,v2],axis=1)))/6.0

m = meshio.read(infile)

# Try to find tetra cells
tet = None
for block in m.cells:
    if block.type in ("tetra", "tetra10", "tetra4"):
        tet = block
        break
if tet is None:
    raise RuntimeError("No tetrahedral cells found in " + infile)

conn = tet.data  # (ncells, 4) for linear tets
if conn.shape[1] != 4:
    # if this is a high-order tet in XDMF, keep the first 4 as the linear vertices
    conn = conn[:, :4]

X = m.points                       # (npoints, 3)
nc = conn.shape[0]
npnts = X.shape[0]

# Identify field
field_name = None
is_point   = False
if "phi" in m.point_data:
    field_name = "phi";  is_point = True
elif "phi_dg0" in (m.cell_data.get("phi_dg0", {}) if isinstance(m.cell_data, dict) else {}):
    field_name = "phi_dg0"; is_point = False
else:
    # meshio stores cell_data as dict(name)->list aligned with m.cells; search it
    if hasattr(m, "cell_data"):
        for name, lst in m.cell_data.items():
            if name == "phi_dg0":
                field_name = "phi_dg0"; is_point = False; break
if field_name is None and "phi_dg0" in m.cell_data_dict:
    field_name = "phi_dg0"; is_point = False
if field_name is None and "phi" in m.point_data:
    field_name = "phi"; is_point = True

if field_name is None:
    raise RuntimeError("Could not find field 'phi' or 'phi_dg0' in " + infile)

# Compute metrics
if is_point:
    # CG1: nodal values with point_data["phi"]
    u = np.array(m.point_data["phi"]).reshape(-1)
    # build lumped volumes per node: sum over incident tets of (vol/4)
    lump = np.zeros(npnts)
    diffsq = np.zeros(npnts)
    linf = 0.0
    for c in range(nc):
        verts = conn[c]
        pts = X[verts]
        vol = tet_volume(pts)
        # analytic at vertices
        ue = np.array([phi0_rect_local(*X[i], a, XS, VS) for i in verts])
        du = u[verts] - ue
        linf = max(linf, float(np.max(np.abs(du))))
        # distribute
        for j,i in enumerate(verts):
            lump[i] += vol/4.0
    # Now compute squared diff again (need ue at each vertex) and accumulate with lump
    # To avoid re-evaluating many times, do it once:
    ue_all = np.array([phi0_rect_local(*X[i], a, XS, VS) for i in range(npnts)])
    du_all = u - ue_all
    diffsq = (du_all**2) * lump
    L2 = math.sqrt(float(np.sum(diffsq)))
    out = {"kind":"CG1","l2":L2,"linf":linf,"dofs":int(u.size)}
else:
    # DG0: cell_data["phi_dg0"] attached to tets
    # meshio stores cell_data as dict of lists aligned w/ cell blocks
    if isinstance(m.cell_data, dict) and "phi_dg0" in m.cell_data:
        # find the entry aligned with our tet block index
        # build a parallel index for cells
        # simplest: collect phi_dg0 from the same position as tet in m.cells
        idx = [i for i,b in enumerate(m.cells) if b is tet][0]
        ucell = np.asarray(m.cell_data["phi_dg0"][idx]).reshape(-1)
    else:
        # fallback: try cell_data_dict
        ucell = np.asarray(m.cell_data_dict["phi_dg0"][tet.type])
    if ucell.shape[0] != nc:
        raise RuntimeError("DG0 cell_data length mismatch.")
    L2_sq = 0.0
    linf = 0.0
    for c in range(nc):
        verts = conn[c]
        pts = X[verts]
        vol = tet_volume(pts)
        bar = np.mean(pts, axis=0)
        ue  = phi0_rect_local(*bar, a, XS, VS)
        du  = float(ucell[c] - ue)
        L2_sq += (du*du) * vol
        linf   = max(linf, abs(du))
    L2 = math.sqrt(float(L2_sq))
    out = {"kind":"DG0","l2":L2,"linf":linf,"dofs":int(nc)}

# write JSON
pathlib.Path(os.path.dirname(outfile)).mkdir(parents=True, exist_ok=True)
with open(outfile, "w") as fh:
    json.dump(out, fh)
print("Wrote", outfile)
PY'
done

# ------------- STAGE D: aggregate a single CSV from Stage C JSONs -------------
python - <<'PY'
import json, csv, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
rows=[]
for jf in sorted(root.glob("compare_json/*.json")):
    name=jf.stem                      # e.g., p4a_deg2 or p4a_deg1_deg0proj
    m=re.match(r"p(\d+)a_deg(\d+)", name)
    if not m: 
        continue
    pad=int(m.group(1)); deg=int(m.group(2))
    d=json.loads(jf.read_text())
    rows.append({
        "pad_a": pad,
        "degree": deg,
        "tag": name,
        "dofs": d.get("dofs"),
        "L2": d.get("l2"),
        "H1_semi": None,
        "Linf": d.get("linf"),
        "kind": d.get("kind")
    })
rows.sort(key=lambda r:(r["pad_a"], r["degree"], r["tag"]))
out = root / "summary_compare_rect_vs_analytic.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pad_a","degree","tag","kind","dofs","L2","H1_semi","Linf"])
    w.writeheader(); w.writerows(rows)
print("Wrote", out)
PY "${RUN_ROOT}"

echo "==> Done."
echo "Stage A XDMFs : ${STAGEA}/p{2,3,4,5}a_deg1(.xdmf, _deg0proj.xdmf)"
echo "Stage B XDMFs : ${STAGEB}/pXa_deg{1,2,3}(.xdmf, _deg0proj.xdmf)"
echo "Compare JSONs : ${CMPDIR}/*.json"
echo "Summary CSV   : ${RUN_ROOT}/summary_compare_rect_vs_analytic.csv"
