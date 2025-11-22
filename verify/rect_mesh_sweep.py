import os, json
import numpy as np
from rect_gate_benchmark import run_once

a_nm  = float(os.environ.get("A_NM", "35.0"))
deg   = int(os.environ.get("DEG", "2"))
if "H_LIST" in os.environ:
    hs_nm = [float(t) for t in os.environ["H_LIST"].split(",")]
else:
    hs_nm = [20.0, 10.0, 7.5, 5.0, 3.5]
p_list = [2, 3, 4, 5]
H_nm   = float(os.environ.get("HEIGHT_NM", "200.0"))
zbar_nm= float(os.environ.get("ZBAR_NM",  str(a_nm)))

a     = a_nm * 1e-9
H     = H_nm * 1e-9
zbar  = zbar_nm * 1e-9
xs_gates = np.array([-2*a, 0.0, 2*a])
Vs_gates = np.array([0.25, 0.10, 0.25])

print(f"[SWEEP] a={a_nm} nm  deg={deg}  H_list={hs_nm} nm  p={p_list}  zbar={zbar_nm} nm")

for p in p_list:
    Lx = Ly = 2.0*p*a
    tag = f"p{p}a"
    for h_nm in hs_nm:
        h = h_nm * 1e-9
        outdir   = f"results/cases/{tag}"
        os.makedirs(outdir, exist_ok=True)
        outbase  = f"{outdir}/phi_{tag}_h{int(round(h_nm))}nm_p{deg}"
        print(f" -> {tag}: h={h_nm} nm, deg={deg}  -> {outbase}.*")
        err_max, err_l2 = run_once(Lx, Ly, H, h, a, xs_gates, Vs_gates, zbar,
                                   outprefix=outbase, deg=deg)
        with open(outbase + "_run.json", "w") as f:
            json.dump({"p":p, "deg":deg, "h_nm":h_nm,
                       "err_line_max":err_max, "err_line_l2":err_l2}, f)
print("[SWEEP] done.")
