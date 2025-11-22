from __future__ import annotations
import json, csv
import numpy as np
from pathlib import Path
from dolfinx import fem, io

def write_all_in_one(domain, phi, pads_on_top, outprefix, eps_r_field=None):
    V1 = fem.functionspace(domain, ("Lagrange", 1))
    phi1 = fem.Function(V1, name="phi"); phi1.interpolate(phi)
    out = f"{outprefix}.xdmf"
    with io.XDMFFile(domain.comm, out, "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(phi1)
        xf.write_function(pads_on_top)
        if eps_r_field is not None:
            xf.write_function(eps_r_field)

def write_metrics_json(path_without_ext: str, tag: str, deg: int, mets: dict):
    Path(path_without_ext).with_suffix(".json").write_text(json.dumps({
        "tag": tag, "degree": deg, "dofs": mets["dofs"],
        "l2": mets["L2"], "h1s": mets["H1_semi"], "linf": mets["Linf"]
    }, indent=2))

def write_line_csv(path_without_ext: str, xs, uh_line, phi0_line):
    csv_path = f"{path_without_ext}_line.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x_m", "phi_FE_V", "phi0_V"])
        for x, uval, aval in zip(xs, uh_line, phi0_line):
            w.writerow([x, uval, aval])
    return csv_path
