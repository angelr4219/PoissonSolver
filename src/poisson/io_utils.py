from __future__ import annotations
from pathlib import Path
from dolfinx.io import XDMFFile

def write_field_xdmf(path, mesh, *fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(mesh.comm, path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        t = 0.0
        for f in fields:
            xdmf.write_function(f, t)
