from __future__ import annotations
from dolfinx.io import XDMFFile

def write_field_xdmf(path, mesh, *fields):
    """
    Write mesh and all scalar fields (Function) to XDMF with t=0.
    """
    with XDMFFile(mesh.comm, path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        t = 0.0
        for f in fields:
            xdmf.write_function(f, t)
