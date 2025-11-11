from __future__ import annotations
import numpy as np
from dolfinx import mesh

def rectangle(domain_comm, L, H, nx, ny, cell_type="triangle"):
    ct = mesh.CellType.triangle if cell_type=="triangle" else mesh.CellType.quadrilateral
    return mesh.create_rectangle(
        domain_comm,
        [np.array([-L, -H], dtype=np.float64), np.array([ L,  H], dtype=np.float64)],
        [nx, ny],
        cell_type=ct
    )
