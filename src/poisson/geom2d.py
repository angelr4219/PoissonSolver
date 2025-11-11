from __future__ import annotations
import numpy as np
from dolfinx import mesh

def rectangle(comm, L: float, H: float, nx: int, ny: int, cell_type: str = "triangle"):
    ct = mesh.CellType.triangle if cell_type=="triangle" else mesh.CellType.quadrilateral
    return mesh.create_rectangle(
        comm,
        [np.array([-L, -H], dtype=np.float64), np.array([ L,  H], dtype=np.float64)],
        [nx, ny],
        cell_type=ct
    )
