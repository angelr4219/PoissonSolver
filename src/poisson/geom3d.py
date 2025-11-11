from __future__ import annotations
import numpy as np
from dolfinx import mesh

def box(comm, L: float, H: float, nx: int, ny: int, nz: int, cell_type: str = "tet"):
    if cell_type == "tet":
        ct = mesh.CellType.tetrahedron
    elif cell_type == "hex":
        ct = mesh.CellType.hexahedron
    else:
        raise ValueError("cell_type must be 'tet' or 'hex'")
    return mesh.create_box(
        comm,
        [np.array([-L,-L,-H], dtype=np.float64), np.array([L,L,H], dtype=np.float64)],
        [nx, ny, nz],
        cell_type=ct
    )
