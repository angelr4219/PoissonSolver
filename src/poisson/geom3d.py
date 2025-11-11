from __future__ import annotations
import numpy as np
from dolfinx import mesh

def box(domain_comm, L, H, nx, ny, nz, cell_type="tet"):
    if cell_type == "tet":
        return mesh.create_box(
            domain_comm,
            [np.array([-L,-L,-H], dtype=np.float64), np.array([L,L,H], dtype=np.float64)],
            [nx, ny, nz],
            cell_type=mesh.CellType.tetrahedron
        )
    elif cell_type == "hex":
        return mesh.create_box(
            domain_comm,
            [np.array([-L,-L,-H], dtype=np.float64), np.array([L,L,H], dtype=np.float64)],
            [nx, ny, nz],
            cell_type=mesh.CellType.hexahedron
        )
    else:
        raise ValueError("cell_type must be 'tet' or 'hex'")
