from __future__ import annotations
import numpy as np
from mpi4py import MPI
from dolfinx import mesh

def build_box(Lx: float, Ly: float, H: float, h: float) -> mesh.Mesh:
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    nz = max(2, int(np.ceil(H  / h)))
    p0 = np.array([-Lx/2, -Ly/2, 0.0], dtype=np.double)
    p1 = np.array([ Lx/2,  Ly/2,  H ], dtype=np.double)
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], (nx, ny, nz),
                           cell_type=mesh.CellType.tetrahedron)
