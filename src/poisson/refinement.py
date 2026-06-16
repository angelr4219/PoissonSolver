from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx import fem


@dataclass
class RefinementBox:
    """Axis-aligned box region that gets denser meshing.

    All lengths in metres (same units as your domain).
    """
    cx: float
    cy: float
    cz: float
    lx: float
    ly: float
    lz: float
    h_fine: float


def build_refined_mesh_3d(
    comm,
    Lx: float,
    Ly: float,
    Lz: float,
    h_coarse: float,
    boxes: list[RefinementBox],
    rank: int = 0,
):
    """Build a 3-D tetrahedral mesh with local refinement controlled by *boxes*.

    The domain is an axis-aligned box of size Lx × Ly × Lz, centred at
    (0, 0, 0) in x/y and starting at z = 0.  Each RefinementBox drives the
    element size down to *h_fine* inside it; everywhere else the mesh uses
    *h_coarse*.  When boxes overlap the minimum element size wins (Gmsh Min
    field).

    Parameters
    ----------
    comm   : MPI communicator
    Lx, Ly : lateral domain extents (m)
    Lz     : domain height (m)
    h_coarse : background mesh size (m)
    boxes  : list of RefinementBox, one per fine region
    rank   : MPI rank that drives Gmsh (usually 0)

    Returns
    -------
    dolfinx.mesh.Mesh
    """
    # OCC's boolean/construction tolerances assume ~unit-scale geometry; at
    # metre-scale (1e-9 magnitude) coordinates gmsh.model.occ.addBox raises
    # "OpenCASCADE exception".  Build the geometry in nanometre-magnitude
    # numbers instead, then rescale the resulting mesh back to metres
    # (consistent with the fix applied to geom_sphere.py).
    _M_TO_NM = 1e9
    _NM_TO_M = 1e-9

    gmsh.initialize()
    gmsh.model.add("refined_domain")

    vol = gmsh.model.occ.addBox(-Lx * _M_TO_NM / 2, -Ly * _M_TO_NM / 2, 0.0,
                                 Lx * _M_TO_NM, Ly * _M_TO_NM, Lz * _M_TO_NM)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [vol], 1)
    gmsh.model.setPhysicalName(3, 1, "domain")

    h_coarse_nm = h_coarse * _M_TO_NM
    h_fine_global_nm = (min(b.h_fine for b in boxes) if boxes else h_coarse) * _M_TO_NM

    field_ids = []
    for box in boxes:
        fid = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(fid, "VIn",  box.h_fine * _M_TO_NM)
        gmsh.model.mesh.field.setNumber(fid, "VOut", h_coarse_nm)
        gmsh.model.mesh.field.setNumber(fid, "XMin", (box.cx - box.lx / 2) * _M_TO_NM)
        gmsh.model.mesh.field.setNumber(fid, "XMax", (box.cx + box.lx / 2) * _M_TO_NM)
        gmsh.model.mesh.field.setNumber(fid, "YMin", (box.cy - box.ly / 2) * _M_TO_NM)
        gmsh.model.mesh.field.setNumber(fid, "YMax", (box.cy + box.ly / 2) * _M_TO_NM)
        gmsh.model.mesh.field.setNumber(fid, "ZMin", (box.cz - box.lz / 2) * _M_TO_NM)
        gmsh.model.mesh.field.setNumber(fid, "ZMax", (box.cz + box.lz / 2) * _M_TO_NM)
        field_ids.append(fid)

    if len(field_ids) == 0:
        bg = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(bg, "F", str(h_coarse_nm))
    elif len(field_ids) == 1:
        bg = field_ids[0]
    else:
        bg = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(bg, "FieldsList", field_ids)

    gmsh.model.mesh.field.setAsBackgroundMesh(bg)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_fine_global_nm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_coarse_nm)

    gmsh.model.mesh.generate(3)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank=rank, gdim=3)
    gmsh.finalize()

    mesh_data.mesh.geometry.x[:] *= _NM_TO_M
    return mesh_data.mesh


def mark_refinement_regions(mesh, boxes: list[RefinementBox]):
    """Return a DG0 Function labelling each cell by which box it falls in.

    Value 0 means coarse background; value i (1-indexed) means the cell
    centre is inside boxes[i-1].  Overlapping boxes: the highest index wins.
    """
    try:
        V0 = fem.functionspace(mesh, ("DG", 0))
    except AttributeError:
        V0 = fem.FunctionSpace(mesh, ("DG", 0))

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)

    num_cells = mesh.topology.index_map(tdim).size_local
    cell_to_vertex = mesh.topology.connectivity(tdim, 0)
    cell_verts = cell_to_vertex.array.reshape(num_cells, -1)
    midpoints = mesh.geometry.x[cell_verts].mean(axis=1)

    arr = np.zeros(num_cells, dtype=np.float64)
    for i, box in enumerate(boxes, start=1):
        mask = (
            (midpoints[:, 0] >= box.cx - box.lx / 2) &
            (midpoints[:, 0] <= box.cx + box.lx / 2) &
            (midpoints[:, 1] >= box.cy - box.ly / 2) &
            (midpoints[:, 1] <= box.cy + box.ly / 2) &
            (midpoints[:, 2] >= box.cz - box.lz / 2) &
            (midpoints[:, 2] <= box.cz + box.lz / 2)
        )
        arr[mask] = float(i)

    region = fem.Function(V0, name="refinement_region")
    region.x.array[:num_cells] = arr
    region.x.scatter_forward()
    return region
