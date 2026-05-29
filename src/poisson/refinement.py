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

    Domain: x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2], z in [0, Lz].
    Each RefinementBox drives element size to h_fine inside it; elsewhere h_coarse.
    Multiple boxes are combined with a Gmsh Min field — finest rule wins in overlaps.
    """
    # Scale all lengths to nm to avoid OpenCASCADE precision issues at sub-100nm scales.
    # OCC's default tolerance (~1e-7) breaks geometry smaller than ~200nm in metres.
    s = 1e9  # m → nm

    gmsh.initialize()
    gmsh.model.add("refined_domain")

    vol = gmsh.model.occ.addBox(-Lx*s/2, -Ly*s/2, 0.0, Lx*s, Ly*s, Lz*s)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [vol], 1)
    gmsh.model.setPhysicalName(3, 1, "domain")

    h_fine_global = min(b.h_fine for b in boxes) if boxes else h_coarse

    field_ids = []
    for box in boxes:
        fid = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(fid, "VIn",  box.h_fine * s)
        gmsh.model.mesh.field.setNumber(fid, "VOut", h_coarse   * s)
        gmsh.model.mesh.field.setNumber(fid, "XMin", (box.cx - box.lx / 2) * s)
        gmsh.model.mesh.field.setNumber(fid, "XMax", (box.cx + box.lx / 2) * s)
        gmsh.model.mesh.field.setNumber(fid, "YMin", (box.cy - box.ly / 2) * s)
        gmsh.model.mesh.field.setNumber(fid, "YMax", (box.cy + box.ly / 2) * s)
        gmsh.model.mesh.field.setNumber(fid, "ZMin", (box.cz - box.lz / 2) * s)
        gmsh.model.mesh.field.setNumber(fid, "ZMax", (box.cz + box.lz / 2) * s)
        field_ids.append(fid)

    if len(field_ids) == 0:
        bg = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(bg, "F", str(h_coarse * s))
    elif len(field_ids) == 1:
        bg = field_ids[0]
    else:
        bg = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(bg, "FieldsList", field_ids)

    gmsh.model.mesh.field.setAsBackgroundMesh(bg)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_fine_global * s)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_coarse      * s)

    gmsh.model.mesh.generate(3)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank=rank, gdim=3)
    gmsh.finalize()

    # Scale coordinates back to metres so the rest of the code stays unit-consistent.
    mesh = mesh_data.mesh
    mesh.geometry.x[:] /= s
    return mesh


def mark_refinement_regions(mesh, boxes: list[RefinementBox]):
    """Return a DG0 Function labelling each cell by which box it falls in.

    Value 0 = coarse background; value i (1-indexed) = inside boxes[i-1].
    """
    try:
        V0 = fem.functionspace(mesh, ("DG", 0))
    except AttributeError:
        V0 = fem.FunctionSpace(mesh, ("DG", 0))

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)

    num_cells    = mesh.topology.index_map(tdim).size_local
    cell_to_vtx  = mesh.topology.connectivity(tdim, 0)
    cell_verts   = cell_to_vtx.array.reshape(num_cells, -1)
    midpoints    = mesh.geometry.x[cell_verts].mean(axis=1)

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
