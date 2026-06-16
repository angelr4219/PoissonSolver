"""Sphere-related mesh generators using gmsh.

Two geometry variants:
  sphere_refined_box  – plain box with a Ball field for local refinement near origin
  sphere_in_box       – box with a spherical void cut out (conducting sphere geometry)

All length inputs are in nanometres.  The OCC geometry kernel is built directly
in nanometre-magnitude coordinates (NOT pre-converted to metres): OCC's boolean
operations (cut/fuse) use an absolute tolerance tuned for ~unit-scale geometry,
and constructing boxes/spheres with 1e-9-scale coordinates causes spurious
"BOPAlgo_AlertBuilderFailed" boolean-cut failures.  The mesh is rescaled to
metres (`domain.geometry.x *= 1e-9`) immediately after gmsh hands it to dolfinx,
so callers still receive an SI-unit mesh as before.
"""
from __future__ import annotations
import numpy as np
import gmsh
try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio

_NM_TO_M = 1e-9


def sphere_refined_box(
    comm,
    L_nm: float,
    R_ref_nm: float,
    h_fine_nm: float,
    h_coarse_nm: float,
):
    """Cubic box [-L/2, L/2]³ with spherical mesh refinement zone at origin.

    The mesh is fine (h_fine) inside a sphere of radius R_ref and relaxes
    to h_coarse at the box boundary.  Use this when the source is a
    regularised point charge (no physical sphere surface).

    Parameters
    ----------
    L_nm       : box side length [nm]
    R_ref_nm   : radius of the fine-mesh zone [nm]
    h_fine_nm  : target element size inside the sphere [nm]
    h_coarse_nm: target element size at the box surface [nm]

    Returns
    -------
    domain, cell_tags, facet_tags
        Outer box faces are all tagged 1.  Mesh geometry is in metres.
    """
    L, R, hf, hc = L_nm, R_ref_nm, h_fine_nm, h_coarse_nm

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sphere_refined_box")

    vol = gmsh.model.occ.addBox(-L/2, -L/2, -L/2, L, L, L)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [vol], tag=1)
    surfs = [e[1] for e in gmsh.model.getEntities(dim=2)]
    gmsh.model.addPhysicalGroup(2, surfs, tag=1)
    gmsh.model.setPhysicalName(2, 1, "outer_boundary")

    fid = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(fid, "Radius",    R)
    gmsh.model.mesh.field.setNumber(fid, "VIn",       hf)
    gmsh.model.mesh.field.setNumber(fid, "VOut",      hc)
    gmsh.model.mesh.field.setNumber(fid, "Thickness", R * 0.5)
    gmsh.model.mesh.field.setAsBackgroundMesh(fid)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hf)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hc)
    gmsh.model.mesh.generate(3)

    domain, cell_tags, facet_tags, *_ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    gmsh.finalize()
    domain.geometry.x[:] *= _NM_TO_M
    return domain, cell_tags, facet_tags


def sphere_in_box(
    comm,
    L_nm: float,
    R_nm: float,
    h_fine_nm: float,
    h_coarse_nm: float,
):
    """Cubic box [-L/2, L/2]³ with a spherical void cut at the origin.

    The void represents a conducting sphere.  Apply φ = V₀ as a Dirichlet
    BC on facet tag 2, and the desired BC (Dirichlet or periodic) on tag 1.

    The mesh is refined near the sphere surface using a Ball field centred
    on the sphere (radius 3R, so fine elements cover ≈ the near-field).

    Parameters
    ----------
    L_nm       : box side length [nm]
    R_nm       : sphere radius [nm]
    h_fine_nm  : element size near the sphere surface [nm]
    h_coarse_nm: element size at the box surface [nm]

    Returns
    -------
    domain, cell_tags, facet_tags
        Box faces tagged 1; sphere surface tagged 2.  Mesh geometry is in metres.
    """
    L, R, hf, hc = L_nm, R_nm, h_fine_nm, h_coarse_nm

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sphere_in_box")

    box    = gmsh.model.occ.addBox(-L/2, -L/2, -L/2, L, L, L)
    sphere = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R)
    out_vols, _ = gmsh.model.occ.cut(
        [(3, box)], [(3, sphere)], removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()

    vol_tag = out_vols[0][1]
    gmsh.model.addPhysicalGroup(3, [vol_tag], tag=1)

    # Classify surfaces by centroid distance from origin (nm-scale here).
    box_surfs, sph_surfs = [], []
    for _, stag in gmsh.model.getEntities(dim=2):
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, stag)
        r_com = np.sqrt(cx**2 + cy**2 + cz**2)
        if r_com < L * 0.4:
            sph_surfs.append(stag)
        else:
            box_surfs.append(stag)

    gmsh.model.addPhysicalGroup(2, box_surfs, tag=1)
    gmsh.model.setPhysicalName(2, 1, "outer_box")
    gmsh.model.addPhysicalGroup(2, sph_surfs, tag=2)
    gmsh.model.setPhysicalName(2, 2, "sphere_surface")

    # Fine mesh in near-field (radius 3R from origin)
    fid = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(fid, "Radius",    R * 3.0)
    gmsh.model.mesh.field.setNumber(fid, "VIn",       hf)
    gmsh.model.mesh.field.setNumber(fid, "VOut",      hc)
    gmsh.model.mesh.field.setNumber(fid, "Thickness", R * 1.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(fid)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hf)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hc)
    gmsh.model.mesh.generate(3)

    domain, cell_tags, facet_tags, *_ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    gmsh.finalize()
    domain.geometry.x[:] *= _NM_TO_M
    return domain, cell_tags, facet_tags
