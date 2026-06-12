"""
Sphere geometry generators for the triple-comparison benchmark.

All lengths in metres internally; callers pass nm and we convert.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI


def _nm(x: float) -> float:
    return x * 1e-9


def sphere_refined_box(
    comm: MPI.Comm,
    box_nm: float,
    sphere_r_nm: float,
    h_near_nm: float,
    h_far_nm: float | None = None,
    msh_path: str | Path | None = None,
):
    """
    Gmsh box mesh with a Ball refinement field centred at the origin.

    The sphere is represented as a point-charge source term (no void cut).
    Facet tag 1 = all box walls.

    Returns (msh, cell_tags, facet_tags) in SI units.
    """
    try:
        import gmsh
    except ImportError as e:
        raise RuntimeError("gmsh is required: pip install gmsh") from e
    try:
        from dolfinx.io import gmshio
    except ImportError:
        from dolfinx.io import gmsh as gmshio  # older dolfinx builds

    L = _nm(box_nm)
    R = _nm(sphere_r_nm)
    h_near = _nm(h_near_nm)
    h_far = _nm(h_far_nm) if h_far_nm is not None else 5.0 * h_near

    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("sphere_refined_box")

        box = gmsh.model.occ.addBox(-L, -L, -L, 2 * L, 2 * L, 2 * L)
        gmsh.model.occ.synchronize()

        # Tag all 6 box faces as physical group 1
        surfs = gmsh.model.getBoundary([(3, box)], oriented=False)
        surf_tags = [abs(s[1]) for s in surfs]
        gmsh.model.addPhysicalGroup(2, surf_tags, tag=1)
        gmsh.model.addPhysicalGroup(3, [box], tag=1)

        # Ball refinement field centred at origin
        f_ball = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(f_ball, "Radius", R * 5)
        gmsh.model.mesh.field.setNumber(f_ball, "VIn", h_near)
        gmsh.model.mesh.field.setNumber(f_ball, "VOut", h_far)
        gmsh.model.mesh.field.setNumber(f_ball, "Thickness", R * 3)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_ball)

        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.mesh.generate(3)

        if msh_path is not None:
            gmsh.write(str(msh_path))

    result = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    msh, cell_tags, facet_tags = result[0], result[1], result[2]

    if comm.rank == 0:
        gmsh.finalize()

    msh.name = "sphere_refined_box"
    return msh, cell_tags, facet_tags


def sphere_in_box(
    comm: MPI.Comm,
    box_nm: float,
    sphere_r_nm: float,
    h_near_nm: float,
    h_far_nm: float | None = None,
    msh_path: str | Path | None = None,
):
    """
    Gmsh box with a spherical void cut out (conducting sphere BC).

    Facet tag 1 = box walls, facet tag 2 = sphere surface.

    Returns (msh, cell_tags, facet_tags) in SI units.
    """
    try:
        import gmsh
    except ImportError as e:
        raise RuntimeError("gmsh is required: pip install gmsh") from e
    try:
        from dolfinx.io import gmshio
    except ImportError:
        from dolfinx.io import gmsh as gmshio

    L = _nm(box_nm)
    R = _nm(sphere_r_nm)
    h_near = _nm(h_near_nm)
    h_far = _nm(h_far_nm) if h_far_nm is not None else 5.0 * h_near

    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("sphere_in_box")

        box = gmsh.model.occ.addBox(-L, -L, -L, 2 * L, 2 * L, 2 * L)
        sphere = gmsh.model.occ.addSphere(0, 0, 0, R)
        domain, _ = gmsh.model.occ.cut([(3, box)], [(3, sphere)])
        gmsh.model.occ.synchronize()

        vol_tag = domain[0][1]

        # Identify surfaces by centroid proximity
        surfs = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
        box_surfs, sph_surfs = [], []
        for dim, tag in surfs:
            bb = gmsh.model.occ.getBoundingBox(dim, tag)
            # sphere surface: bounding box is tight around sphere
            extent = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
            if extent < 2.5 * R:
                sph_surfs.append(tag)
            else:
                box_surfs.append(tag)

        gmsh.model.addPhysicalGroup(2, box_surfs, tag=1)
        gmsh.model.addPhysicalGroup(2, sph_surfs, tag=2)
        gmsh.model.addPhysicalGroup(3, [vol_tag], tag=1)

        f_ball = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(f_ball, "Radius", R * 5)
        gmsh.model.mesh.field.setNumber(f_ball, "VIn", h_near)
        gmsh.model.mesh.field.setNumber(f_ball, "VOut", h_far)
        gmsh.model.mesh.field.setNumber(f_ball, "Thickness", R * 3)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_ball)

        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.mesh.generate(3)

        if msh_path is not None:
            gmsh.write(str(msh_path))

    result = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    msh, cell_tags, facet_tags = result[0], result[1], result[2]

    if comm.rank == 0:
        gmsh.finalize()

    msh.name = "sphere_in_box"
    return msh, cell_tags, facet_tags
