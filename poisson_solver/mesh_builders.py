from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import math

import numpy as np
from mpi4py import MPI
import gmsh
from dolfinx import mesh as dmesh

try:
    from dolfinx.io import gmshio as gmshio
except Exception:
    from dolfinx.io import gmsh as gmshio

from .common import COMM, RANK, Phys


def build_mesh_topdisk_3d(
    Lx: float,
    Ly: float,
    z_top: float,
    Lz: float,
    disk_xc: float,
    disk_yc: float,
    disk_R: float,
    h: float,
    phys: Phys,
    split_z: Optional[float],
    model_name: str = "topdisk3d",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    if RANK == 0:
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        xmin, xmax = -0.5 * Lx, 0.5 * Lx
        ymin, ymax = -0.5 * Ly, 0.5 * Ly
        z_bot = z_top - Lz

        dev = occ.addBox(xmin, ymin, z_bot, xmax - xmin, ymax - ymin, Lz)
        occ.synchronize()

        if split_z is not None and (0.0 < split_z < Lz):
            z_split = z_top - split_z
            slab = occ.addBox(xmin, ymin, z_split, xmax - xmin, ymax - ymin, z_top - z_split)
            occ.synchronize()
            occ.fragment([(3, dev)], [(3, slab)])
            occ.synchronize()

        disk = occ.addDisk(disk_xc, disk_yc, z_top, disk_R, disk_R)
        occ.synchronize()

        vols_now = gmsh.model.getEntities(dim=3)
        _, out_map = occ.fragment(vols_now, [(2, disk)])
        occ.synchronize()

        disk_out = out_map[len(vols_now)]
        disk_surface_ids = [tag for (dim, tag) in disk_out if dim == 2]

        vols = gmsh.model.getEntities(dim=3)
        if split_z is None:
            gmsh.model.addPhysicalGroup(3, [t for (_, t) in vols], phys.VOL0)
            gmsh.model.setPhysicalName(3, phys.VOL0, "vol0")
        else:
            z_split = z_top - split_z
            v0, v1 = [], []
            for (d, vid) in vols:
                bx0, by0, bz0, bx1, by1, bz1 = occ.getBoundingBox(d, vid)
                zc = 0.5 * (bz0 + bz1)
                (v0 if zc >= z_split else v1).append(vid)

            if v0:
                gmsh.model.addPhysicalGroup(3, v0, phys.VOL0)
                gmsh.model.setPhysicalName(3, phys.VOL0, "vol_upper")
            if v1:
                gmsh.model.addPhysicalGroup(3, v1, phys.VOL1)
                gmsh.model.setPhysicalName(3, phys.VOL1, "vol_lower")

        surfs = gmsh.model.getEntities(dim=2)
        top_ids, bot_ids, side_ids = [], [], []

        tolz = max(1e-6 * max(Lx, Ly, Lz, 1.0), 0.25 * h)
        disk_set = set(disk_surface_ids)

        for (dim, sid) in surfs:
            bx0, by0, bz0, bx1, by1, bz1 = occ.getBoundingBox(dim, sid)
            is_top = (abs(bz0 - z_top) < tolz) and (abs(bz1 - z_top) < tolz)
            is_bot = (abs(bz0 - z_bot) < tolz) and (abs(bz1 - z_bot) < tolz)

            if is_top:
                if sid in disk_set:
                    continue
                top_ids.append(sid)
            elif is_bot:
                bot_ids.append(sid)
            else:
                side_ids.append(sid)

        if disk_surface_ids:
            gmsh.model.addPhysicalGroup(2, disk_surface_ids, phys.TOPDISK)
            gmsh.model.setPhysicalName(2, phys.TOPDISK, "top_disk")
        else:
            print("[WARN] Disk fragment produced 0 surface ids.")

        if top_ids:
            gmsh.model.addPhysicalGroup(2, top_ids, phys.TOP)
            gmsh.model.setPhysicalName(2, phys.TOP, "top_ground")
        if bot_ids:
            gmsh.model.addPhysicalGroup(2, bot_ids, phys.BOTTOM)
            gmsh.model.setPhysicalName(2, phys.BOTTOM, "bottom")
        if side_ids:
            gmsh.model.addPhysicalGroup(2, side_ids, phys.SIDES)
            gmsh.model.setPhysicalName(2, phys.SIDES, "sides")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]
    gmsh.finalize()
    return msh, ct, ft


def gmsh_build_box_with_refinement_field(
    comm: MPI.Comm,
    Lx: float,
    Ly: float,
    Lz: float,
    R: float,
    z0: float,
    n_diam: int,
    refine_band_R: float,
    far_h_factor: float,
    msh_path: Path,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
):
    if comm.rank != 0:
        return None

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("charged_disk_box")
    occ = gmsh.model.occ

    if xmin is None or xmax is None:
        xmin = -0.5 * Lx
        xmax = 0.5 * Lx
    if ymin is None or ymax is None:
        ymin = -0.5 * Ly
        ymax = 0.5 * Ly
    if zmin is None or zmax is None:
        zmin = -0.5 * Lz
        zmax = 0.5 * Lz

    box = occ.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin)

    c = occ.addCircle(0.0, 0.0, z0, R)
    cl = occ.addCurveLoop([c])
    disk_surf = occ.addPlaneSurface([cl])

    occ.synchronize()

    pg_bulk = 1
    gmsh.model.addPhysicalGroup(3, [box], pg_bulk)
    gmsh.model.setPhysicalName(3, pg_bulk, "bulk")

    h_disk = (2.0 * R) / float(n_diam)
    h_far = far_h_factor * h_disk

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", [disk_surf])

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h_disk)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", refine_band_R * R)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h_disk)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)

    gmsh.model.mesh.generate(3)

    msh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    return dict(
        h_disk=h_disk,
        h_far=h_far,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        zmin=zmin, zmax=zmax,
    )


def read_gmsh_mesh(msh_path: Path, comm: MPI.Comm, gdim: int = 3):
    out = gmshio.read_from_msh(str(msh_path), comm, gdim=gdim)

    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            return out[0], out[1], out[2]
        raise RuntimeError(f"Unexpected read_from_msh tuple length: {len(out)}")

    if hasattr(out, "mesh"):
        msh = out.mesh
        ct = getattr(out, "cell_tags", None)
        ft = getattr(out, "facet_tags", None)
        if msh is None:
            raise RuntimeError("read_from_msh returned object without mesh")
        return msh, ct, ft

    raise RuntimeError(f"Unsupported read_from_msh return type: {type(out)}")


def build_box_mesh(comm: MPI.Comm, Lx: float, Ly: float, H: float, h: float):
    from dolfinx import mesh

    x0, y0 = -Lx / 2.0, -Ly / 2.0
    nx = max(2, int(math.ceil(Lx / h)))
    ny = max(2, int(math.ceil(Ly / h)))
    nz = max(2, int(math.ceil(H / h)))

    domain = mesh.create_box(
        comm,
        [
            np.array([x0, y0, 0.0], dtype=np.float64),
            np.array([x0 + Lx, y0 + Ly, H], dtype=np.float64),
        ],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain, nx, ny, nz, x0, y0


def build_rod_with_bore(
    Lx: float,
    Ly: float,
    Lz: float,
    R: float,
    h: float,
):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    if RANK == 0:
        gmsh.model.add("rod_with_bore")
        occ = gmsh.model.occ

        box = occ.addBox(0, 0, 0, Lx, Ly, Lz)
        cyl = occ.addCylinder(Lx / 2, Ly / 2, 0, 0, 0, Lz, R)

        occ.fragment([(3, box)], [(3, cyl)])
        occ.synchronize()

        vols = gmsh.model.getEntities(dim=3)
        v_tags = [tag for (_, tag) in vols]
        v_measures = []
        for tag in v_tags:
            mass = gmsh.model.occ.getMass(3, tag)
            v_measures.append((tag, mass))
        v_measures.sort(key=lambda x: x[1])

        air_vol = v_measures[0][0]
        gmsh.model.addPhysicalGroup(3, [air_vol], tag=101)
        gmsh.model.setPhysicalName(3, 101, "air")

        outer_surfs = gmsh.model.getEntities(dim=2)
        gmsh.model.addPhysicalGroup(2, [s for (_, s) in outer_surfs], tag=301)
        gmsh.model.setPhysicalName(2, 301, "outer")

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        gmsh.model.mesh.generate(3)

    domain, ct, ft = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    gmsh.finalize()
    return domain, ct, ft
