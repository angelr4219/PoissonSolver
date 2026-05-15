from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh

from .common import COMM, RANK, Phys

try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio


# ---------------------------------------------------------------------------
# Structured box mesh (square_gate_stack)
# ---------------------------------------------------------------------------

def build_box_mesh(
    comm: MPI.Comm,
    Lx: float,
    Ly: float,
    H: float,
    h: float,
) -> Tuple[dmesh.Mesh, int, int, int, float, float]:
    """
    Build a structured tetrahedral box mesh.

    Domain: x in [x0, x0+Lx], y in [y0, y0+Ly], z in [0, H]
    where x0 = -Lx/2, y0 = -Ly/2 (centered in x-y).

    Returns (msh, nx, ny, nz, x0, y0).
    """
    x0 = -0.5 * Lx
    y0 = -0.5 * Ly

    nx = max(2, int(math.ceil(Lx / h)))
    ny = max(2, int(math.ceil(Ly / h)))
    nz = max(2, int(math.ceil(H / h)))

    p0 = np.array([x0, y0, 0.0], dtype=np.float64)
    p1 = np.array([x0 + Lx, y0 + Ly, H], dtype=np.float64)

    msh = dmesh.create_box(
        comm, [p0, p1], [nx, ny, nz], cell_type=dmesh.CellType.tetrahedron
    )
    return msh, nx, ny, nz, x0, y0


# ---------------------------------------------------------------------------
# gmsh box with refinement field (charged_disk_box)
# ---------------------------------------------------------------------------

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
    msh_path,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> dict:
    """
    Build a gmsh box mesh with a Box size field refined around the disk region.

    Explicit bounds (xmin/xmax/ymin/ymax/zmin/zmax) override the default centered
    symmetric box.  Only rank-0 calls gmsh; all ranks get the mesh path.

    Returns a dict with keys: h_disk, h_far, xmin, xmax, ymin, ymax, zmin, zmax.
    """
    import gmsh

    _xmin = float(xmin) if xmin is not None else -0.5 * Lx
    _xmax = float(xmax) if xmax is not None else +0.5 * Lx
    _ymin = float(ymin) if ymin is not None else -0.5 * Ly
    _ymax = float(ymax) if ymax is not None else +0.5 * Ly
    _zmin = float(zmin) if zmin is not None else -0.5 * Lz
    _zmax = float(zmax) if zmax is not None else +0.5 * Lz

    h_disk = (2.0 * R) / float(n_diam)
    h_far = far_h_factor * h_disk

    msh_path = Path(msh_path)

    if comm.rank == 0:
        msh_path.parent.mkdir(parents=True, exist_ok=True)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("charged_disk_box")
        occ = gmsh.model.occ

        occ.addBox(_xmin, _ymin, _zmin, _xmax - _xmin, _ymax - _ymin, _zmax - _zmin)
        occ.synchronize()

        vols = gmsh.model.getEntities(dim=3)
        pg_bulk = 1
        gmsh.model.addPhysicalGroup(3, [t for (_, t) in vols], pg_bulk)
        gmsh.model.setPhysicalName(3, pg_bulk, "bulk")

        # Refinement box centered on the reference disk (x/y assumed near origin)
        band_xy = float(refine_band_R) * R
        band_z = max(5.0 * h_disk, 0.5 * float(refine_band_R) * R)

        f_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f_box, "VIn", h_disk)
        gmsh.model.mesh.field.setNumber(f_box, "VOut", h_far)
        gmsh.model.mesh.field.setNumber(f_box, "XMin", -band_xy)
        gmsh.model.mesh.field.setNumber(f_box, "XMax",  band_xy)
        gmsh.model.mesh.field.setNumber(f_box, "YMin", -band_xy)
        gmsh.model.mesh.field.setNumber(f_box, "YMax",  band_xy)
        gmsh.model.mesh.field.setNumber(f_box, "ZMin", float(z0) - band_z)
        gmsh.model.mesh.field.setNumber(f_box, "ZMax", float(z0) + band_z)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_box)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h_disk)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(msh_path))
        gmsh.finalize()

    comm.Barrier()

    return {
        "h_disk": h_disk,
        "h_far": h_far,
        "xmin": _xmin,
        "xmax": _xmax,
        "ymin": _ymin,
        "ymax": _ymax,
        "zmin": _zmin,
        "zmax": _zmax,
    }


def read_gmsh_mesh(
    msh_path,
    comm: MPI.Comm,
    gdim: int = 3,
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    """Read a gmsh .msh file and return (msh, cell_tags, facet_tags)."""
    data = gmshio.read_from_msh(str(msh_path), comm, gdim=gdim)
    if hasattr(data, "mesh"):
        return data.mesh, data.cell_tags, data.facet_tags
    msh, cell_tags, facet_tags = data
    return msh, cell_tags, facet_tags


# ---------------------------------------------------------------------------
# Top-disk 3D electrode (topdisk3d)
# ---------------------------------------------------------------------------

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
    split_z: Optional[float] = None,
    model_name: str = "topdisk3d",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    """
    Build a gmsh 3D box mesh with a top circular Dirichlet disk patch.

    The box runs from z = z_top - Lz to z = z_top, with the disk on the top face.
    If split_z is given (in the same units as Lz, measured from the top), the box
    is fragmented at that depth to allow two dielectric regions.

    Physical groups assigned follow the Phys dataclass tags.
    """
    import gmsh

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

        # Optional dielectric split plane
        if split_z is not None and (0.0 < float(split_z) < Lz):
            z_split = z_top - float(split_z)
            slab = occ.addBox(xmin, ymin, z_split, xmax - xmin, ymax - ymin, z_top - z_split)
            occ.synchronize()
            occ.fragment([(3, dev)], [(3, slab)])
            occ.synchronize()

        # Add disk surface at top face and fragment volumes with it
        disk = occ.addDisk(float(disk_xc), float(disk_yc), z_top, float(disk_R), float(disk_R))
        occ.synchronize()

        vols_now = gmsh.model.getEntities(dim=3)
        out_dimtags, out_map = occ.fragment(vols_now, [(2, disk)])
        occ.synchronize()

        # out_map[-1] holds the entities generated from the disk input
        disk_out = out_map[len(vols_now)]
        disk_surface_ids = [tag for (dim, tag) in disk_out if dim == 2]

        # Tag volumes
        vols = gmsh.model.getEntities(dim=3)
        if split_z is None:
            gmsh.model.addPhysicalGroup(3, [t for (_, t) in vols], phys.VOL0)
            gmsh.model.setPhysicalName(3, phys.VOL0, "vol0")
        else:
            z_split_coord = z_top - float(split_z)
            v0, v1 = [], []
            for (d, vid) in vols:
                bx0, by0, bz0, bx1, by1, bz1 = occ.getBoundingBox(d, vid)
                zc = 0.5 * (bz0 + bz1)
                (v0 if zc >= z_split_coord else v1).append(vid)
            if v0:
                gmsh.model.addPhysicalGroup(3, v0, phys.VOL0)
                gmsh.model.setPhysicalName(3, phys.VOL0, "vol_upper")
            if v1:
                gmsh.model.addPhysicalGroup(3, v1, phys.VOL1)
                gmsh.model.setPhysicalName(3, phys.VOL1, "vol_lower")

        # Tag surfaces
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


# ---------------------------------------------------------------------------
# Rod with cylindrical bore (rod_bore)
# ---------------------------------------------------------------------------

def build_rod_with_bore(
    Lx: float,
    Ly: float,
    Lz: float,
    R: float,
    h: float,
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    """
    Build a rectangular prism (Lx × Ly × Lz) with a cylindrical bore of radius R
    along the z-axis, centred at (Lx/2, Ly/2).

    Physical groups:
      Volume 101 : air (the bore)
      Volume 200 : solid (the prism minus bore)
      Surface 301: all outer boundary faces

    Use tag_rod_layers() to further sub-divide the solid by z-slab.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)

    if RANK == 0:
        gmsh.model.add("rod_with_bore")
        occ = gmsh.model.occ

        box = occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
        cyl = occ.addCylinder(0.5 * Lx, 0.5 * Ly, 0.0, 0.0, 0.0, Lz, R)

        # Fragment: keeps both volumes as separate subdomains
        ov, _ = occ.fragment([(3, box)], [(3, cyl)])
        occ.synchronize()

        # Identify air volume (cylinder bore) by smallest volume
        vols = gmsh.model.getEntities(dim=3)
        v_mass = [(tag, occ.getMass(3, tag)) for (_, tag) in vols]
        v_mass.sort(key=lambda x: x[1])
        air_vol = v_mass[0][0]
        solid_vols = [tag for (tag, _) in v_mass[1:]]

        gmsh.model.addPhysicalGroup(3, [air_vol], tag=101)
        gmsh.model.setPhysicalName(3, 101, "air")
        gmsh.model.addPhysicalGroup(3, solid_vols, tag=200)
        gmsh.model.setPhysicalName(3, 200, "solid")

        # Tag all outer boundary surfaces
        surfs = gmsh.model.getEntities(dim=2)
        gmsh.model.addPhysicalGroup(2, [s for (_, s) in surfs], tag=301)
        gmsh.model.setPhysicalName(2, 301, "outer")

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]
    gmsh.finalize()
    return msh, ct, ft
