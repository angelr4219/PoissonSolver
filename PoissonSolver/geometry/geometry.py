"""
Geometry construction module for Poisson and Laplace problems.

This module contains dataclasses for common geometries and helper
functions to build FEniCSx meshes for those geometries.  Where
appropriate, gmsh is used to generate complex shapes (e.g., a rod
with a cylindrical bore or a rectangular domain with a circular
hole), and the resulting meshes are converted into FEniCSx meshes
using ``gmshio.model_to_mesh``.  Simpler boxes can be created
directly with the native FEniCSx ``mesh.create_box`` routine.

Each builder returns a tuple ``(domain, cell_tags, facet_tags)``.
``domain`` is the FEniCSx mesh, ``cell_tags`` is a ``MeshTags`` or
``MeshValueCollection`` tagging each cell by integer IDs, and
``facet_tags`` tags boundary facets so that boundary conditions can
be applied.

Users are encouraged to create instances of the dataclasses below and
pass them to the corresponding build functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import gmshio
from dolfinx.mesh import meshtags, locate_entities_boundary

@dataclass
class RefinedDiskParams:
    """Parameters for constructing a 3‑D box with a locally refined disk region.

    A circular disk of radius ``R`` at height ``z0`` (both in meters) is used
    to define a distance‑based mesh sizing field.  Near the disk the mesh
    characteristic length is ``h_near = 2*R / n_diam`` and away from the
    disk the size grows to ``h_far = far_h_factor*h_near``.  A band of
    thickness ``refine_band_R * R`` controls the transition between the
    near and far regions.

    Attributes
    ----------
    Lx, Ly, Lz : float
        Dimensions of the surrounding box in metres.
    R : float
        Disk radius in metres.
    z0 : float
        Height of the disk centre (z‑coordinate) in metres.  The disk
        occupies the slab ``|z - z0| <= t/2``.
    n_diam : int
        Target number of elements across the disk diameter.  The near
        element size is computed as ``h_near = 2*R / n_diam``.
    refine_band_R : float
        Thickness of the refinement band in units of the radius.  The
        distance field transitions from near to far sizes over a band
        ``refine_band_R*R``.
    far_h_factor : float
        Factor multiplying ``h_near`` to obtain the far field size.
    """
    Lx: float
    Ly: float
    Lz: float
    R: float
    z0: float
    n_diam: int
    refine_band_R: float
    far_h_factor: float


def build_refined_disk_box(params: RefinedDiskParams,
                           comm: MPI.Comm = MPI.COMM_WORLD
                           ) -> Tuple[mesh.Mesh, meshtags, meshtags, dict]:
    """Build a 3‑D box mesh with a locally refined circular disk region.

    This helper reproduces the functionality of
    ``gmsh_build_box_with_refine_to_disk_surface`` from the legacy script.
    It constructs a Gmsh model with a box and a distance‑based size
    field around a circular disk on a plane ``z=z0``.  After meshing,
    the model is converted to a FEniCSx mesh and returned along with
    empty cell and facet taggings (no physical groups are defined).

    Parameters
    ----------
    params : RefinedDiskParams
        Geometry and refinement parameters.  See the dataclass
        documentation for details.
    comm : MPI.Comm, optional
        MPI communicator for parallel mesh construction and partitioning.

    Returns
    -------
    domain : dolfinx.mesh.Mesh
        The tetrahedral mesh of the domain.
    cell_tags : dolfinx.mesh.MeshTags
        Empty cell tags (no subdomains are defined for this geometry).
    facet_tags : dolfinx.mesh.MeshTags
        Empty facet tags.  Dirichlet boundary conditions can be applied
        using ``locate_entities_boundary`` instead.
    info : dict
        Dictionary containing ``h_near`` and ``h_far`` used for mesh
        sizing.  This replicates the diagnostic information from the
        original script.
    """
    R = params.R
    # Compute target mesh sizes
    h_near = (2.0 * R) / float(params.n_diam)
    h_far = params.far_h_factor * h_near
    info = {"h_near": h_near, "h_far": h_far}

    # Only rank 0 performs the Gmsh CAD operations
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("refined_disk_box")
        occ = gmsh.model.occ
        # Box centred at origin: extend from -Lx/2 to Lx/2 etc.
        x0 = -0.5 * params.Lx
        y0 = -0.5 * params.Ly
        zmin = -0.5 * params.Lz
        box = occ.addBox(x0, y0, zmin, params.Lx, params.Ly, params.Lz)
        # Helper disk surface for distance field; lives at z0
        c = occ.addCircle(0.0, 0.0, params.z0, R)
        cl = occ.addCurveLoop([c])
        disk_surf = occ.addPlaneSurface([cl])
        occ.synchronize()
        # Physical group for volume (optional; id=1)
        gmsh.model.addPhysicalGroup(3, [box], 1)
        gmsh.model.setPhysicalName(3, 1, "bulk")
        # Define distance‑based mesh size field
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", [disk_surf])
        f_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h_near)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", params.refine_band_R * R)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
        # Global characteristic lengths
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * h_near)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)
        gmsh.model.mesh.generate(3)
    # Convert to FEniCSx mesh (domain, cell_tags, facet_tags)
    out = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    domain = out[0]
    # gmshio.model_to_mesh returns empty MeshTags for cell and facet tags
    cell_tags = out[1]
    facet_tags = out[2]
    if comm.rank == 0:
        gmsh.finalize()
    return domain, cell_tags, facet_tags, info


@dataclass
class RodBoreParams:
    """Parameters defining a rod with a cylindrical through‑bore.

    Attributes
    ----------
    Lx, Ly, Lz : float
        Dimensions of the rectangular prism in x, y and z directions.
    R : float
        Radius of the cylindrical bore centred at ``(Lx/2, Ly/2)``.
    h : float
        Target mesh size for gmsh.
    t_si_bot, t_sige, t_si_top : float
        Thicknesses of the bottom, middle (SiGe) and top silicon layers in z.  These
        should sum to ``Lz``.
    """
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 3.0
    R: float = 0.15
    h: float = 0.05
    t_si_bot: float = 1.0
    t_sige: float = 1.0
    t_si_top: float = 1.0


def build_rod_with_bore(params: RodBoreParams,
                        comm: MPI.Comm = MPI.COMM_WORLD
                        ) -> Tuple[mesh.Mesh, meshtags, meshtags]:
    """Construct a 3‑D rod with a cylindrical bore and return the mesh and tags.

    The rod is a rectangular box of dimensions ``Lx×Ly×Lz`` centred at the origin,
    and a cylinder of radius ``R`` is carved out along the z direction.  The
    resulting domain contains two volumes: one representing air (the bore) and
    the other representing solid material.  The solid region is further
    subdivided into three layers along z according to ``t_si_bot``, ``t_sige``
    and ``t_si_top``.

    Returns
    -------
    domain : dolfinx.mesh.Mesh
        The tetrahedral mesh of the domain.
    cell_tags : dolfinx.mesh.MeshTags
        Tags marking each cell by integer ID.  Tag 101 denotes air in the bore,
        and tags 201–203 denote the bottom Si layer, the SiGe layer and the
        top Si layer respectively.
    facet_tags : dolfinx.mesh.MeshTags
        Tags marking boundary facets.  Tag 301 marks the outer boundary.
    """
    # Consistency check on layer thicknesses
    assert abs(params.t_si_bot + params.t_sige + params.t_si_top - params.Lz) < 1e-12, \
        "Layer thicknesses must sum to Lz."

    gmsh.initialize()
    gmsh.model.add("rod_with_bore")

    # Create the base box and the cylindrical bore
    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, params.Lx, params.Ly, params.Lz)
    cyl = gmsh.model.occ.addCylinder(params.Lx / 2, params.Ly / 2, 0.0,
                                     0.0, 0.0, params.Lz, params.R)
    # Fragment to retain both volumes separately
    gmsh.model.occ.fragment([(3, box)], [(3, cyl)])
    gmsh.model.occ.synchronize()

    # Identify the volumes and choose the smaller one as the bore (air)
    vols = gmsh.model.getEntities(dim=3)
    v_tags = [tag for (_, tag) in vols]
    measures: list[tuple[int, float]] = []
    for tag in v_tags:
        _, mass = gmsh.model.occ.getCenterOfMass(3, tag), gmsh.model.occ.getMass(3, tag)
        measures.append((tag, mass))
    measures.sort(key=lambda x: x[1])
    air_tag = measures[0][0]
    solid_tags = [t for t, _ in measures[1:]]

    # Mark the air volume
    gmsh.model.addPhysicalGroup(3, [air_tag], tag=101)
    gmsh.model.setPhysicalName(3, 101, "air")

    # Mark all boundary surfaces as outer boundary (tag 301)
    outer_surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [s for (_, s) in outer_surfaces], tag=301)
    gmsh.model.setPhysicalName(2, 301, "outer")

    # Mesh size control
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), params.h)

    gmsh.model.mesh.generate(3)
    # Convert to FEniCSx mesh
    domain, cell_tags_raw, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0)
    gmsh.finalize()

    # Subdivide the solid region into layers along z: 201–203
    x = domain.geometry.x
    from dolfinx.cpp.mesh import entities_to_geometry  # type: ignore
    n_cells = domain.topology.index_map(domain.topology.dim).size_local
    # Map cell index to its vertices' indices
    cells_as_nodes = entities_to_geometry(domain, domain.topology.dim,
                                          np.arange(n_cells, dtype=np.int32),
                                          False)
    cell_z = np.mean(x[cells_as_nodes, 2], axis=1)

    # Start with the raw tags (101 for air, 0 for solid)
    tags_array = np.zeros(cell_tags_raw.values.shape, dtype=np.int32)
    tags_array[cell_tags_raw.values == 101] = 101

    # Compute z thresholds
    z1 = params.t_si_bot
    z2 = params.t_si_bot + params.t_sige
    solid_idx = np.where(tags_array == 0)[0]
    tags_array[solid_idx[(cell_z[solid_idx] >= 0.0) & (cell_z[solid_idx] < z1)]] = 201
    tags_array[solid_idx[(cell_z[solid_idx] >= z1) & (cell_z[solid_idx] < z2)]] = 202
    tags_array[solid_idx[(cell_z[solid_idx] >= z2) & (cell_z[solid_idx] <= params.Lz)]] = 203

    cell_tags = meshtags(domain, domain.topology.dim,
                         np.arange(len(tags_array), dtype=np.int32), tags_array)
    return domain, cell_tags, facet_tags


@dataclass
class DiskParams:
    """Parameters for a 2‑D rectangular domain with a circular hole."""
    Lx: float = 1.0
    Ly: float = 1.0
    R: float = 0.2
    h: float = 0.05


def build_disk_with_hole(params: DiskParams,
                         comm: MPI.Comm = MPI.COMM_WORLD
                         ) -> Tuple[mesh.Mesh, meshtags, meshtags]:
    """Construct a 2‑D rectangular domain with a centred circular hole.

    The domain is a rectangle of size ``Lx×Ly`` with a circular hole of
    radius ``R`` centred at ``(Lx/2, Ly/2)``.  Boundary facets are
    tagged as follows: tag 1 corresponds to the outer boundary and tag 2
    corresponds to the hole boundary.  All cells receive tag 1.
    """
    gmsh.initialize()
    gmsh.model.add("disk2d")

    # Rectangle and disk
    rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, params.Lx, params.Ly)
    disk = gmsh.model.occ.addDisk(params.Lx / 2, params.Ly / 2, 0.0,
                                  params.R, params.R)
    # Cut the hole out of the rectangle; removeObject removes the rectangle
    gmsh.model.occ.cut([(2, rect)], [(2, disk)], removeObject=True, removeTool=False)
    gmsh.model.occ.synchronize()

    # Tag boundaries: outer = 1, hole = 2; cells = 1
    # Outer boundary includes all edges except the hole edges
    all_bnds = gmsh.model.getBoundary([(2, rect)], oriented=False, recursive=False)
    hole_bnds = gmsh.model.getBoundary([(2, disk)], oriented=False, recursive=False)
    outer_edges = [e for e in all_bnds if e not in hole_bnds]
    gmsh.model.addPhysicalGroup(1, [e[1] for e in outer_edges], tag=1)
    gmsh.model.addPhysicalGroup(1, [e[1] for e in hole_bnds], tag=2)
    gmsh.model.addPhysicalGroup(2, [rect], tag=1)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), params.h)
    gmsh.model.mesh.generate(2)

    domain, cell_tags, facet_tags, *_ = gmshio.model_to_mesh(gmsh.model, comm, 0)
    gmsh.finalize()
    return domain, cell_tags, facet_tags


@dataclass
class GateBoxParams:
    """Parameters for a rectangular box used in gate benchmarks.

    Attributes
    ----------
    Lx, Ly, H : float
        Domain lengths in x, y and z directions.  The top surface lies at z=0 and
        the box extends down to z=-H.
    h : float
        Target mesh size.  This parameter controls either the gmsh element size
        or the number of subdivisions when using a native box mesh.
    use_gmsh : bool
        If True, gmsh will be used to generate the mesh and to tag the top
        surface as physical group 1.  If False, the native FEniCSx ``create_box``
        function will be used and the top facets will be tagged manually.
    """
    Lx: float = 1e-7
    Ly: float = 1e-7
    H: float = 2e-7
    h: float = 5e-9
    use_gmsh: bool = False


def build_gate_box(params: GateBoxParams,
                   comm: MPI.Comm = MPI.COMM_WORLD
                   ) -> Tuple[mesh.Mesh, meshtags, meshtags]:
    """Build a rectangular box mesh for gate benchmarks and tag the top surface.

    The returned boundary facet tags have tag 1 on the top surface (z = 0)
    and tag 0 elsewhere.  If ``use_gmsh`` is True, gmsh is used to generate
    the mesh; otherwise the native FEniCSx ``create_box`` function is used.
    """
    if params.use_gmsh:
        gmsh.initialize()
        gmsh.model.add("gate_box")
        # Box extends from (-Lx/2, -Ly/2, 0) to (Lx/2, Ly/2, H)
        vol = gmsh.model.occ.addBox(-params.Lx / 2, -params.Ly / 2, 0.0,
                                    params.Lx, params.Ly, params.H)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", params.h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", params.h)
        gmsh.model.mesh.generate(3)
        # Tag top surface (z = 0) as physical group 1
        top_faces = []
        for dim, tag in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if abs(com[2] - 0.0) < 1e-12:
                top_faces.append(tag)
        if top_faces:
            gmsh.model.addPhysicalGroup(2, top_faces, tag=1)
        # Convert to FEniCSx mesh
        domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
        gmsh.finalize()
        return domain, cell_tags, facet_tags
    else:
        # Use the native FEniCSx create_box routine.  The mesh is generated
        # directly by DOLFINx, and the top surface facets are tagged
        # manually.  No physical groups are defined on the cells.
        # Compute the number of subdivisions in each coordinate direction
        # based on the target mesh size ``h``.
        nx = max(2, int(np.ceil(params.Lx / params.h)))
        ny = max(2, int(np.ceil(params.Ly / params.h)))
        nz = max(2, int(np.ceil(params.H  / params.h)))
        # Define the bounding box: from (−Lx/2, −Ly/2, 0) to (Lx/2, Ly/2, H)
        p0 = np.array([-params.Lx / 2, -params.Ly / 2, 0.0], dtype=np.double)
        p1 = np.array([ params.Lx / 2,  params.Ly / 2,  params.H], dtype=np.double)
        domain = mesh.create_box(
            comm,
            [p0, p1],
            (nx, ny, nz),
            cell_type=mesh.CellType.tetrahedron,
        )
        # No cell tags for a homogeneous box (tag value 0 on all cells)
        num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
        cell_indices = np.arange(num_cells_local, dtype=np.int32)
        cell_values = np.zeros(num_cells_local, dtype=np.int32)
        cell_tags = meshtags(domain, domain.topology.dim, cell_indices, cell_values)
        # Identify top boundary facets by z coordinate (z ≈ 0) and tag them as 1.
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim - 1, tdim)
        facets = locate_entities_boundary(
            domain,
            tdim - 1,
            lambda x: np.isclose(x[2], 0.0, atol=1e-12),
        )
        facet_values = np.ones(len(facets), dtype=np.int32)
        facet_tags = meshtags(domain, tdim - 1, facets, facet_values)
        return domain, cell_tags, facet_tags


# -----------------------------------------------------------------------------
# Generic mesh loader for arbitrary geometries
# -----------------------------------------------------------------------------

def load_gmsh_mesh(filename: str,
                   dim: int,
                   comm: MPI.Comm = MPI.COMM_WORLD,
                   rank: int = 0
                   ) -> Tuple[mesh.Mesh, meshtags, meshtags]:
    """Load a Gmsh-generated mesh from a ``.msh`` file.

    This utility provides a simple way to import arbitrary geometries created
    with Gmsh into a distributed DOLFINx mesh.  It uses the ``gmshio`` module
    to read the mesh and extract cell and facet markers defined via physical
    groups in the Gmsh model.  The returned objects can be used directly
    with FEniCSx solvers and post‑processing routines.

    Parameters
    ----------
    filename : str
        Path to a ``.msh`` file produced by Gmsh.  The file should contain
        physical groups for cells and (optionally) facets.
    dim : int
        Geometrical dimension of the mesh.  For example, ``dim=3`` for
        tetrahedral or hexahedral meshes, and ``dim=2`` for planar meshes.
    comm : MPI.Comm, optional
        MPI communicator over which to distribute the mesh.  Defaults to
        ``MPI.COMM_WORLD``.
    rank : int, optional
        Rank on which the Gmsh model is initialised when reading the file.
        For serial reading, leave at its default value ``0``.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        The distributed FEniCSx mesh.
    cell_tags : dolfinx.mesh.MeshTags
        Integer tags associated with each cell.  If no cell tags are present
        in the ``.msh`` file, this will be an empty tagging.
    facet_tags : dolfinx.mesh.MeshTags
        Integer tags associated with boundary facets.  If no facet tags are
        present in the ``.msh`` file, this will be an empty tagging.

    Notes
    -----
    This function wraps ``dolfinx.io.gmshio.read_from_msh`` for convenience.
    For large meshes, it is advisable to call this function once and
    subsequently save the mesh in XDMF format to avoid re‑reading the ``.msh``
    file on every run.
    """
    # ``gmshio.read_from_msh`` loads a mesh generated by Gmsh and returns
    # distributed mesh and tags.  It handles partitioning internally.
    domain, cell_tags, facet_tags = gmshio.read_from_msh(
        filename=filename, comm=comm, rank=rank, gdim=dim
    )
    return domain, cell_tags, facet_tags