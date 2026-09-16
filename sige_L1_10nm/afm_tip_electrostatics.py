#!/usr/bin/env python3
<<<<<<< HEAD
"""3D AFM-tip Laplace solver using Gmsh and DOLFINx.

Coordinates are in nm, potential is in V, and grad(phi) is therefore V/nm.
The conducting tip is removed from the air volume and its cavity surface is
used as a Dirichlet boundary.

Geometry:
- sample box below z=0
- air box above z=0
- conducting AFM tip made from:
    1) spherical apex
    2) conical shank
    3) cylindrical shaft above the cone
=======
"""3D AFM-tip electrostatics using Gmsh + DOLFINx.

Geometry, all lengths in nm:
- Sample surface at z = 0
- Sample occupies -sample_depth <= z <= 0
- Air occupies 0 <= z <= air_height
- Spherical apex radius = tip_radius
- Lowest apex point is gap above z = 0
- Conical shank starts at the sphere center plane
- Shank widens from tip_radius to shank_top_radius
- Cylindrical shaft sits above the shank

The AFM conductor itself is not meshed. Instead, its three geometric pieces
are subtracted sequentially from the air domain. The resulting cavity surface
is tagged TIP and receives the AFM Dirichlet voltage.

PDE:
    div(eps_r grad(phi)) = 0

Volume tags:
    AIR = 1
    SAMPLE = 2

Facet tags:
    TIP = 11
    BOTTOM = 12
    OUTER = 13
    INTERFACE = 14
>>>>>>> 7aa12b7edce39372da0bae7cbab48b3ea8409797
"""

import argparse
import json
from pathlib import Path

import gmsh
import numpy as np
import ufl
from mpi4py import MPI

from dolfinx import default_scalar_type, fem, io, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
<<<<<<< HEAD

try:
    from dolfinx.io import gmsh as gmshio
except ImportError:
    from dolfinx.io import gmshio


# Volume tags
AIR, SAMPLE = 1, 2

# Facet tags
TIP, BOTTOM, OUTER, INTERFACE = 11, 12, 13, 14


def arguments():
    p = argparse.ArgumentParser(description="3D AFM-tip electrostatics")

    # Domain dimensions, in nm
    p.add_argument("--lx", type=float, default=300.0)
    p.add_argument("--ly", type=float, default=300.0)
    p.add_argument("--sample-depth", type=float, default=100.0)
    p.add_argument("--air-height", type=float, default=260.0)

    # AFM-tip geometry, in nm
    p.add_argument("--gap", type=float, default=10.0)
    p.add_argument("--tip-radius", type=float, default=20.0)
    p.add_argument("--cone-height", type=float, default=100.0)
    p.add_argument("--shank-radius", type=float, default=60.0)
    p.add_argument("--shaft-radius", type=float, default=120.0)
    p.add_argument("--shaft-height", type=float, default=100.0)

    # Mesh settings, in nm
    p.add_argument("--h-fine", type=float, default=2.0)
    p.add_argument("--h-coarse", type=float, default=20.0)
    p.add_argument("--refine-dist-min", type=float, default=10.0)
    p.add_argument("--refine-dist-max", type=float, default=80.0)
    p.add_argument("--gmsh-order", type=int, default=1, choices=(1, 2))
    p.add_argument("--gmsh-verbosity", type=int, default=2)

    # Material properties and BCs
    p.add_argument("--eps-air", type=float, default=1.0)
    p.add_argument("--eps-sample", type=float, default=11.7)
    p.add_argument("--tip-voltage", type=float, default=1.0)
    p.add_argument("--bottom-voltage", type=float, default=0.0)
    p.add_argument("--outer-voltage", type=float, default=0.0)
    p.add_argument(
        "--outer-neumann",
        action="store_true",
        help="Use zero-Neumann on side and top boundaries instead of Dirichlet."
    )

    p.add_argument("--degree", type=int, default=1)
    p.add_argument("--mesh-only", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/afm_tip"))
=======
from dolfinx.io import gmsh as gmshio


AIR = 1
SAMPLE = 2

TIP = 11
BOTTOM = 12
OUTER = 13
INTERFACE = 14


def parse_args():
    p = argparse.ArgumentParser(description="3D AFM-tip electrostatics")

    # Domain [nm]
    p.add_argument("--lx", type=float, default=300.0)
    p.add_argument("--ly", type=float, default=300.0)
    p.add_argument("--sample-depth", type=float, default=100.0)
    p.add_argument("--air-height", type=float, default=320.0)

    # AFM geometry [nm]
    p.add_argument("--gap", type=float, default=10.0)
    p.add_argument("--tip-radius", type=float, default=10.0)
    p.add_argument("--shank-height", type=float, default=160.0)
    p.add_argument("--shank-top-radius", type=float, default=120.0)
    p.add_argument("--shaft-radius", type=float, default=120.0)
    p.add_argument("--shaft-height", type=float, default=100.0)

    p.add_argument(
        "--shaft-overlap",
        type=float,
        default=0.5,
        help=(
            "Small overlap [nm] between shaft and shank "
            "for robust cavity construction."
        ),
    )

    # Mesh [nm]
    p.add_argument("--h-fine", type=float, default=5.0)
    p.add_argument("--h-coarse", type=float, default=20.0)
    p.add_argument("--refine-dist-min", type=float, default=10.0)
    p.add_argument("--refine-dist-max", type=float, default=80.0)

    p.add_argument(
        "--gmsh-order",
        type=int,
        default=1,
        choices=(1, 2),
    )

    p.add_argument(
        "--gmsh-verbosity",
        type=int,
        default=2,
    )

    # Electrostatics
    p.add_argument("--eps-air", type=float, default=1.0)
    p.add_argument("--eps-sample", type=float, default=11.7)

    p.add_argument("--tip-voltage", type=float, default=1.0)
    p.add_argument("--bottom-voltage", type=float, default=0.0)
    p.add_argument("--outer-voltage", type=float, default=0.0)

    p.add_argument(
        "--outer-neumann",
        action="store_true",
        help=(
            "Leave OUTER unpinned, giving the natural "
            "zero-flux boundary condition."
        ),
    )

    p.add_argument("--degree", type=int, default=1)

    p.add_argument(
        "--mesh-only",
        action="store_true",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/afm_tip_final"),
    )
>>>>>>> 7aa12b7edce39372da0bae7cbab48b3ea8409797

    return p.parse_args()


<<<<<<< HEAD
def check(a):
    values = [
        a.lx, a.ly, a.sample_depth, a.air_height,
        a.gap, a.tip_radius, a.cone_height, a.shank_radius,
        a.shaft_radius, a.shaft_height,
        a.h_fine, a.h_coarse, a.degree
    ]
    if any(v <= 0 for v in values):
        raise ValueError("All lengths, mesh sizes, and degree must be positive.")

    if a.h_fine > a.h_coarse:
        raise ValueError("Require h_fine <= h_coarse.")

    if not (0 <= a.refine_dist_min < a.refine_dist_max):
        raise ValueError("Require 0 <= refine_dist_min < refine_dist_max.")

    if a.shank_radius < a.tip_radius:
        raise ValueError("Require shank_radius >= tip_radius.")

    if a.shaft_radius < a.shank_radius:
        raise ValueError("Require shaft_radius >= shank_radius.")

    if a.shaft_radius >= 0.45 * min(a.lx, a.ly):
        raise ValueError("The shaft is too wide for the lateral domain.")

    zc = a.gap + a.tip_radius
    tip_top = max(
        a.gap + 2.0 * a.tip_radius,
        zc + a.cone_height + a.shaft_height
    )

    if tip_top >= a.air_height:
        raise ValueError(
            "The tip intersects the top of the air box. "
            "Increase --air-height or reduce tip dimensions."
        )


def physical(dim, entities, tag, name):
    if not entities:
        raise RuntimeError(f"Physical group '{name}' is empty.")
    gmsh.model.addPhysicalGroup(dim, entities, tag)
    gmsh.model.setPhysicalName(dim, tag, name)


def constant_plane(vmin, vmax, target, tol):
    return abs(vmin - target) < tol and abs(vmax - target) < tol


def classify_entities(a):
    air = []
    sample = []

    for _, tag in gmsh.model.getEntities(3):
        zc = gmsh.model.occ.getCenterOfMass(3, tag)[2]
        if zc < 0.0:
            sample.append(tag)
        else:
            air.append(tag)

    if len(air) != 1 or len(sample) != 1:
        raise RuntimeError(
            f"Expected exactly one air and one sample volume, got air={air}, sample={sample}."
        )

    surfaces = {"tip": [], "bottom": [], "outer": [], "interface": []}

    tol = 1.0e-7 * max(a.lx, a.ly, a.air_height, a.sample_depth)
    x0, x1 = -a.lx / 2.0, a.lx / 2.0
    y0, y1 = -a.ly / 2.0, a.ly / 2.0

    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)

        if constant_plane(zmin, zmax, -a.sample_depth, tol):
            surfaces["bottom"].append(tag)
        elif constant_plane(zmin, zmax, 0.0, tol):
            surfaces["interface"].append(tag)
        elif (
            constant_plane(xmin, xmax, x0, tol)
            or constant_plane(xmin, xmax, x1, tol)
            or constant_plane(ymin, ymax, y0, tol)
            or constant_plane(ymin, ymax, y1, tol)
            or constant_plane(zmin, zmax, a.air_height, tol)
        ):
            surfaces["outer"].append(tag)
        else:
            surfaces["tip"].append(tag)

    if any(not group for group in surfaces.values()):
        raise RuntimeError(f"Surface classification failed: {surfaces}")

    return air, sample, surfaces


def build_gmsh(a, msh_path):
    gmsh.model.add("afm_tip")
    occ = gmsh.model.occ

    # Sample volume: -sample_depth <= z <= 0
    sample = occ.addBox(
        -a.lx / 2.0, -a.ly / 2.0, -a.sample_depth,
        a.lx, a.ly, a.sample_depth
    )

    # Air volume: 0 <= z <= air_height
    air = occ.addBox(
        -a.lx / 2.0, -a.ly / 2.0, 0.0,
        a.lx, a.ly, a.air_height
    )

    # AFM tip pieces
    zc = a.gap + a.tip_radius

    sphere = occ.addSphere(0.0, 0.0, zc, a.tip_radius)

    cone = occ.addCone(
        0.0, 0.0, zc,
        0.0, 0.0, a.cone_height,
        a.tip_radius, a.shank_radius
    )

    shaft = occ.addCylinder(
        0.0, 0.0, zc + a.cone_height,
        0.0, 0.0, a.shaft_height,
        a.shaft_radius
    )

    tip_sc, _ = occ.fuse([(3, sphere)], [(3, cone)], removeObject=True, removeTool=True)
    tip, _ = occ.fuse(tip_sc, [(3, shaft)], removeObject=True, removeTool=True)

    # Remove the tip from the air to create the conducting cavity boundary
    air_cut, _ = occ.cut([(3, air)], tip, removeObject=True, removeTool=True)

    # Conforming air/sample interface
    occ.fragment([(3, sample)], air_cut, removeObject=True, removeTool=True)
    occ.synchronize()

    air_vols, sample_vols, surf = classify_entities(a)

    physical(3, air_vols, AIR, "air")
    physical(3, sample_vols, SAMPLE, "sample")
    physical(2, surf["tip"], TIP, "tip")
    physical(2, surf["bottom"], BOTTOM, "bottom")
    physical(2, surf["outer"], OUTER, "outer")
    physical(2, surf["interface"], INTERFACE, "sample_air_interface")

    # Mesh refinement around the complete tip surface
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", surf["tip"])
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 150)

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", a.h_fine)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", a.h_coarse)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", a.refine_dist_min)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", a.refine_dist_max)

    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.model.mesh.generate(3)

    if a.gmsh_order == 2:
        gmsh.model.mesh.setOrder(2)

    gmsh.write(str(msh_path))


def unpack(data):
    if hasattr(data, "mesh"):
        return data.mesh, data.cell_tags, data.facet_tags
    return data


def count_owned(mesh, tags, marker):
    entities = tags.find(marker)
    nlocal = np.count_nonzero(entities < mesh.topology.index_map(tags.dim).size_local)
    return mesh.comm.allreduce(int(nlocal), op=MPI.SUM)


def scalar(mesh, expression):
    local = fem.assemble_scalar(fem.form(expression))
    return mesh.comm.allreduce(local, op=MPI.SUM)


def tag_report(mesh, ct, ft):
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=ft)

    rows = [
        ("air cells", AIR, count_owned(mesh, ct, AIR), scalar(mesh, 1.0 * dx(AIR))),
        ("sample cells", SAMPLE, count_owned(mesh, ct, SAMPLE), scalar(mesh, 1.0 * dx(SAMPLE))),
        ("tip facets", TIP, count_owned(mesh, ft, TIP), scalar(mesh, 1.0 * ds(TIP))),
        ("bottom facets", BOTTOM, count_owned(mesh, ft, BOTTOM), scalar(mesh, 1.0 * ds(BOTTOM))),
        ("outer facets", OUTER, count_owned(mesh, ft, OUTER), scalar(mesh, 1.0 * ds(OUTER))),
        ("interface facets", INTERFACE, count_owned(mesh, ft, INTERFACE), scalar(mesh, 1.0 * dS(INTERFACE))),
    ]

    if mesh.comm.rank == 0:
        print("\nTag integrity")
        print("-" * 74)
        print(f"{'group':22s} {'tag':>5s} {'entities':>12s} {'measure':>22s}")
        for name, tag, count, measure in rows:
            print(f"{name:22s} {tag:5d} {count:12d} {measure:22.8e}")
        print("-" * 74)


def make_eps(mesh, ct, a):
    Q = fem.functionspace(mesh, ("DG", 0))
    eps = fem.Function(Q, name="relative_permittivity")
    eps.x.array[:] = a.eps_air

    tdim = mesh.topology.dim
    air_dofs = fem.locate_dofs_topological(Q, tdim, ct.find(AIR))
    sample_dofs = fem.locate_dofs_topological(Q, tdim, ct.find(SAMPLE))

    eps.x.array[air_dofs] = a.eps_air
    eps.x.array[sample_dofs] = a.eps_sample
    eps.x.scatter_forward()

    return eps


def bc(V, ft, marker, value):
    facets = ft.find(marker)

    local_count = len(facets)
    global_count = V.mesh.comm.allreduce(local_count, op=MPI.SUM)

    if global_count == 0:
        raise RuntimeError(f"Facet tag {marker} is empty on the entire mesh.")

    dofs = fem.locate_dofs_topological(V, ft.dim, facets, remote=True)
    return fem.dirichletbc(default_scalar_type(value), dofs, V)


def solve(mesh, ct, ft, a):
    V = fem.functionspace(mesh, ("Lagrange", a.degree))
    eps = make_eps(mesh, ct, a)

    bcs = [
        bc(V, ft, TIP, a.tip_voltage),
        bc(V, ft, BOTTOM, a.bottom_voltage),
    ]

    if not a.outer_neumann:
        bcs.append(bc(V, ft, OUTER, a.outer_voltage))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    zero = fem.Constant(mesh, default_scalar_type(0.0))
    lhs = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = zero * v * ufl.dx
=======
def validate(a):
    positive = {
        "lx": a.lx,
        "ly": a.ly,
        "sample_depth": a.sample_depth,
        "air_height": a.air_height,
        "gap": a.gap,
        "tip_radius": a.tip_radius,
        "shank_height": a.shank_height,
        "shank_top_radius": a.shank_top_radius,
        "shaft_radius": a.shaft_radius,
        "shaft_height": a.shaft_height,
        "h_fine": a.h_fine,
        "h_coarse": a.h_coarse,
        "degree": a.degree,
    }

    for name, value in positive.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be positive, got {value}."
            )

    if a.shaft_overlap < 0:
        raise ValueError(
            "--shaft-overlap must be >= 0."
        )

    if a.shaft_overlap >= a.shaft_height:
        raise ValueError(
            "--shaft-overlap must be smaller than --shaft-height."
        )

    if a.h_fine > a.h_coarse:
        raise ValueError(
            "Require h_fine <= h_coarse."
        )

    if not (
        0.0
        <= a.refine_dist_min
        < a.refine_dist_max
    ):
        raise ValueError(
            "Require 0 <= refine_dist_min < refine_dist_max."
        )

    if a.shank_top_radius < a.tip_radius:
        raise ValueError(
            "Require shank_top_radius >= tip_radius."
        )

    if a.shaft_radius < a.shank_top_radius:
        raise ValueError(
            "Require shaft_radius >= shank_top_radius."
        )

    if a.shaft_radius >= 0.5 * min(a.lx, a.ly):
        raise ValueError(
            "The shaft does not fit inside the lateral air box. "
            "Increase --lx/--ly or reduce --shaft-radius."
        )

    z_sphere_center = (
        a.gap
        + a.tip_radius
    )

    z_shank_top = (
        z_sphere_center
        + a.shank_height
    )

    z_tip_top = (
        z_shank_top
        + a.shaft_height
    )

    if z_tip_top >= a.air_height:
        raise ValueError(
            f"Tip reaches z={z_tip_top:g} nm but "
            f"air-height={a.air_height:g} nm. "
            "Increase --air-height."
        )


def add_physical_group(
    dim,
    entities,
    marker,
    name,
):
    if not entities:
        raise RuntimeError(
            f"Physical group '{name}' is empty."
        )

    gmsh.model.addPhysicalGroup(
        dim,
        entities,
        marker,
    )

    gmsh.model.setPhysicalName(
        dim,
        marker,
        name,
    )


def plane_match(
    vmin,
    vmax,
    target,
    tol,
):
    return (
        abs(vmin - target) < tol
        and abs(vmax - target) < tol
    )


def classify_entities(a):
    """
    Identify the air/sample volumes and boundary surfaces.
    """

    air_vols = []
    sample_vols = []

    for _, tag in gmsh.model.getEntities(3):

        zc = gmsh.model.occ.getCenterOfMass(
            3,
            tag,
        )[2]

        if zc < 0.0:
            sample_vols.append(tag)
        else:
            air_vols.append(tag)

    if (
        len(air_vols) != 1
        or len(sample_vols) != 1
    ):
        raise RuntimeError(
            "Expected exactly one air volume and one sample "
            "volume after Boolean operations. "
            f"Found air={air_vols}, sample={sample_vols}."
        )

    surfaces = {
        "tip": [],
        "bottom": [],
        "outer": [],
        "interface": [],
    }

    tol = (
        1.0e-7
        * max(
            a.lx,
            a.ly,
            a.sample_depth,
            a.air_height,
            1.0,
        )
    )

    xmin_dom = -0.5 * a.lx
    xmax_dom = +0.5 * a.lx

    ymin_dom = -0.5 * a.ly
    ymax_dom = +0.5 * a.ly

    for _, tag in gmsh.model.getEntities(2):

        (
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
        ) = gmsh.model.getBoundingBox(
            2,
            tag,
        )

        # Bottom of semiconductor
        if plane_match(
            zmin,
            zmax,
            -a.sample_depth,
            tol,
        ):

            surfaces["bottom"].append(
                tag
            )

        # Air/sample interface
        elif plane_match(
            zmin,
            zmax,
            0.0,
            tol,
        ):

            surfaces["interface"].append(
                tag
            )

        # Outer air/sample boundaries
        elif (
            plane_match(
                xmin,
                xmax,
                xmin_dom,
                tol,
            )
            or plane_match(
                xmin,
                xmax,
                xmax_dom,
                tol,
            )
            or plane_match(
                ymin,
                ymax,
                ymin_dom,
                tol,
            )
            or plane_match(
                ymin,
                ymax,
                ymax_dom,
                tol,
            )
            or plane_match(
                zmin,
                zmax,
                a.air_height,
                tol,
            )
        ):

            surfaces["outer"].append(
                tag
            )

        else:
            # Anything left is an AFM cavity surface.
            surfaces["tip"].append(
                tag
            )

    for name, tags in surfaces.items():

        if not tags:
            raise RuntimeError(
                f"Surface group '{name}' is empty."
            )

    return (
        air_vols,
        sample_vols,
        surfaces,
    )


def build_gmsh(
    a,
    msh_path,
):
    """
    Build and mesh the complete Gmsh geometry.

    Important:
    We do NOT fuse the AFM pieces.

    The sphere, cone, and shaft are sequentially
    subtracted from the air volume.
    """

    gmsh.model.add(
        "afm_tip_final"
    )

    occ = gmsh.model.occ

    # ==========================================================
    # SAMPLE
    #
    # -sample_depth <= z <= 0
    # ==========================================================

    sample = occ.addBox(
        -0.5 * a.lx,
        -0.5 * a.ly,
        -a.sample_depth,
        a.lx,
        a.ly,
        a.sample_depth,
    )

    # ==========================================================
    # AIR
    #
    # 0 <= z <= air_height
    # ==========================================================

    air = occ.addBox(
        -0.5 * a.lx,
        -0.5 * a.ly,
        0.0,
        a.lx,
        a.ly,
        a.air_height,
    )

    # ==========================================================
    # AFM GEOMETRY
    #
    # Defaults:
    #
    # sample surface        z =   0 nm
    # apex bottom           z =  10 nm
    # sphere center         z =  20 nm
    # shank top             z = 180 nm
    # shaft top             z = 280 nm
    #
    #                _________
    #               |         |
    #               | SHAFT   |
    #               | R=120   |
    #               |_________|
    #                 \     /
    #                  \   /
    #                   \ /
    #                   /\
    #                  (  )
    #                   \/
    #
    #               sample
    # -------------------------------- z=0
    #
    # ==========================================================

    z_sphere_center = (
        a.gap
        + a.tip_radius
    )

    z_shank_top = (
        z_sphere_center
        + a.shank_height
    )

    # ----------------------------------------------------------
    # 1. Spherical apex
    # ----------------------------------------------------------

    sphere = occ.addSphere(
        0.0,
        0.0,
        z_sphere_center,
        a.tip_radius,
    )

    # ----------------------------------------------------------
    # 2. Conical shank
    #
    # bottom radius = 10 nm
    # top radius    = 120 nm
    # height        = 160 nm
    #
    # The cone begins at the sphere center plane.
    # This naturally overlaps the upper part of the sphere.
    # ----------------------------------------------------------

    cone = occ.addCone(
        0.0,
        0.0,
        z_sphere_center,
        0.0,
        0.0,
        a.shank_height,
        a.tip_radius,
        a.shank_top_radius,
    )

    # ----------------------------------------------------------
    # 3. Cylindrical shaft
    #
    # Slight overlap with the cone is intentional.
    #
    # Nominal:
    # radius = 120 nm
    # height = 100 nm
    # ----------------------------------------------------------

    shaft_z0 = (
        z_shank_top
        - a.shaft_overlap
    )

    shaft_dz = (
        a.shaft_height
        + a.shaft_overlap
    )

    shaft = occ.addCylinder(
        0.0,
        0.0,
        shaft_z0,
        0.0,
        0.0,
        shaft_dz,
        a.shaft_radius,
    )

    occ.synchronize()

    # ==========================================================
    # CUT THE CONDUCTOR FROM THE AIR
    #
    # IMPORTANT:
    #
    # There is deliberately NO occ.fuse() here.
    #
    # We subtract:
    #
    # air - sphere
    #     - cone
    #     - shaft
    #
    # sequentially.
    #
    # This creates one conductor-shaped cavity without needing
    # OpenCASCADE to fuse the conductor solids first.
    # ==========================================================

    air_parts = [
        (3, air)
    ]

    conductor_parts = [
        (
            "spherical apex",
            sphere,
        ),
        (
            "conical shank",
            cone,
        ),
        (
            "cylindrical shaft",
            shaft,
        ),
    ]

    for (
        label,
        tool,
    ) in conductor_parts:

        air_parts, _ = occ.cut(
            air_parts,
            [(3, tool)],
            removeObject=True,
            removeTool=True,
        )

        if not air_parts:
            raise RuntimeError(
                "Air subtraction failed while removing "
                f"{label}."
            )

        occ.synchronize()

    # ==========================================================
    # AIR/SAMPLE CONFORMING INTERFACE
    # ==========================================================

    fragmented, _ = occ.fragment(
        [(3, sample)],
        air_parts,
        removeObject=True,
        removeTool=True,
    )

    if not fragmented:
        raise RuntimeError(
            "Air/sample fragmentation failed."
        )

    occ.synchronize()

    # ==========================================================
    # CLASSIFY ENTITIES
    # ==========================================================

    (
        air_vols,
        sample_vols,
        surfaces,
    ) = classify_entities(a)

    # ==========================================================
    # PHYSICAL VOLUME GROUPS
    # ==========================================================

    add_physical_group(
        3,
        air_vols,
        AIR,
        "air",
    )

    add_physical_group(
        3,
        sample_vols,
        SAMPLE,
        "sample",
    )

    # ==========================================================
    # PHYSICAL FACET GROUPS
    # ==========================================================

    add_physical_group(
        2,
        surfaces["tip"],
        TIP,
        "afm_tip",
    )

    add_physical_group(
        2,
        surfaces["bottom"],
        BOTTOM,
        "bottom",
    )

    add_physical_group(
        2,
        surfaces["outer"],
        OUTER,
        "outer",
    )

    add_physical_group(
        2,
        surfaces["interface"],
        INTERFACE,
        "sample_air_interface",
    )

    # ==========================================================
    # LOCAL MESH REFINEMENT
    #
    # Fine near AFM conductor surfaces.
    # Coarse farther away.
    # ==========================================================

    distance = (
        gmsh.model.mesh.field.add(
            "Distance"
        )
    )

    gmsh.model.mesh.field.setNumbers(
        distance,
        "SurfacesList",
        surfaces["tip"],
    )

    gmsh.model.mesh.field.setNumber(
        distance,
        "Sampling",
        150,
    )

    threshold = (
        gmsh.model.mesh.field.add(
            "Threshold"
        )
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "InField",
        distance,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "SizeMin",
        a.h_fine,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "SizeMax",
        a.h_coarse,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "DistMin",
        a.refine_dist_min,
    )

    gmsh.model.mesh.field.setNumber(
        threshold,
        "DistMax",
        a.refine_dist_max,
    )

    gmsh.model.mesh.field.setAsBackgroundMesh(
        threshold
    )

    # Do not let geometry points or curvature override our
    # background mesh sizing.
    gmsh.option.setNumber(
        "Mesh.MeshSizeExtendFromBoundary",
        0,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeFromPoints",
        0,
    )

    gmsh.option.setNumber(
        "Mesh.MeshSizeFromCurvature",
        0,
    )

    # 3D tetrahedral algorithm
    gmsh.option.setNumber(
        "Mesh.Algorithm3D",
        10,
    )

    gmsh.option.setNumber(
        "Mesh.Optimize",
        1,
    )

    gmsh.option.setNumber(
        "Mesh.OptimizeNetgen",
        1,
    )

    # ==========================================================
    # GENERATE MESH
    # ==========================================================

    gmsh.model.mesh.generate(
        3
    )

    if a.gmsh_order == 2:

        gmsh.model.mesh.setOrder(
            2
        )

    gmsh.write(
        str(msh_path)
    )


def count_owned(
    mesh,
    tags,
    marker,
):
    entities = tags.find(
        marker
    )

    index_map = (
        mesh.topology.index_map(
            tags.dim
        )
    )

    local_count = np.count_nonzero(
        entities
        < index_map.size_local
    )

    return mesh.comm.allreduce(
        int(local_count),
        op=MPI.SUM,
    )


def global_scalar(
    mesh,
    expression,
):
    local = fem.assemble_scalar(
        fem.form(
            expression
        )
    )

    return mesh.comm.allreduce(
        local,
        op=MPI.SUM,
    )


def tag_report(
    mesh,
    ct,
    ft,
):
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_data=ct,
    )

    ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=ft,
    )

    dS = ufl.Measure(
        "dS",
        domain=mesh,
        subdomain_data=ft,
    )

    rows = [
        (
            "air cells",
            AIR,
            count_owned(
                mesh,
                ct,
                AIR,
            ),
            global_scalar(
                mesh,
                1.0 * dx(AIR),
            ),
        ),
        (
            "sample cells",
            SAMPLE,
            count_owned(
                mesh,
                ct,
                SAMPLE,
            ),
            global_scalar(
                mesh,
                1.0 * dx(SAMPLE),
            ),
        ),
        (
            "tip facets",
            TIP,
            count_owned(
                mesh,
                ft,
                TIP,
            ),
            global_scalar(
                mesh,
                1.0 * ds(TIP),
            ),
        ),
        (
            "bottom facets",
            BOTTOM,
            count_owned(
                mesh,
                ft,
                BOTTOM,
            ),
            global_scalar(
                mesh,
                1.0 * ds(BOTTOM),
            ),
        ),
        (
            "outer facets",
            OUTER,
            count_owned(
                mesh,
                ft,
                OUTER,
            ),
            global_scalar(
                mesh,
                1.0 * ds(OUTER),
            ),
        ),
        (
            "interface facets",
            INTERFACE,
            count_owned(
                mesh,
                ft,
                INTERFACE,
            ),
            global_scalar(
                mesh,
                1.0 * dS(INTERFACE),
            ),
        ),
    ]

    if mesh.comm.rank == 0:

        print(
            "\nTag integrity"
        )

        print(
            "-" * 82
        )

        print(
            f"{'group':24s}"
            f"{'tag':>8s}"
            f"{'entities':>14s}"
            f"{'measure [nm^d]':>26s}"
        )

        print(
            "-" * 82
        )

        for (
            name,
            marker,
            count,
            measure,
        ) in rows:

            print(
                f"{name:24s}"
                f"{marker:8d}"
                f"{count:14d}"
                f"{measure:26.8e}"
            )

        print(
            "-" * 82
        )


def make_material_fields(
    mesh,
    ct,
    a,
):
    Q = fem.functionspace(
        mesh,
        (
            "DG",
            0,
        ),
    )

    eps = fem.Function(
        Q,
        name="relative_permittivity",
    )

    material_id = fem.Function(
        Q,
        name="material_id",
    )

    eps.x.array[:] = (
        a.eps_air
    )

    material_id.x.array[:] = (
        AIR
    )

    tdim = (
        mesh.topology.dim
    )

    air_dofs = (
        fem.locate_dofs_topological(
            Q,
            tdim,
            ct.find(AIR),
        )
    )

    sample_dofs = (
        fem.locate_dofs_topological(
            Q,
            tdim,
            ct.find(SAMPLE),
        )
    )

    eps.x.array[
        air_dofs
    ] = a.eps_air

    eps.x.array[
        sample_dofs
    ] = a.eps_sample

    material_id.x.array[
        air_dofs
    ] = AIR

    material_id.x.array[
        sample_dofs
    ] = SAMPLE

    eps.x.scatter_forward()

    material_id.x.scatter_forward()

    return (
        eps,
        material_id,
    )


def make_bc(
    V,
    ft,
    marker,
    value,
):
    facets = ft.find(
        marker
    )

    local_count = len(
        facets
    )

    global_count = (
        V.mesh.comm.allreduce(
            local_count,
            op=MPI.SUM,
        )
    )

    if global_count == 0:

        raise RuntimeError(
            f"Facet tag {marker} "
            "is empty on the entire mesh."
        )

    dofs = (
        fem.locate_dofs_topological(
            V,
            ft.dim,
            facets,
            remote=True,
        )
    )

    return fem.dirichletbc(
        default_scalar_type(
            value
        ),
        dofs,
        V,
    )


def solve(
    mesh,
    ct,
    ft,
    a,
):
    V = fem.functionspace(
        mesh,
        (
            "Lagrange",
            a.degree,
        ),
    )

    (
        eps,
        material_id,
    ) = make_material_fields(
        mesh,
        ct,
        a,
    )

    bcs = [
        make_bc(
            V,
            ft,
            TIP,
            a.tip_voltage,
        ),
        make_bc(
            V,
            ft,
            BOTTOM,
            a.bottom_voltage,
        ),
    ]

    if not a.outer_neumann:

        bcs.append(
            make_bc(
                V,
                ft,
                OUTER,
                a.outer_voltage,
            )
        )

    u = ufl.TrialFunction(
        V
    )

    v = ufl.TestFunction(
        V
    )

    zero = fem.Constant(
        mesh,
        default_scalar_type(
            0.0
        ),
    )

    lhs = (
        ufl.inner(
            eps
            * ufl.grad(u),
            ufl.grad(v),
        )
        * ufl.dx
    )

    rhs = (
        zero
        * v
        * ufl.dx
    )
>>>>>>> 7aa12b7edce39372da0bae7cbab48b3ea8409797

    problem = LinearProblem(
        lhs,
        rhs,
        bcs=bcs,
<<<<<<< HEAD
        petsc_options_prefix="afm_tip_",
=======
        petsc_options_prefix=(
            "afm_tip_"
        ),
>>>>>>> 7aa12b7edce39372da0bae7cbab48b3ea8409797
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1.0e-10,
            "ksp_atol": 1.0e-12,
            "ksp_max_it": 2000,
            "ksp_error_if_not_converged": True,
        },
    )

    phi = problem.solve()
<<<<<<< HEAD
    phi.name = "phi_V"
    phi.x.scatter_forward()

    values = phi.x.array.real
    pmin = mesh.comm.allreduce(values.min() if len(values) else np.inf, op=MPI.MIN)
    pmax = mesh.comm.allreduce(values.max() if len(values) else -np.inf, op=MPI.MAX)

    if mesh.comm.rank == 0:
        print("\nSolve results")
        print(f"phi min/max [V]   : {pmin:.12e}  {pmax:.12e}")
        print(f"KSP iterations    : {problem.solver.getIterationNumber()}")
        print(f"KSP reason        : {problem.solver.getConvergedReason()}")
        print(f"Global phi dofs   : {V.dofmap.index_map.size_global * V.dofmap.index_map_bs:,d}")

    return phi, eps


def metadata(a):
    return {
        "units": {"length": "nm", "potential": "V"},
        "geometry": {
            "lx": a.lx,
            "ly": a.ly,
            "sample_depth": a.sample_depth,
            "air_height": a.air_height,
            "gap": a.gap,
            "tip_radius": a.tip_radius,
            "cone_height": a.cone_height,
            "shank_radius": a.shank_radius,
            "shaft_radius": a.shaft_radius,
            "shaft_height": a.shaft_height,
        },
        "mesh": {
            "h_fine": a.h_fine,
            "h_coarse": a.h_coarse,
            "refine_dist_min": a.refine_dist_min,
            "refine_dist_max": a.refine_dist_max,
            "gmsh_order": a.gmsh_order,
            "degree": a.degree,
        },
        "materials": {
            "eps_air": a.eps_air,
            "eps_sample": a.eps_sample,
        },
        "voltages": {
            "tip": a.tip_voltage,
            "bottom": a.bottom_voltage,
            "outer": None if a.outer_neumann else a.outer_voltage,
        },
        "tags": {
            "volumes": {"air": AIR, "sample": SAMPLE},
            "facets": {"tip": TIP, "bottom": BOTTOM, "outer": OUTER, "interface": INTERFACE},
=======

    phi.name = (
        "phi_V"
    )

    phi.x.scatter_forward()

    values = (
        phi.x.array.real
    )

    local_min = (
        values.min()
        if len(values)
        else np.inf
    )

    local_max = (
        values.max()
        if len(values)
        else -np.inf
    )

    phi_min = (
        mesh.comm.allreduce(
            local_min,
            op=MPI.MIN,
        )
    )

    phi_max = (
        mesh.comm.allreduce(
            local_max,
            op=MPI.MAX,
        )
    )

    ndofs = (
        V.dofmap.index_map.size_global
        * V.dofmap.index_map_bs
    )

    if mesh.comm.rank == 0:

        print(
            "\nSolve results"
        )

        print(
            "phi min/max [V] : "
            f"{phi_min:.12e}  "
            f"{phi_max:.12e}"
        )

        print(
            "KSP iterations   : "
            f"{problem.solver.getIterationNumber()}"
        )

        print(
            "KSP reason       : "
            f"{problem.solver.getConvergedReason()}"
        )

        print(
            "Global phi dofs  : "
            f"{ndofs:,d}"
        )

    return (
        phi,
        eps,
        material_id,
    )


def build_metadata(a):
    z_sphere_center = (
        a.gap
        + a.tip_radius
    )

    z_shank_top = (
        z_sphere_center
        + a.shank_height
    )

    z_tip_top = (
        z_shank_top
        + a.shaft_height
    )

    return {
        "units": {
            "length": "nm",
            "potential": "V",
        },

        "geometry": {
            "lx": a.lx,
            "ly": a.ly,
            "sample_depth": (
                a.sample_depth
            ),
            "air_height": (
                a.air_height
            ),
            "gap": a.gap,
            "tip_radius": (
                a.tip_radius
            ),
            "apex_bottom_z": (
                a.gap
            ),
            "sphere_center_z": (
                z_sphere_center
            ),
            "shank_height": (
                a.shank_height
            ),
            "shank_bottom_radius": (
                a.tip_radius
            ),
            "shank_top_radius": (
                a.shank_top_radius
            ),
            "shank_top_z": (
                z_shank_top
            ),
            "shaft_radius": (
                a.shaft_radius
            ),
            "shaft_height": (
                a.shaft_height
            ),
            "shaft_overlap": (
                a.shaft_overlap
            ),
            "tip_top_z": (
                z_tip_top
            ),
        },

        "mesh": {
            "h_fine": (
                a.h_fine
            ),
            "h_coarse": (
                a.h_coarse
            ),
            "refine_dist_min": (
                a.refine_dist_min
            ),
            "refine_dist_max": (
                a.refine_dist_max
            ),
            "gmsh_order": (
                a.gmsh_order
            ),
            "fe_degree": (
                a.degree
            ),
        },

        "materials": {
            "air": {
                "tag": AIR,
                "epsilon_r": (
                    a.eps_air
                ),
            },

            "sample": {
                "tag": SAMPLE,
                "epsilon_r": (
                    a.eps_sample
                ),
            },
        },

        "boundary_tags": {
            "tip": TIP,
            "bottom": BOTTOM,
            "outer": OUTER,
            "interface": INTERFACE,
        },

        "voltages": {
            "tip": (
                a.tip_voltage
            ),
            "bottom": (
                a.bottom_voltage
            ),
            "outer": (
                None
                if a.outer_neumann
                else a.outer_voltage
            ),
>>>>>>> 7aa12b7edce39372da0bae7cbab48b3ea8409797
        },
    }


<<<<<<< HEAD
def write_files(mesh, ct, ft, phi, eps, out):
    ct.name = "cell_tags"
    ft.name = "facet_tags"

    with io.XDMFFile(mesh.comm, out / "afm_tip_mesh.xdmf", "w") as f:
        f.write_mesh(mesh)
        f.write_meshtags(ct, mesh.geometry)
        f.write_meshtags(ft, mesh.geometry)

    if phi is not None:
        with io.XDMFFile(mesh.comm, out / "afm_tip_solution.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(phi)
            f.write_function(eps)


def main():
    a = arguments()
    check(a)

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        with open(a.output / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata(a), f, indent=2)

    comm.barrier()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(a.gmsh_verbosity > 0))
    gmsh.option.setNumber("General.Verbosity", a.gmsh_verbosity)

    msh_path = a.output / "afm_tip.msh"

    try:
        if comm.rank == 0:
            build_gmsh(a, msh_path)

        partitioner = dmesh.create_cell_partitioner(
            dmesh.GhostMode.shared_facet,
            2,
        )

        data = gmshio.model_to_mesh(
            gmsh.model,
            comm,
            0,
            gdim=3,
            partitioner=partitioner,
        )
        mesh, ct, ft = unpack(data)

    finally:
        gmsh.finalize()

    if ct is None or ft is None:
        raise RuntimeError("Physical tags did not survive Gmsh-to-DOLFINx conversion.")

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    if comm.rank == 0:
        ncells = mesh.topology.index_map(mesh.topology.dim).size_global
        nnodes = mesh.geometry.index_map().size_global

        print("\nAFM-tip geometry")
        print(f"box [nm]              : {a.lx:g} x {a.ly:g} x ({a.sample_depth:g} sample + {a.air_height:g} air)")
        print(f"gap / tip-radius [nm] : {a.gap:g} / {a.tip_radius:g}")
        print(f"cone height [nm]      : {a.cone_height:g}")
        print(f"shank radius [nm]     : {a.shank_radius:g}")
        print(f"shaft radius [nm]     : {a.shaft_radius:g}")
        print(f"shaft height [nm]     : {a.shaft_height:g}")
        print(f"h fine/coarse [nm]    : {a.h_fine:g} / {a.h_coarse:g}")
        print(f"cells/nodes           : {ncells:,d} / {nnodes:,d}")
        print(f"output                : {a.output}")

    tag_report(mesh, ct, ft)

    phi = None
    eps = None

    if not a.mesh_only:
        phi, eps = solve(mesh, ct, ft, a)

    write_files(mesh, ct, ft, phi, eps, a.output)

    if comm.rank == 0:
        print("\nWrote:")
        print(f"  {msh_path}")
        print(f"  {a.output / 'afm_tip_mesh.xdmf'}")
        if phi is not None:
            print(f"  {a.output / 'afm_tip_solution.xdmf'}")
        print(f"  {a.output / 'run_metadata.json'}")
=======
def write_outputs(
    mesh,
    ct,
    ft,
    phi,
    eps,
    material_id,
    output,
):
    ct.name = (
        "cell_tags"
    )

    ft.name = (
        "facet_tags"
    )

    mesh_path = (
        output
        / "afm_tip_mesh.xdmf"
    )

    with io.XDMFFile(
        mesh.comm,
        mesh_path,
        "w",
    ) as xdmf:

        xdmf.write_mesh(
            mesh
        )

        xdmf.write_meshtags(
            ct,
            mesh.geometry,
        )

        xdmf.write_meshtags(
            ft,
            mesh.geometry,
        )

    if phi is not None:

        sol_path = (
            output
            / "afm_tip_solution.xdmf"
        )

        with io.XDMFFile(
            mesh.comm,
            sol_path,
            "w",
        ) as xdmf:

            xdmf.write_mesh(
                mesh
            )

            xdmf.write_function(
                phi
            )

            xdmf.write_function(
                eps
            )

            xdmf.write_function(
                material_id
            )


def main():
    a = parse_args()

    validate(
        a
    )

    comm = (
        MPI.COMM_WORLD
    )

    # ==========================================================
    # OUTPUT DIRECTORY AND RUN METADATA
    # ==========================================================

    if comm.rank == 0:

        a.output.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            a.output
            / "run_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                build_metadata(a),
                f,
                indent=2,
            )

    comm.barrier()

    # ==========================================================
    # GMSH
    # ==========================================================

    gmsh.initialize()

    gmsh.option.setNumber(
        "General.Terminal",
        int(
            a.gmsh_verbosity
            > 0
        ),
    )

    gmsh.option.setNumber(
        "General.Verbosity",
        a.gmsh_verbosity,
    )

    msh_path = (
        a.output
        / "afm_tip.msh"
    )

    try:

        if comm.rank == 0:

            build_gmsh(
                a,
                msh_path,
            )

        # Shared facet ghosting is required because
        # tag_report evaluates dS(INTERFACE) in parallel.
        partitioner = (
            dmesh.create_cell_partitioner(
                dmesh.GhostMode.shared_facet,
                2,
            )
        )

        data = (
            gmshio.model_to_mesh(
                gmsh.model,
                comm,
                0,
                gdim=3,
                partitioner=partitioner,
            )
        )

        mesh = data.mesh
        ct = data.cell_tags
        ft = data.facet_tags

    finally:

        gmsh.finalize()

    if (
        ct is None
        or ft is None
    ):

        raise RuntimeError(
            "Physical tags did not survive "
            "Gmsh-to-DOLFINx conversion."
        )

    # ==========================================================
    # MESH CONNECTIVITY
    # ==========================================================

    tdim = (
        mesh.topology.dim
    )

    fdim = (
        tdim - 1
    )

    mesh.topology.create_connectivity(
        fdim,
        tdim,
    )

    mesh.topology.create_connectivity(
        tdim,
        fdim,
    )

    # ==========================================================
    # GEOMETRY REPORT
    # ==========================================================

    if comm.rank == 0:

        ncells = (
            mesh.topology
            .index_map(tdim)
            .size_global
        )

        nnodes = (
            mesh.geometry
            .index_map()
            .size_global
        )

        z_sphere_center = (
            a.gap
            + a.tip_radius
        )

        z_shank_top = (
            z_sphere_center
            + a.shank_height
        )

        z_tip_top = (
            z_shank_top
            + a.shaft_height
        )

        print(
            "\nAFM-tip geometry"
        )

        print(
            "-" * 70
        )

        print(
            "domain x/y [nm]       : "
            f"{a.lx:g} x {a.ly:g}"
        )

        print(
            "sample depth [nm]     : "
            f"{a.sample_depth:g}"
        )

        print(
            "air height [nm]       : "
            f"{a.air_height:g}"
        )

        print(
            "gap [nm]              : "
            f"{a.gap:g}"
        )

        print(
            "tip radius [nm]       : "
            f"{a.tip_radius:g}"
        )

        print(
            "sphere center z [nm]  : "
            f"{z_sphere_center:g}"
        )

        print(
            "shank height [nm]     : "
            f"{a.shank_height:g}"
        )

        print(
            "shank top radius [nm] : "
            f"{a.shank_top_radius:g}"
        )

        print(
            "shank top z [nm]      : "
            f"{z_shank_top:g}"
        )

        print(
            "shaft radius [nm]     : "
            f"{a.shaft_radius:g}"
        )

        print(
            "shaft height [nm]     : "
            f"{a.shaft_height:g}"
        )

        print(
            "tip top z [nm]        : "
            f"{z_tip_top:g}"
        )

        print(
            "h fine/coarse [nm]    : "
            f"{a.h_fine:g} / "
            f"{a.h_coarse:g}"
        )

        print(
            "cells/nodes           : "
            f"{ncells:,d} / "
            f"{nnodes:,d}"
        )

        print(
            "output                : "
            f"{a.output}"
        )

        print(
            "-" * 70
        )

    # ==========================================================
    # TAG VERIFICATION
    # ==========================================================

    tag_report(
        mesh,
        ct,
        ft,
    )

    # ==========================================================
    # SOLVE
    # ==========================================================

    phi = None
    eps = None
    material_id = None

    if not a.mesh_only:

        (
            phi,
            eps,
            material_id,
        ) = solve(
            mesh,
            ct,
            ft,
            a,
        )

    # ==========================================================
    # OUTPUT
    # ==========================================================

    write_outputs(
        mesh,
        ct,
        ft,
        phi,
        eps,
        material_id,
        a.output,
    )

    if comm.rank == 0:

        print(
            "\nWrote:"
        )

        print(
            f"  {msh_path}"
        )

        print(
            "  "
            f"{a.output / 'afm_tip_mesh.xdmf'}"
        )

        print(
            "  "
            f"{a.output / 'afm_tip_mesh.h5'}"
        )

        if phi is not None:

            print(
                "  "
                f"{a.output / 'afm_tip_solution.xdmf'}"
            )

            print(
                "  "
                f"{a.output / 'afm_tip_solution.h5'}"
            )

        print(
            "  "
            f"{a.output / 'run_metadata.json'}"
        )
>>>>>>> 7aa12b7edce39372da0bae7cbab48b3ea8409797


if __name__ == "__main__":
    main()
