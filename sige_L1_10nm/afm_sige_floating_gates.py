#!/usr/bin/env python3
"""
3D AFM-tip + layered Si/SiGe electrostatics with two floating embedded gates.

Coordinates:
  z = 0 nm : top semiconductor surface
  z < 0    : air
  z > 0    : semiconductor, downward

Layer stack:
   0-2 nm       Si
   2-30 nm      SiGe
  30-40 nm      Si
  40-43 nm      SiGe
  43-53 nm      Si
  53-2053 nm    SiGe buffer

AFM:
  spherical apex R=10 nm, gap=10 nm
  conical shank h=160 nm, R: 10 -> 120 nm
  cylindrical shaft R=120 nm, h=100 nm

Floating metal gates:
  Gate 1 centered at z=35 nm
  Gate 2 centered at z=48 nm
  cylindrical disks, R=10 nm, default thickness=2 nm

The metal interiors are excluded from the dielectric mesh. Each gate is an
equipotential conductor. Its voltage is solved from Q=0 using superposition
and a 2x2 capacitance matrix.

Default tip sweep:
  Vtip = 0, +1, -1 V

Back gate:
  z = 2053 nm, Vback = -4.4 V

PDE:
  div(eps_r grad(phi)) = 0
"""

import argparse
import csv
import json
from pathlib import Path

import gmsh
import numpy as np
import ufl
from mpi4py import MPI

from dolfinx import default_scalar_type, fem, io, mesh as dmesh, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio


EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19

# Cell tags
AIR = 1
SI_TOP = 2
SIGE_TOP = 3
SI_WELL1 = 4
SIGE_MID = 5
SI_WELL2 = 6
SIGE_BUFFER = 7

# Facet tags
AFM_TIP = 11
GATE1 = 21
GATE2 = 22
BACK_GATE = 30
OUTER = 31
IFACE_Z0 = 40
IFACE_Z2 = 41
IFACE_Z30 = 42
IFACE_Z40 = 43
IFACE_Z43 = 44
IFACE_Z53 = 45


def parse_args():
    p = argparse.ArgumentParser(
        description="AFM + layered Si/SiGe device with two floating gates"
    )

    # Domain
    p.add_argument("--lx", type=float, default=300.0)
    p.add_argument("--ly", type=float, default=300.0)
    p.add_argument("--air-height", type=float, default=320.0)
    p.add_argument("--buffer-thickness", type=float, default=2000.0)

    # AFM
    p.add_argument("--gap", type=float, default=10.0)
    p.add_argument("--tip-radius", type=float, default=10.0)
    p.add_argument("--shank-height", type=float, default=160.0)
    p.add_argument("--shank-top-radius", type=float, default=120.0)
    p.add_argument("--shaft-radius", type=float, default=120.0)
    p.add_argument("--shaft-height", type=float, default=100.0)
    p.add_argument("--shaft-overlap", type=float, default=0.5)

    # Floating gates
    p.add_argument("--gate-radius", type=float, default=10.0)
    p.add_argument("--gate-thickness", type=float, default=2.0)
    p.add_argument("--gate1-z", type=float, default=35.0)
    p.add_argument("--gate2-z", type=float, default=48.0)
    p.add_argument("--gate-x", type=float, default=0.0)
    p.add_argument("--gate-y", type=float, default=0.0)

    # Materials
    p.add_argument("--eps-air", type=float, default=1.0)
    p.add_argument("--eps-si", type=float, default=11.7)
    p.add_argument("--eps-sige", type=float, default=12.0)

    # Voltages
    p.add_argument("--back-voltage", type=float, default=-4.4)
    p.add_argument("--outer-voltage", type=float, default=0.0)
    p.add_argument("--outer-neumann", action="store_true")
    p.add_argument(
        "--tip-voltages",
        type=float,
        nargs="+",
        default=[0.0, 1.0, -1.0],
    )

    # Mesh
    p.add_argument("--h-fine", type=float, default=5.0)
    p.add_argument("--h-coarse", type=float, default=20.0)
    p.add_argument("--refine-dist-min", type=float, default=10.0)
    p.add_argument("--refine-dist-max", type=float, default=80.0)
    p.add_argument("--gmsh-order", type=int, default=1, choices=(1, 2))
    p.add_argument("--gmsh-verbosity", type=int, default=2)
    p.add_argument("--degree", type=int, default=1)

    # Probe
    p.add_argument("--probe-z", type=float, default=35.0)
    p.add_argument("--probe-y", type=float, default=0.0)
    p.add_argument("--probe-half-width", type=float, default=100.0)
    p.add_argument("--probe-points", type=int, default=201)

    p.add_argument("--mesh-only", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/afm_layered_floating"),
    )
    return p.parse_args()


def device_bottom(a):
    return 53.0 + a.buffer_thickness


def validate(a):
    positive = {
        "lx": a.lx, "ly": a.ly, "air_height": a.air_height,
        "buffer_thickness": a.buffer_thickness, "gap": a.gap,
        "tip_radius": a.tip_radius, "shank_height": a.shank_height,
        "shank_top_radius": a.shank_top_radius,
        "shaft_radius": a.shaft_radius, "shaft_height": a.shaft_height,
        "gate_radius": a.gate_radius, "gate_thickness": a.gate_thickness,
        "h_fine": a.h_fine, "h_coarse": a.h_coarse,
        "degree": a.degree, "probe_points": a.probe_points,
    }

    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    if a.h_fine > a.h_coarse:
        raise ValueError("Require h_fine <= h_coarse.")

    if not (0 <= a.refine_dist_min < a.refine_dist_max):
        raise ValueError("Require 0 <= refine_dist_min < refine_dist_max.")

    if a.shank_top_radius < a.tip_radius:
        raise ValueError("Require shank_top_radius >= tip_radius.")

    if a.shaft_radius < a.shank_top_radius:
        raise ValueError("Require shaft_radius >= shank_top_radius.")

    if a.shaft_radius >= 0.5 * min(a.lx, a.ly):
        raise ValueError("AFM shaft does not fit laterally.")

    if not (0 <= a.shaft_overlap < a.shaft_height):
        raise ValueError("Require 0 <= shaft_overlap < shaft_height.")

    zc = -a.gap - a.tip_radius
    z_top = zc - a.shank_height - a.shaft_height

    if z_top <= -a.air_height:
        raise ValueError(
            f"AFM reaches z={z_top:g} nm but air starts at "
            f"z={-a.air_height:g} nm. Increase --air-height."
        )

    ht = 0.5 * a.gate_thickness

    if not (30.0 < a.gate1_z - ht and a.gate1_z + ht < 40.0):
        raise ValueError(
            "Gate 1 must fit fully inside the 30-40 nm Si layer."
        )

    if not (43.0 < a.gate2_z - ht and a.gate2_z + ht < 53.0):
        raise ValueError(
            "Gate 2 must fit fully inside the 43-53 nm Si layer."
        )


def add_group(dim, entities, tag, name):
    entities = list(dict.fromkeys(int(e) for e in entities))

    if not entities:
        raise RuntimeError(f"Physical group '{name}' is empty.")

    gmsh.model.addPhysicalGroup(dim, entities, tag)
    gmsh.model.setPhysicalName(dim, tag, name)


def plane(vmin, vmax, target, tol):
    return (
        abs(vmin - target) <= tol
        and abs(vmax - target) <= tol
    )


def bbox_inside_gate(
    bbox,
    x0,
    y0,
    z0,
    r,
    t,
    tol,
):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    ht = 0.5 * t

    return (
        xmin >= x0 - r - tol
        and xmax <= x0 + r + tol
        and ymin >= y0 - r - tol
        and ymax <= y0 + r + tol
        and zmin >= z0 - ht - tol
        and zmax <= z0 + ht + tol
    )


def classify_volumes(a):
    bottom = device_bottom(a)

    groups = {
        AIR: [],
        SI_TOP: [],
        SIGE_TOP: [],
        SI_WELL1: [],
        SIGE_MID: [],
        SI_WELL2: [],
        SIGE_BUFFER: [],
    }

    for _, tag in gmsh.model.getEntities(3):
        zc = gmsh.model.occ.getCenterOfMass(3, tag)[2]

        if zc < 0:
            marker = AIR
        elif zc < 2:
            marker = SI_TOP
        elif zc < 30:
            marker = SIGE_TOP
        elif zc < 40:
            marker = SI_WELL1
        elif zc < 43:
            marker = SIGE_MID
        elif zc < 53:
            marker = SI_WELL2
        elif zc <= bottom + 1e-8:
            marker = SIGE_BUFFER
        else:
            raise RuntimeError(
                f"Unexpected volume {tag}, center z={zc}."
            )

        groups[marker].append(tag)

    for marker, tags in groups.items():
        if not tags:
            raise RuntimeError(
                f"Cell group {marker} is empty."
            )

    return groups


def classify_surfaces(a):
    bottom = device_bottom(a)

    tol = 1e-7 * max(
        a.lx,
        a.ly,
        a.air_height,
        bottom,
        1.0,
    )

    xmin_d = -0.5 * a.lx
    xmax_d = 0.5 * a.lx
    ymin_d = -0.5 * a.ly
    ymax_d = 0.5 * a.ly

    groups = {
        AFM_TIP: [],
        GATE1: [],
        GATE2: [],
        BACK_GATE: [],
        OUTER: [],
        IFACE_Z0: [],
        IFACE_Z2: [],
        IFACE_Z30: [],
        IFACE_Z40: [],
        IFACE_Z43: [],
        IFACE_Z53: [],
    }

    interface_planes = [
        (0.0, IFACE_Z0),
        (2.0, IFACE_Z2),
        (30.0, IFACE_Z30),
        (40.0, IFACE_Z40),
        (43.0, IFACE_Z43),
        (53.0, IFACE_Z53),
    ]

    unclassified = []

    for _, tag in gmsh.model.getEntities(2):
        bbox = gmsh.model.getBoundingBox(2, tag)

        (
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
        ) = bbox

        if plane(zmin, zmax, bottom, tol):
            groups[BACK_GATE].append(tag)
            continue

        if (
            plane(xmin, xmax, xmin_d, tol)
            or plane(xmin, xmax, xmax_d, tol)
            or plane(ymin, ymax, ymin_d, tol)
            or plane(ymin, ymax, ymax_d, tol)
            or plane(zmin, zmax, -a.air_height, tol)
        ):
            groups[OUTER].append(tag)
            continue

        matched = False

        for z0, marker in interface_planes:
            if plane(zmin, zmax, z0, tol):
                groups[marker].append(tag)
                matched = True
                break

        if matched:
            continue

        if bbox_inside_gate(
            bbox,
            a.gate_x,
            a.gate_y,
            a.gate1_z,
            a.gate_radius,
            a.gate_thickness,
            tol,
        ):
            groups[GATE1].append(tag)
            continue

        if bbox_inside_gate(
            bbox,
            a.gate_x,
            a.gate_y,
            a.gate2_z,
            a.gate_radius,
            a.gate_thickness,
            tol,
        ):
            groups[GATE2].append(tag)
            continue

        if zmax < -tol:
            groups[AFM_TIP].append(tag)
            continue

        unclassified.append(
            {
                "tag": int(tag),
                "bbox": list(
                    map(float, bbox)
                ),
            }
        )

    if unclassified:
        raise RuntimeError(
            "Unclassified Gmsh surfaces:\n"
            + json.dumps(
                unclassified,
                indent=2,
            )
        )

    for marker, tags in groups.items():
        if not tags:
            raise RuntimeError(
                f"Facet group {marker} is empty."
            )

    return groups


def cut_one(objects, tool, label):
    result, _ = gmsh.model.occ.cut(
        objects,
        [(3, tool)],
        removeObject=True,
        removeTool=True,
    )

    if not result:
        raise RuntimeError(
            f"Boolean subtraction failed: {label}."
        )

    gmsh.model.occ.synchronize()

    return result


def build_gmsh(a, msh_path):
    gmsh.model.add(
        "afm_layered_floating"
    )

    occ = gmsh.model.occ

    # ------------------------------------------------------------------
    # Air
    # ------------------------------------------------------------------

    air = occ.addBox(
        -a.lx / 2,
        -a.ly / 2,
        -a.air_height,
        a.lx,
        a.ly,
        a.air_height,
    )

    # ------------------------------------------------------------------
    # Semiconductor layers
    # ------------------------------------------------------------------

    si_top = occ.addBox(
        -a.lx / 2,
        -a.ly / 2,
        0,
        a.lx,
        a.ly,
        2,
    )

    sige_top = occ.addBox(
        -a.lx / 2,
        -a.ly / 2,
        2,
        a.lx,
        a.ly,
        28,
    )

    si_well1 = occ.addBox(
        -a.lx / 2,
        -a.ly / 2,
        30,
        a.lx,
        a.ly,
        10,
    )

    sige_mid = occ.addBox(
        -a.lx / 2,
        -a.ly / 2,
        40,
        a.lx,
        a.ly,
        3,
    )

    si_well2 = occ.addBox(
        -a.lx / 2,
        -a.ly / 2,
        43,
        a.lx,
        a.ly,
        10,
    )

    sige_buffer = occ.addBox(
        -a.lx / 2,
        -a.ly / 2,
        53,
        a.lx,
        a.ly,
        a.buffer_thickness,
    )

    # ------------------------------------------------------------------
    # AFM in negative z.
    #
    # Apex nearest sample is z = -gap.
    # ------------------------------------------------------------------

    zc = -a.gap - a.tip_radius

    z_shank_top = (
        zc
        - a.shank_height
    )

    sphere = occ.addSphere(
        0,
        0,
        zc,
        a.tip_radius,
    )

    cone = occ.addCone(
        0,
        0,
        zc,
        0,
        0,
        -a.shank_height,
        a.tip_radius,
        a.shank_top_radius,
    )

    shaft_z0 = (
        z_shank_top
        + a.shaft_overlap
    )

    shaft_dz = -(
        a.shaft_height
        + a.shaft_overlap
    )

    shaft = occ.addCylinder(
        0,
        0,
        shaft_z0,
        0,
        0,
        shaft_dz,
        a.shaft_radius,
    )

    occ.synchronize()

    # Sequential subtraction. No fragile conductor fuse.
    air_parts = [
        (3, air)
    ]

    air_parts = cut_one(
        air_parts,
        sphere,
        "AFM sphere",
    )

    air_parts = cut_one(
        air_parts,
        cone,
        "AFM cone",
    )

    air_parts = cut_one(
        air_parts,
        shaft,
        "AFM shaft",
    )

    # ------------------------------------------------------------------
    # Floating metal gate cavities
    # ------------------------------------------------------------------

    ht = 0.5 * a.gate_thickness

    g1 = occ.addCylinder(
        a.gate_x,
        a.gate_y,
        a.gate1_z - ht,
        0,
        0,
        a.gate_thickness,
        a.gate_radius,
    )

    g2 = occ.addCylinder(
        a.gate_x,
        a.gate_y,
        a.gate2_z - ht,
        0,
        0,
        a.gate_thickness,
        a.gate_radius,
    )

    occ.synchronize()

    well1_parts = cut_one(
        [(3, si_well1)],
        g1,
        "Gate 1",
    )

    well2_parts = cut_one(
        [(3, si_well2)],
        g2,
        "Gate 2",
    )

    # ------------------------------------------------------------------
    # Make dielectric interfaces conforming
    # ------------------------------------------------------------------

    all_volumes = (
        air_parts
        + [(3, si_top)]
        + [(3, sige_top)]
        + well1_parts
        + [(3, sige_mid)]
        + well2_parts
        + [(3, sige_buffer)]
    )

    fragmented, _ = occ.fragment(
        [all_volumes[0]],
        all_volumes[1:],
        removeObject=True,
        removeTool=True,
    )

    if not fragmented:
        raise RuntimeError(
            "Final dielectric fragmentation failed."
        )

    occ.synchronize()

    volume_groups = classify_volumes(a)
    surface_groups = classify_surfaces(a)

    volume_names = {
        AIR: "air",
        SI_TOP: "si_0_2nm",
        SIGE_TOP: "sige_2_30nm",
        SI_WELL1: "si_30_40nm",
        SIGE_MID: "sige_40_43nm",
        SI_WELL2: "si_43_53nm",
        SIGE_BUFFER: "sige_buffer",
    }

    for marker, tags in volume_groups.items():
        add_group(
            3,
            tags,
            marker,
            volume_names[marker],
        )

    surface_names = {
        AFM_TIP: "afm_tip",
        GATE1: "floating_gate_1",
        GATE2: "floating_gate_2",
        BACK_GATE: "back_gate",
        OUTER: "outer",
        IFACE_Z0: "interface_z0",
        IFACE_Z2: "interface_z2",
        IFACE_Z30: "interface_z30",
        IFACE_Z40: "interface_z40",
        IFACE_Z43: "interface_z43",
        IFACE_Z53: "interface_z53",
    }

    for marker, tags in surface_groups.items():
        add_group(
            2,
            tags,
            marker,
            surface_names[marker],
        )

    # ------------------------------------------------------------------
    # Mesh refinement near AFM and floating gates
    # ------------------------------------------------------------------

    refine_surfaces = (
        surface_groups[AFM_TIP]
        + surface_groups[GATE1]
        + surface_groups[GATE2]
    )

    distance = (
        gmsh.model.mesh.field.add(
            "Distance"
        )
    )

    gmsh.model.mesh.field.setNumbers(
        distance,
        "SurfacesList",
        refine_surfaces,
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

    imap = mesh.topology.index_map(
        tags.dim
    )

    local = np.count_nonzero(
        entities
        < imap.size_local
    )

    return mesh.comm.allreduce(
        int(local),
        op=MPI.SUM,
    )


def global_scalar(
    mesh,
    expr,
):
    local = fem.assemble_scalar(
        fem.form(
            expr
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

    rows = []

    for name, marker in [
        ("air", AIR),
        ("Si 0-2", SI_TOP),
        ("SiGe 2-30", SIGE_TOP),
        ("Si well1 30-40", SI_WELL1),
        ("SiGe 40-43", SIGE_MID),
        ("Si well2 43-53", SI_WELL2),
        ("SiGe buffer", SIGE_BUFFER),
    ]:
        rows.append(
            (
                name,
                marker,
                count_owned(
                    mesh,
                    ct,
                    marker,
                ),
                global_scalar(
                    mesh,
                    1.0
                    * dx(marker),
                ),
            )
        )

    for name, marker in [
        ("AFM tip", AFM_TIP),
        ("floating gate 1", GATE1),
        ("floating gate 2", GATE2),
        ("back gate", BACK_GATE),
        ("outer", OUTER),
    ]:
        rows.append(
            (
                name,
                marker,
                count_owned(
                    mesh,
                    ft,
                    marker,
                ),
                global_scalar(
                    mesh,
                    1.0
                    * ds(marker),
                ),
            )
        )

    for name, marker in [
        ("interface z=0", IFACE_Z0),
        ("interface z=2", IFACE_Z2),
        ("interface z=30", IFACE_Z30),
        ("interface z=40", IFACE_Z40),
        ("interface z=43", IFACE_Z43),
        ("interface z=53", IFACE_Z53),
    ]:
        rows.append(
            (
                name,
                marker,
                count_owned(
                    mesh,
                    ft,
                    marker,
                ),
                global_scalar(
                    mesh,
                    1.0
                    * dS(marker),
                ),
            )
        )

    if mesh.comm.rank == 0:
        print(
            "\nTag integrity"
        )

        print(
            "-" * 86
        )

        print(
            f"{'group':24s}"
            f"{'tag':>8s}"
            f"{'entities':>14s}"
            f"{'measure':>24s}"
        )

        print(
            "-" * 86
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
                f"{measure:24.8e}"
            )

        print(
            "-" * 86
        )


def make_material_fields(
    mesh,
    ct,
    a,
):
    Q = fem.functionspace(
        mesh,
        ("DG", 0),
    )

    eps = fem.Function(
        Q,
        name="relative_permittivity",
    )

    material_id = fem.Function(
        Q,
        name="material_id",
    )

    eps_map = {
        AIR: a.eps_air,
        SI_TOP: a.eps_si,
        SIGE_TOP: a.eps_sige,
        SI_WELL1: a.eps_si,
        SIGE_MID: a.eps_sige,
        SI_WELL2: a.eps_si,
        SIGE_BUFFER: a.eps_sige,
    }

    eps.x.array[:] = (
        a.eps_air
    )

    material_id.x.array[:] = (
        AIR
    )

    tdim = mesh.topology.dim

    for marker, eps_r in eps_map.items():
        dofs = fem.locate_dofs_topological(
            Q,
            tdim,
            ct.find(marker),
        )

        eps.x.array[dofs] = (
            eps_r
        )

        material_id.x.array[dofs] = (
            marker
        )

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

    global_count = (
        V.mesh.comm.allreduce(
            len(facets),
            op=MPI.SUM,
        )
    )

    if global_count == 0:
        raise RuntimeError(
            f"Facet tag {marker} is empty."
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


def solve_dirichlet(
    V,
    eps,
    ft,
    tip_voltage,
    gate1_voltage,
    gate2_voltage,
    back_voltage,
    outer_voltage,
    outer_neumann,
    prefix,
):
    bcs = [
        make_bc(
            V,
            ft,
            AFM_TIP,
            tip_voltage,
        ),
        make_bc(
            V,
            ft,
            GATE1,
            gate1_voltage,
        ),
        make_bc(
            V,
            ft,
            GATE2,
            gate2_voltage,
        ),
        make_bc(
            V,
            ft,
            BACK_GATE,
            back_voltage,
        ),
    ]

    if not outer_neumann:
        bcs.append(
            make_bc(
                V,
                ft,
                OUTER,
                outer_voltage,
            )
        )

    u = ufl.TrialFunction(
        V
    )

    v = ufl.TestFunction(
        V
    )

    zero = fem.Constant(
        V.mesh,
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

    problem = LinearProblem(
        lhs,
        rhs,
        bcs=bcs,
        petsc_options_prefix=prefix,
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "gamg",
            "ksp_gmres_restart": 100,
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
            "ksp_max_it": 3000,
            "ksp_error_if_not_converged": True,
        },
    )

    phi = problem.solve()

    phi.x.scatter_forward()

    return (
        phi,
        problem.solver.getIterationNumber(),
        problem.solver.getConvergedReason(),
    )


def conductor_flux(
    mesh,
    ft,
    eps,
    phi,
    marker,
):
    """
    flux = integral eps_r * grad(phi).n_domain dA.

    Q_conductor = eps0 * 1e-9 * flux [C]
    because coordinates are in nm.
    """

    ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=ft,
    )

    n = ufl.FacetNormal(
        mesh
    )

    return float(
        global_scalar(
            mesh,
            eps
            * ufl.dot(
                ufl.grad(phi),
                n,
            )
            * ds(marker),
        )
    )


def flux_to_charge_c(
    flux,
):
    return (
        EPS0
        * 1e-9
        * flux
    )


def compose_solution(
    V,
    base,
    psi1,
    psi2,
    vg1,
    vg2,
    name="phi_V",
):
    phi = fem.Function(
        V,
        name=name,
    )

    phi.x.array[:] = (
        base.x.array
        + vg1
        * psi1.x.array
        + vg2
        * psi2.x.array
    )

    phi.x.scatter_forward()

    return phi


def solve_floating_case(
    V,
    eps,
    ft,
    psi1,
    psi2,
    Cflux,
    tip_voltage,
    a,
    prefix,
):
    base, iterations, reason = (
        solve_dirichlet(
            V,
            eps,
            ft,
            tip_voltage=tip_voltage,
            gate1_voltage=0.0,
            gate2_voltage=0.0,
            back_voltage=a.back_voltage,
            outer_voltage=a.outer_voltage,
            outer_neumann=a.outer_neumann,
            prefix=prefix,
        )
    )

    q0 = np.array(
        [
            conductor_flux(
                V.mesh,
                ft,
                eps,
                base,
                GATE1,
            ),
            conductor_flux(
                V.mesh,
                ft,
                eps,
                base,
                GATE2,
            ),
        ]
    )

    vg1, vg2 = np.linalg.solve(
        Cflux,
        -q0,
    )

    phi = compose_solution(
        V,
        base,
        psi1,
        psi2,
        float(vg1),
        float(vg2),
    )

    residual = np.array(
        [
            conductor_flux(
                V.mesh,
                ft,
                eps,
                phi,
                GATE1,
            ),
            conductor_flux(
                V.mesh,
                ft,
                eps,
                phi,
                GATE2,
            ),
        ]
    )

    return {
        "phi": phi,
        "vg1": float(vg1),
        "vg2": float(vg2),
        "residual_flux": residual,
        "iterations": int(iterations),
        "reason": int(reason),
    }


def voltage_label(
    v,
):
    if abs(v) < 5e-13:
        return "0V"

    sign = (
        "p"
        if v > 0
        else "m"
    )

    magnitude = (
        f"{abs(v):g}"
        .replace(
            ".",
            "p",
        )
    )

    return (
        f"{sign}"
        f"{magnitude}"
        "V"
    )


def global_min_max(
    phi,
):
    values = (
        phi.x.array.real
    )

    lmin = (
        values.min()
        if len(values)
        else np.inf
    )

    lmax = (
        values.max()
        if len(values)
        else -np.inf
    )

    mesh = (
        phi.function_space.mesh
    )

    return (
        float(
            mesh.comm.allreduce(
                lmin,
                op=MPI.MIN,
            )
        ),
        float(
            mesh.comm.allreduce(
                lmax,
                op=MPI.MAX,
            )
        ),
    )


def evaluate_points_parallel(
    phi,
    points,
):
    mesh = (
        phi.function_space.mesh
    )

    tdim = (
        mesh.topology.dim
    )

    points = np.asarray(
        points,
        dtype=mesh.geometry.x.dtype,
    )

    tree = geometry.bb_tree(
        mesh,
        tdim,
    )

    candidates = (
        geometry.compute_collisions_points(
            tree,
            points,
        )
    )

    colliding = (
        geometry.compute_colliding_cells(
            mesh,
            candidates,
            points,
        )
    )

    local_sum = np.zeros(
        len(points)
    )

    local_count = np.zeros(
        len(points),
        dtype=np.int32,
    )

    n_owned = (
        mesh.topology
        .index_map(tdim)
        .size_local
    )

    eval_points = []
    eval_cells = []
    eval_ids = []

    for i in range(
        len(points)
    ):
        owned = [
            int(c)
            for c
            in colliding.links(i)
            if int(c)
            < n_owned
        ]

        if owned:
            eval_points.append(
                points[i]
            )

            eval_cells.append(
                owned[0]
            )

            eval_ids.append(
                i
            )

    if eval_points:
        p = np.asarray(
            eval_points,
            dtype=mesh.geometry.x.dtype,
        )

        c = np.asarray(
            eval_cells,
            dtype=np.int32,
        )

        vals = np.asarray(
            phi.eval(
                p,
                c,
            )
        ).reshape(
            len(eval_ids),
            -1,
        )[:, 0].real

        for i, val in zip(
            eval_ids,
            vals,
        ):
            local_sum[i] = (
                float(val)
            )

            local_count[i] = (
                1
            )

    total_sum = np.zeros_like(
        local_sum
    )

    total_count = np.zeros_like(
        local_count
    )

    mesh.comm.Allreduce(
        local_sum,
        total_sum,
        op=MPI.SUM,
    )

    mesh.comm.Allreduce(
        local_count,
        total_count,
        op=MPI.SUM,
    )

    out = np.full(
        len(points),
        np.nan,
    )

    found = (
        total_count > 0
    )

    out[found] = (
        total_sum[found]
        / total_count[found]
    )

    return out


def inside_gate(
    x,
    y,
    z,
    x0,
    y0,
    z0,
    r,
    t,
):
    return (
        (x - x0) ** 2
        + (y - y0) ** 2
        <= r ** 2
        + 1e-12
        and abs(
            z - z0
        )
        <= 0.5 * t
        + 1e-12
    )


def write_probe_csv(
    phi,
    a,
    path,
    vg1,
    vg2,
):
    xs = np.linspace(
        -a.probe_half_width,
        a.probe_half_width,
        a.probe_points,
    )

    points = np.column_stack(
        [
            xs,
            np.full_like(
                xs,
                a.probe_y,
            ),
            np.full_like(
                xs,
                a.probe_z,
            ),
        ]
    )

    values = (
        evaluate_points_parallel(
            phi,
            points,
        )
    )

    regions = np.full(
        len(xs),
        "dielectric",
        dtype=object,
    )

    for i, x in enumerate(xs):
        if inside_gate(
            x,
            a.probe_y,
            a.probe_z,
            a.gate_x,
            a.gate_y,
            a.gate1_z,
            a.gate_radius,
            a.gate_thickness,
        ):
            values[i] = vg1
            regions[i] = (
                "gate1_metal"
            )

        elif inside_gate(
            x,
            a.probe_y,
            a.probe_z,
            a.gate_x,
            a.gate_y,
            a.gate2_z,
            a.gate_radius,
            a.gate_thickness,
        ):
            values[i] = vg2
            regions[i] = (
                "gate2_metal"
            )

        elif np.isnan(
            values[i]
        ):
            regions[i] = (
                "outside_mesh"
            )

    if (
        phi.function_space
        .mesh.comm.rank
        == 0
    ):
        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.writer(
                f
            )

            w.writerow(
                [
                    "x_nm",
                    "y_nm",
                    "z_nm",
                    "phi_V",
                    "region",
                ]
            )

            for (
                x,
                val,
                region,
            ) in zip(
                xs,
                values,
                regions,
            ):
                w.writerow(
                    [
                        f"{x:.12g}",
                        f"{a.probe_y:.12g}",
                        f"{a.probe_z:.12g}",
                        (
                            f"{val:.16e}"
                            if np.isfinite(val)
                            else "nan"
                        ),
                        region,
                    ]
                )


def write_mesh(
    mesh,
    ct,
    ft,
    output,
):
    ct.name = (
        "cell_tags"
    )

    ft.name = (
        "facet_tags"
    )

    with io.XDMFFile(
        mesh.comm,
        output
        / "device_mesh.xdmf",
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


def write_solution(
    mesh,
    phi,
    eps,
    material_id,
    path,
):
    with io.XDMFFile(
        mesh.comm,
        path,
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


def make_metadata(
    a,
):
    zc = (
        -a.gap
        - a.tip_radius
    )

    z_shank_top = (
        zc
        - a.shank_height
    )

    return {
        "units": {
            "length": "nm",
            "potential": "V",
        },

        "coordinates": {
            "surface_z_nm": 0,
            "air": "z<0",
            "device": "z>0",
        },

        "domain": {
            "lx_nm": a.lx,
            "ly_nm": a.ly,
            "air_height_nm": (
                a.air_height
            ),
            "device_bottom_nm": (
                device_bottom(a)
            ),
        },

        "layers": [
            [
                "Si",
                0,
                2,
                a.eps_si,
                SI_TOP,
            ],
            [
                "SiGe",
                2,
                30,
                a.eps_sige,
                SIGE_TOP,
            ],
            [
                "Si",
                30,
                40,
                a.eps_si,
                SI_WELL1,
            ],
            [
                "SiGe",
                40,
                43,
                a.eps_sige,
                SIGE_MID,
            ],
            [
                "Si",
                43,
                53,
                a.eps_si,
                SI_WELL2,
            ],
            [
                "SiGe",
                53,
                device_bottom(a),
                a.eps_sige,
                SIGE_BUFFER,
            ],
        ],

        "afm": {
            "gap_nm": a.gap,
            "tip_radius_nm": (
                a.tip_radius
            ),
            "sphere_center_z_nm": (
                zc
            ),
            "shank_height_nm": (
                a.shank_height
            ),
            "shank_top_radius_nm": (
                a.shank_top_radius
            ),
            "shank_top_z_nm": (
                z_shank_top
            ),
            "shaft_radius_nm": (
                a.shaft_radius
            ),
            "shaft_height_nm": (
                a.shaft_height
            ),
            "tip_top_z_nm": (
                z_shank_top
                - a.shaft_height
            ),
        },

        "floating_gates": {
            "shape": (
                "cylindrical "
                "excluded metal cavities"
            ),
            "radius_nm": (
                a.gate_radius
            ),
            "thickness_nm": (
                a.gate_thickness
            ),
            "gate1_center_nm": [
                a.gate_x,
                a.gate_y,
                a.gate1_z,
            ],
            "gate2_center_nm": [
                a.gate_x,
                a.gate_y,
                a.gate2_z,
            ],
            "constraint": (
                "equipotential, "
                "zero net free charge"
            ),
        },

        "voltages": {
            "tip_sweep_V": (
                a.tip_voltages
            ),
            "back_gate_V": (
                a.back_voltage
            ),
            "outer_V": (
                None
                if a.outer_neumann
                else a.outer_voltage
            ),
        },

        "mesh": {
            "h_fine_nm": (
                a.h_fine
            ),
            "h_coarse_nm": (
                a.h_coarse
            ),
            "degree": (
                a.degree
            ),
        },

        "probe": {
            "z_nm": (
                a.probe_z
            ),
            "y_nm": (
                a.probe_y
            ),
            "half_width_nm": (
                a.probe_half_width
            ),
            "points": (
                a.probe_points
            ),
        },
    }


def main():
    a = parse_args()

    validate(
        a
    )

    comm = (
        MPI.COMM_WORLD
    )

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
                make_metadata(a),
                f,
                indent=2,
            )

    comm.barrier()

    msh_path = (
        a.output
        / "device.msh"
    )

    gmsh.initialize()

    gmsh.option.setNumber(
        "General.Terminal",
        int(
            a.gmsh_verbosity > 0
        ),
    )

    gmsh.option.setNumber(
        "General.Verbosity",
        a.gmsh_verbosity,
    )

    try:
        if comm.rank == 0:
            build_gmsh(
                a,
                msh_path,
            )

        partitioner = (
            dmesh.create_cell_partitioner(
                dmesh.GhostMode.shared_facet,
                2,
            )
        )

        data = gmshio.model_to_mesh(
            gmsh.model,
            comm,
            0,
            gdim=3,
            partitioner=partitioner,
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

        print(
            "\nLayered AFM + "
            "floating-gate geometry"
        )

        print(
            "=" * 78
        )

        print(
            f"box x/y [nm]         : "
            f"{a.lx:g} x {a.ly:g}"
        )

        print(
            f"air                  : "
            f"{-a.air_height:g} "
            "<= z <= 0 nm"
        )

        print(
            f"device               : "
            f"0 <= z <= "
            f"{device_bottom(a):g} nm"
        )

        print(
            f"AFM gap/Rtip [nm]    : "
            f"{a.gap:g} / "
            f"{a.tip_radius:g}"
        )

        print(
            f"AFM shank [nm]       : "
            f"h={a.shank_height:g}, "
            f"Rtop="
            f"{a.shank_top_radius:g}"
        )

        print(
            f"AFM shaft [nm]       : "
            f"h={a.shaft_height:g}, "
            f"R={a.shaft_radius:g}"
        )

        print(
            f"Gate 1               : "
            f"R={a.gate_radius:g}, "
            f"t={a.gate_thickness:g}, "
            f"z={a.gate1_z:g}"
        )

        print(
            f"Gate 2               : "
            f"R={a.gate_radius:g}, "
            f"t={a.gate_thickness:g}, "
            f"z={a.gate2_z:g}"
        )

        print(
            f"back gate             : "
            f"z={device_bottom(a):g} nm, "
            f"V={a.back_voltage:g} V"
        )

        print(
            f"h fine/coarse [nm]    : "
            f"{a.h_fine:g} / "
            f"{a.h_coarse:g}"
        )

        print(
            f"cells/nodes           : "
            f"{ncells:,d} / "
            f"{nnodes:,d}"
        )

        print(
            f"output                : "
            f"{a.output}"
        )

        print(
            "=" * 78
        )

    tag_report(
        mesh,
        ct,
        ft,
    )

    write_mesh(
        mesh,
        ct,
        ft,
        a.output,
    )

    if a.mesh_only:
        if comm.rank == 0:
            print(
                "\nMesh-only run complete."
            )

        return

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

    # ----------------------------------------------------------
    # Floating-gate influence basis
    # ----------------------------------------------------------

    if comm.rank == 0:
        print(
            "\nBuilding floating-gate "
            "capacitance matrix..."
        )

    (
        psi1,
        it1,
        r1,
    ) = solve_dirichlet(
        V,
        eps,
        ft,
        tip_voltage=0,
        gate1_voltage=1,
        gate2_voltage=0,
        back_voltage=0,
        outer_voltage=0,
        outer_neumann=a.outer_neumann,
        prefix="gate_basis_1_",
    )

    psi1.name = (
        "gate1_influence"
    )

    (
        psi2,
        it2,
        r2,
    ) = solve_dirichlet(
        V,
        eps,
        ft,
        tip_voltage=0,
        gate1_voltage=0,
        gate2_voltage=1,
        back_voltage=0,
        outer_voltage=0,
        outer_neumann=a.outer_neumann,
        prefix="gate_basis_2_",
    )

    psi2.name = (
        "gate2_influence"
    )

    Cflux = np.array(
        [
            [
                conductor_flux(
                    mesh,
                    ft,
                    eps,
                    psi1,
                    GATE1,
                ),
                conductor_flux(
                    mesh,
                    ft,
                    eps,
                    psi2,
                    GATE1,
                ),
            ],
            [
                conductor_flux(
                    mesh,
                    ft,
                    eps,
                    psi1,
                    GATE2,
                ),
                conductor_flux(
                    mesh,
                    ft,
                    eps,
                    psi2,
                    GATE2,
                ),
            ],
        ]
    )

    cond = float(
        np.linalg.cond(
            Cflux
        )
    )

    if (
        not np.isfinite(cond)
        or cond > 1e12
    ):
        raise RuntimeError(
            "Floating-gate capacitance "
            "matrix badly conditioned: "
            f"{cond:.6e}"
        )

    # flux/V -> C/V -> aF
    C_aF = (
        EPS0
        * 1e9
        * Cflux
    )

    if comm.rank == 0:
        print(
            "\nFloating-gate "
            "capacitance matrix [aF]"
        )

        print(
            C_aF
        )

        print(
            f"condition number      : "
            f"{cond:.6e}"
        )

        print(
            f"basis KSP iterations  : "
            f"{it1}, {it2}"
        )

        print(
            f"basis KSP reasons     : "
            f"{r1}, {r2}"
        )

    results = {}
    table = []

    # ----------------------------------------------------------
    # Tip sweep
    # ----------------------------------------------------------

    for vt in a.tip_voltages:
        label = voltage_label(
            vt
        )

        if comm.rank == 0:
            print(
                "\n"
                + "-" * 78
            )

            print(
                f"Solving V_tip = "
                f"{vt:g} V"
            )

            print(
                "-" * 78
            )

        result = solve_floating_case(
            V,
            eps,
            ft,
            psi1,
            psi2,
            Cflux,
            tip_voltage=vt,
            a=a,
            prefix=f"tip_{label}_",
        )

        phi = result[
            "phi"
        ]

        phi.name = (
            "phi_V"
        )

        (
            pmin,
            pmax,
        ) = global_min_max(
            phi
        )

        q1 = flux_to_charge_c(
            result[
                "residual_flux"
            ][0]
        )

        q2 = flux_to_charge_c(
            result[
                "residual_flux"
            ][1]
        )

        if comm.rank == 0:
            print(
                f"floating V_G1 [V]     : "
                f"{result['vg1']:.12e}"
            )

            print(
                f"floating V_G2 [V]     : "
                f"{result['vg2']:.12e}"
            )

            print(
                f"Q_G1 residual [C]     : "
                f"{q1:.12e}"
            )

            print(
                f"Q_G2 residual [C]     : "
                f"{q2:.12e}"
            )

            print(
                f"Q_G1 residual [e]     : "
                f"{q1/E_CHARGE:.12e}"
            )

            print(
                f"Q_G2 residual [e]     : "
                f"{q2/E_CHARGE:.12e}"
            )

            print(
                f"phi min/max [V]       : "
                f"{pmin:.12e}  "
                f"{pmax:.12e}"
            )

            print(
                f"base KSP iterations   : "
                f"{result['iterations']}"
            )

            print(
                f"base KSP reason       : "
                f"{result['reason']}"
            )

        write_solution(
            mesh,
            phi,
            eps,
            material_id,
            (
                a.output
                / f"solution_tip_{label}.xdmf"
            ),
        )

        write_probe_csv(
            phi,
            a,
            (
                a.output
                / (
                    f"probe_z"
                    f"{a.probe_z:g}"
                    f"_tip_{label}.csv"
                )
            ),
            result[
                "vg1"
            ],
            result[
                "vg2"
            ],
        )

        results[
            float(vt)
        ] = result

        table.append(
            {
                "tip_voltage_V": (
                    float(vt)
                ),
                "gate1_voltage_V": (
                    result["vg1"]
                ),
                "gate2_voltage_V": (
                    result["vg2"]
                ),
                "gate1_charge_residual_C": (
                    q1
                ),
                "gate2_charge_residual_C": (
                    q2
                ),
                "phi_min_V": (
                    pmin
                ),
                "phi_max_V": (
                    pmax
                ),
                "ksp_iterations_base": (
                    result[
                        "iterations"
                    ]
                ),
                "ksp_reason_base": (
                    result[
                        "reason"
                    ]
                ),
            }
        )

    # ----------------------------------------------------------
    # Delta phi relative to Vtip = 0
    # ----------------------------------------------------------

    zero_key = next(
        (
            k
            for k in results
            if abs(k) < 5e-13
        ),
        None,
    )

    if zero_key is not None:
        zero = results[
            zero_key
        ]

        for (
            vt,
            result,
        ) in results.items():
            if abs(vt) < 5e-13:
                continue

            label = voltage_label(
                vt
            )

            delta = fem.Function(
                V,
                name="delta_phi_V",
            )

            delta.x.array[:] = (
                result[
                    "phi"
                ].x.array
                - zero[
                    "phi"
                ].x.array
            )

            delta.x.scatter_forward()

            write_solution(
                mesh,
                delta,
                eps,
                material_id,
                (
                    a.output
                    / (
                        f"delta_phi_tip_"
                        f"{label}"
                        "_minus_0V.xdmf"
                    )
                ),
            )

            write_probe_csv(
                delta,
                a,
                (
                    a.output
                    / (
                        f"delta_probe_z"
                        f"{a.probe_z:g}"
                        f"_tip_{label}"
                        "_minus_0V.csv"
                    )
                ),
                (
                    result[
                        "vg1"
                    ]
                    - zero[
                        "vg1"
                    ]
                ),
                (
                    result[
                        "vg2"
                    ]
                    - zero[
                        "vg2"
                    ]
                ),
            )

    # ----------------------------------------------------------
    # Summary CSV
    # ----------------------------------------------------------

    if comm.rank == 0:
        summary = (
            a.output
            / "floating_gate_results.csv"
        )

        fields = [
            "tip_voltage_V",
            "gate1_voltage_V",
            "gate2_voltage_V",
            "gate1_charge_residual_C",
            "gate2_charge_residual_C",
            "phi_min_V",
            "phi_max_V",
            "ksp_iterations_base",
            "ksp_reason_base",
        ]

        with open(
            summary,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            w.writeheader()

            w.writerows(
                table
            )

        np.savetxt(
            (
                a.output
                / (
                    "floating_gate_"
                    "capacitance_matrix_aF.txt"
                )
            ),
            C_aF,
            header=(
                "Rows: charge Gate1, Gate2. "
                "Columns: Gate1/2 influence. "
                "Units: aF."
            ),
        )

        print(
            "\nSummary"
        )

        print(
            "-" * 98
        )

        print(
            f"{'Vtip [V]':>10s}"
            f"{'VG1 [V]':>18s}"
            f"{'VG2 [V]':>18s}"
            f"{'Q1 resid [C]':>20s}"
            f"{'Q2 resid [C]':>20s}"
        )

        print(
            "-" * 98
        )

        for row in table:
            print(
                f"{row['tip_voltage_V']:10.4g}"
                f"{row['gate1_voltage_V']:18.10e}"
                f"{row['gate2_voltage_V']:18.10e}"
                f"{row['gate1_charge_residual_C']:20.6e}"
                f"{row['gate2_charge_residual_C']:20.6e}"
            )

        print(
            "-" * 98
        )

        print(
            "\nWrote:"
        )

        print(
            f"  {msh_path}"
        )

        print(
            f"  "
            f"{a.output / 'device_mesh.xdmf'}"
        )

        print(
            f"  "
            f"{a.output / 'device_mesh.h5'}"
        )

        print(
            f"  {summary}"
        )

        print(
            "  solution_tip_*.xdmf + .h5"
        )

        print(
            f"  probe_z"
            f"{a.probe_z:g}"
            "_tip_*.csv"
        )

        if zero_key is not None:
            print(
                "  "
                "delta_phi_tip_*_minus_0V.xdmf "
                "+ .h5"
            )

            print(
                f"  delta_probe_z"
                f"{a.probe_z:g}"
                "_tip_*_minus_0V.csv"
            )


if __name__ == "__main__":
    main()
