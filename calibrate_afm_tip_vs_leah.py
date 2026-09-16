#!/usr/bin/env python3

"""
AFM-tip driven Si/SiGe electrostatics -- CALIBRATION variant.

This is sige_afm_tip_two_probes.py's geometry/solver, unchanged, with
three differences aimed at matching Leah's basePotential3d(1).vtk:

1. Runs are numbered under notebooks/afm_vs_leah_compare/runN instead
   of results/sige_adm_tip/runN, so afm_tip_vs_leah_vtk_comparison.ipynb
   can point at successive attempts.
2. --bottom-voltage defaults to -4.4 V (Leah's VTK's own bottom face,
   confirmed by direct inspection with inspect_base_vtk.py) instead of
   0 V.
3. The cone/shank can now be specified by HALF-ANGLE directly
   (--cone-half-angle-deg) instead of only indirectly via
   --shank-radius/--cone-height -- a more natural knob for calibrating
   against an unknown real tip shape.

Nothing about Leah's actual tip geometry, gap, or voltage is known --
only the effect it left in the VTK (a footprint on z=0 peaking at
-1.235V at the center, decaying to -0.06V by r=150nm, on a -4.4V
bottom). Every geometry parameter here is a free variable to sweep,
run over run, until the exported comparison notebook shows <1%
relative error against that VTK.

Each execution automatically creates:

    notebooks/afm_vs_leah_compare/run1
    notebooks/afm_vs_leah_compare/run2
    notebooks/afm_vs_leah_compare/run3
    ...

unless --output is explicitly supplied.

Geometry
--------
Air:
    -air_height <= z <= 0

Semiconductor stack:
    0 ->    2 nm : Si
    2 ->   30 nm : SiGe
   30 ->   40 nm : Si
                  passive circular probe at z=35 nm
   40 ->   43 nm : SiGe
   43 ->   53 nm : Si
                  passive circular probe at z=48 nm
   53 -> 2053 nm : SiGe buffer

AFM tip:
    conductive
    default voltage = +1 V (uncalibrated -- sweep this too)

Bottom/back gate:
    z = 2053 nm
    default voltage = -4.4 V (matches Leah's VTK)

The two internal circular disks are passive probe surfaces.
They do NOT receive Dirichlet voltages.

Equation:
    div(eps_r grad(phi)) = 0
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import json
import os
import re

import gmsh
import numpy as np
import ufl

from dolfinx import fem, geometry, io


# ---------------------------------------------------------------------
# DOLFINx compatibility
# ---------------------------------------------------------------------

try:
    from dolfinx.io import gmsh as gmshio
except ImportError:
    from dolfinx.io import gmshio

try:
    from dolfinx.fem.petsc import LinearProblem
except ImportError:
    from dolfinx.fem import petsc as fem_petsc
    LinearProblem = fem_petsc.LinearProblem


# =====================================================================
# Material tags
# =====================================================================

MAT_AIR = 1
MAT_SI_TOP = 2
MAT_SIGE_28 = 3
MAT_SI_PROBE1 = 4
MAT_SIGE_3 = 5
MAT_SI_PROBE2 = 6
MAT_SIGE_BUFFER = 7


# =====================================================================
# Facet tags
# =====================================================================

FACET_TIP = 101
FACET_BOTTOM = 102
FACET_PROBE1 = 103
FACET_PROBE2 = 104
FACET_OUTER = 105


# =====================================================================
# Arguments
# =====================================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="AFM-tip driven Si/SiGe electrostatics"
    )

    # Domain
    p.add_argument("--lx", type=float, default=300.0)
    p.add_argument("--ly", type=float, default=300.0)

    # Air
    p.add_argument(
        "--air-height",
        type=float,
        default=260.0,
        help="Must exceed gap+2*tip_radius+cone_height+shaft_height or the shaft gets clipped"
    )

    # Materials
    p.add_argument(
        "--eps-air",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--eps-si",
        type=float,
        default=11.7
    )

    p.add_argument(
        "--eps-sige",
        type=float,
        default=12.0
    )

    # AFM geometry
    p.add_argument(
        "--gap",
        type=float,
        default=30.0,
        help="Tip apex height above the sample surface [nm]"
    )

    p.add_argument(
        "--tip-radius",
        type=float,
        default=20.0
    )

    p.add_argument(
        "--cone-height",
        type=float,
        default=100.0
    )

    p.add_argument(
        "--shank-radius",
        type=float,
        default=60.0,
        help="Ignored if --cone-half-angle-deg is given (that computes this instead)"
    )

    p.add_argument(
        "--cone-half-angle-deg",
        type=float,
        default=None,
        help=(
            "If given, OVERRIDES --shank-radius: "
            "shank_radius = tip_radius + cone_height * tan(angle). "
            "A more direct/physical knob than radius+height for sweeping the "
            "tip's taper shape (typical real AFM tip half-angles are roughly 10-20deg)."
        )
    )

    p.add_argument(
        "--shaft-radius",
        type=float,
        default=60.0,
        help="Should match the cone's TOP radius (--shank-radius, or the computed "
             "value if --cone-half-angle-deg is used) to connect flush"
    )

    p.add_argument(
        "--shaft-height",
        type=float,
        default=60.0
    )

    # Voltages
    p.add_argument(
        "--tip-voltage",
        type=float,
        default=1.0,
        help="Uncalibrated -- Leah's actual tip bias is unknown, sweep this"
    )

    p.add_argument(
        "--bottom-voltage",
        type=float,
        default=-4.4,
        help="Matches Leah's VTK bottom face exactly (confirmed via inspect_base_vtk.py)"
    )

    # Passive probes
    p.add_argument(
        "--probe1-radius",
        type=float,
        default=10.0
    )

    p.add_argument(
        "--probe2-radius",
        type=float,
        default=10.0
    )

    # Mesh
    p.add_argument(
        "--h-apex",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--h-device",
        type=float,
        default=2.0
    )

    p.add_argument(
        "--h-near",
        type=float,
        default=5.0
    )

    p.add_argument(
        "--h-bottom",
        type=float,
        default=100.0
    )

    p.add_argument(
        "--degree",
        type=int,
        default=1
    )

    # Output handling
    p.add_argument(
        "--results-root",
        type=str,
        default="notebooks/afm_vs_leah_compare",
        help=(
            "Root directory containing run1, run2, run3, ..."
        )
    )

    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional explicit output folder. "
            "If omitted, next runN folder is chosen automatically."
        )
    )

    return p.parse_args()


# =====================================================================
# Output folder handling
# =====================================================================

def next_run_directory(root):

    """
    Return the next unused folder:

        root/run1
        root/run2
        root/run3
        ...

    Existing unrelated files/folders are ignored.
    """

    os.makedirs(
        root,
        exist_ok=True
    )

    run_numbers = []

    pattern = re.compile(
        r"^run(\d+)$"
    )

    for name in os.listdir(root):

        full_path = os.path.join(
            root,
            name
        )

        if not os.path.isdir(full_path):
            continue

        match = pattern.match(name)

        if match:
            run_numbers.append(
                int(match.group(1))
            )

    if len(run_numbers) == 0:
        next_number = 1
    else:
        next_number = max(run_numbers) + 1

    return os.path.join(
        root,
        f"run{next_number}"
    )


# =====================================================================
# Helpers
# =====================================================================

def root_print(comm, *args):

    if comm.rank == 0:
        print(*args, flush=True)


def physical(dim, entities, tag, name):

    entities = list(entities)

    if len(entities) == 0:
        raise RuntimeError(
            f"Physical group '{name}' is empty."
        )

    gmsh.model.addPhysicalGroup(
        dim,
        entities,
        tag
    )

    gmsh.model.setPhysicalName(
        dim,
        tag,
        name
    )


# =====================================================================
# Geometry
# =====================================================================

def build_geometry(a, comm):

    # If a half-angle was given, it OVERRIDES --shank-radius: recompute
    # shank_radius from tip_radius/cone_height/angle instead. Mutating
    # `a` here (rather than returning a separate value) means every
    # downstream use of a.shank_radius -- including the run_results.json
    # metadata written in main() -- automatically sees the effective
    # value actually used for meshing, not the stale CLI default.
    if a.cone_half_angle_deg is not None:
        a.shank_radius = a.tip_radius + a.cone_height * np.tan(np.radians(a.cone_half_angle_deg))

    # Semiconductor interfaces
    z0 = 0.0
    z1 = 2.0
    z2 = 30.0
    z3 = 40.0
    z4 = 43.0
    z5 = 53.0
    z6 = 2053.0

    z_probe1 = 35.0
    z_probe2 = 48.0

    if comm.rank == 0:

        gmsh.initialize()

        gmsh.option.setNumber(
            "General.Terminal",
            1
        )

        gmsh.model.add(
            "sige_afm_tip"
        )

        occ = gmsh.model.occ

        xmin = -a.lx / 2.0
        xmax = +a.lx / 2.0

        ymin = -a.ly / 2.0
        ymax = +a.ly / 2.0

        # -------------------------------------------------------------
        # Semiconductor volumes
        # -------------------------------------------------------------

        v1 = occ.addBox(
            xmin, ymin, z0,
            a.lx, a.ly, z1-z0
        )

        v2 = occ.addBox(
            xmin, ymin, z1,
            a.lx, a.ly, z2-z1
        )

        v3 = occ.addBox(
            xmin, ymin, z2,
            a.lx, a.ly, z3-z2
        )

        v4 = occ.addBox(
            xmin, ymin, z3,
            a.lx, a.ly, z4-z3
        )

        v5 = occ.addBox(
            xmin, ymin, z4,
            a.lx, a.ly, z5-z4
        )

        v6 = occ.addBox(
            xmin, ymin, z5,
            a.lx, a.ly, z6-z5
        )

        # -------------------------------------------------------------
        # Air
        # -------------------------------------------------------------

        air = occ.addBox(
            xmin,
            ymin,
            -a.air_height,
            a.lx,
            a.ly,
            a.air_height
        )

        # -------------------------------------------------------------
        # AFM tip
        #
        # z is positive downward into the sample.
        # Therefore air and AFM tip are at negative z.
        # -------------------------------------------------------------

        sphere_center_z = -(
            a.gap
            + a.tip_radius
        )

        sphere = occ.addSphere(
            0.0,
            0.0,
            sphere_center_z,
            a.tip_radius
        )

        cone = occ.addCone(
            0.0,
            0.0,
            sphere_center_z,
            0.0,
            0.0,
            -a.cone_height,
            a.tip_radius,
            a.shank_radius
        )

        # Cylindrical shaft, sitting directly on top of the cone's wide
        # end (continues further -z from where the cone stops). Radius
        # should match --shank-radius so it connects flush.
        z_cone_top = sphere_center_z - a.cone_height

        shaft = occ.addCylinder(
            0.0,
            0.0,
            z_cone_top,
            0.0,
            0.0,
            -a.shaft_height,
            a.shaft_radius
        )

        # Fuse sphere + cone + shaft in ONE call. Splitting this into two
        # sequential pairwise fuses is unreliable here -- OCC's boolean
        # fuse intermittently returns an empty result on the second call
        # (leaves two separate solids instead of merging), which then
        # makes the downstream air-cut boolean fail with a confusing
        # "BOPAlgo_AlertTooFewArguments". A single 3-way fuse does not
        # have this problem.
        tip, _ = occ.fuse(
            [(3, sphere)],
            [(3, cone), (3, shaft)],
            removeObject=True,
            removeTool=True
        )

        occ.synchronize()

        air_cut, _ = occ.cut(
            [(3, air)],
            tip,
            removeObject=True,
            removeTool=True
        )

        # -------------------------------------------------------------
        # Make all dielectric interfaces conforming
        # -------------------------------------------------------------

        tools = [
            (3, v2),
            (3, v3),
            (3, v4),
            (3, v5),
            (3, v6)
        ]

        tools.extend(
            air_cut
        )

        occ.fragment(
            [(3, v1)],
            tools,
            removeObject=True,
            removeTool=True
        )

        occ.synchronize()

        # -------------------------------------------------------------
        # Classify volumes
        # -------------------------------------------------------------

        air_vols = []
        si_top = []
        sige28 = []
        si_upper = []
        sige3 = []
        si_lower = []
        buffer = []

        for dim, tag in gmsh.model.getEntities(3):

            _, _, cz = occ.getCenterOfMass(
                dim,
                tag
            )

            if cz < 0.0:
                air_vols.append(tag)

            elif cz < z1:
                si_top.append(tag)

            elif cz < z2:
                sige28.append(tag)

            elif cz < z3:
                si_upper.append(tag)

            elif cz < z4:
                sige3.append(tag)

            elif cz < z5:
                si_lower.append(tag)

            else:
                buffer.append(tag)

        material_groups = [
            (MAT_AIR, air_vols, "air"),
            (MAT_SI_TOP, si_top, "Si_2nm"),
            (MAT_SIGE_28, sige28, "SiGe_28nm"),
            (MAT_SI_PROBE1, si_upper, "Si_10nm_upper"),
            (MAT_SIGE_3, sige3, "SiGe_3nm"),
            (MAT_SI_PROBE2, si_lower, "Si_10nm_lower"),
            (
                MAT_SIGE_BUFFER,
                buffer,
                "SiGe_buffer_2000nm"
            )
        ]

        for tag, volumes, name in material_groups:

            physical(
                3,
                volumes,
                tag,
                name
            )

        # -------------------------------------------------------------
        # Passive circular probe surfaces
        # -------------------------------------------------------------

        probe1 = occ.addDisk(
            0.0,
            0.0,
            z_probe1,
            a.probe1_radius,
            a.probe1_radius
        )

        probe2 = occ.addDisk(
            0.0,
            0.0,
            z_probe2,
            a.probe2_radius,
            a.probe2_radius
        )

        occ.synchronize()

        for volume in si_upper:

            gmsh.model.mesh.embed(
                2,
                [probe1],
                3,
                volume
            )

        for volume in si_lower:

            gmsh.model.mesh.embed(
                2,
                [probe2],
                3,
                volume
            )

        physical(
            2,
            [probe1],
            FACET_PROBE1,
            "probe1"
        )

        physical(
            2,
            [probe2],
            FACET_PROBE2,
            "probe2"
        )

        # -------------------------------------------------------------
        # Tip surfaces
        # -------------------------------------------------------------

        tip_surfaces = []

        tol = 1.0e-6

        for air_tag in air_vols:

            boundary = gmsh.model.getBoundary(
                [(3, air_tag)],
                oriented=False,
                recursive=False
            )

            for dim, tag in boundary:

                if dim != 2:
                    continue

                bbox = gmsh.model.getBoundingBox(
                    2,
                    tag
                )

                sx0, sy0, sz0, sx1, sy1, sz1 = bbox

                on_box = (
                    (
                        abs(sx0-xmin) < tol
                        and abs(sx1-xmin) < tol
                    )
                    or
                    (
                        abs(sx0-xmax) < tol
                        and abs(sx1-xmax) < tol
                    )
                    or
                    (
                        abs(sy0-ymin) < tol
                        and abs(sy1-ymin) < tol
                    )
                    or
                    (
                        abs(sy0-ymax) < tol
                        and abs(sy1-ymax) < tol
                    )
                    or
                    (
                        abs(sz0+a.air_height) < tol
                        and abs(sz1+a.air_height) < tol
                    )
                    or
                    (
                        abs(sz0) < tol
                        and abs(sz1) < tol
                    )
                )

                if not on_box:
                    tip_surfaces.append(tag)

        tip_surfaces = sorted(
            set(tip_surfaces)
        )

        physical(
            2,
            tip_surfaces,
            FACET_TIP,
            "AFM_tip"
        )

        # -------------------------------------------------------------
        # Bottom and outer surfaces
        # -------------------------------------------------------------

        bottom_surfaces = []
        outer_surfaces = []

        for dim, tag in gmsh.model.getEntities(2):

            if tag in [
                probe1,
                probe2
            ]:
                continue

            bbox = gmsh.model.getBoundingBox(
                2,
                tag
            )

            sx0, sy0, sz0, sx1, sy1, sz1 = bbox

            if (
                abs(sz0-z6) < tol
                and
                abs(sz1-z6) < tol
            ):

                bottom_surfaces.append(tag)

            elif (
                (
                    abs(sx0-xmin) < tol
                    and abs(sx1-xmin) < tol
                )
                or
                (
                    abs(sx0-xmax) < tol
                    and abs(sx1-xmax) < tol
                )
                or
                (
                    abs(sy0-ymin) < tol
                    and abs(sy1-ymin) < tol
                )
                or
                (
                    abs(sy0-ymax) < tol
                    and abs(sy1-ymax) < tol
                )
                or
                (
                    abs(sz0+a.air_height) < tol
                    and abs(sz1+a.air_height) < tol
                )
            ):

                outer_surfaces.append(tag)

        physical(
            2,
            bottom_surfaces,
            FACET_BOTTOM,
            "bottom_back_gate"
        )

        physical(
            2,
            sorted(set(outer_surfaces)),
            FACET_OUTER,
            "outer"
        )

        # -------------------------------------------------------------
        # Mesh refinement near AFM tip
        # -------------------------------------------------------------

        distance_tip = gmsh.model.mesh.field.add(
            "Distance"
        )

        gmsh.model.mesh.field.setNumbers(
            distance_tip,
            "SurfacesList",
            tip_surfaces
        )

        gmsh.model.mesh.field.setNumber(
            distance_tip,
            "Sampling",
            100
        )

        tip_field = gmsh.model.mesh.field.add(
            "Threshold"
        )

        gmsh.model.mesh.field.setNumber(
            tip_field,
            "InField",
            distance_tip
        )

        gmsh.model.mesh.field.setNumber(
            tip_field,
            "SizeMin",
            a.h_apex
        )

        gmsh.model.mesh.field.setNumber(
            tip_field,
            "SizeMax",
            a.h_near
        )

        gmsh.model.mesh.field.setNumber(
            tip_field,
            "DistMin",
            10.0
        )

        gmsh.model.mesh.field.setNumber(
            tip_field,
            "DistMax",
            100.0
        )

        # -------------------------------------------------------------
        # Central device refinement
        # -------------------------------------------------------------

        device_field = gmsh.model.mesh.field.add(
            "Box"
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "VIn",
            a.h_device
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "VOut",
            a.h_bottom
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "XMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "XMax",
            60.0
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "YMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "YMax",
            60.0
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "ZMin",
            -30.0
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "ZMax",
            80.0
        )

        # -------------------------------------------------------------
        # Coarsen with depth
        # -------------------------------------------------------------

        depth_field = gmsh.model.mesh.field.add(
            "MathEval"
        )

        expression = (
            f"{a.h_near}"
            f"+({a.h_bottom}-{a.h_near})"
            f"*max(0,z-80)/({z6}-80)"
        )

        gmsh.model.mesh.field.setString(
            depth_field,
            "F",
            expression
        )

        background = gmsh.model.mesh.field.add(
            "Min"
        )

        gmsh.model.mesh.field.setNumbers(
            background,
            "FieldsList",
            [
                tip_field,
                device_field,
                depth_field
            ]
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(
            background
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeExtendFromBoundary",
            0
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeFromPoints",
            0
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature",
            0
        )

        gmsh.option.setNumber(
            "Mesh.Algorithm3D",
            10
        )

        root_print(
            comm,
            ""
        )

        root_print(
            comm,
            "Generating AFM + Si/SiGe mesh..."
        )

        gmsh.model.mesh.generate(3)

    # -------------------------------------------------------------
    # Convert Gmsh mesh
    # -------------------------------------------------------------

    mesh_data = gmshio.model_to_mesh(
        gmsh.model,
        comm,
        0,
        gdim=3
    )

    if hasattr(mesh_data, "mesh"):

        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
        facet_tags = mesh_data.facet_tags

    else:

        domain, cell_tags, facet_tags = mesh_data

    if comm.rank == 0:
        gmsh.finalize()

    return (
        domain,
        cell_tags,
        facet_tags,
        z_probe1,
        z_probe2,
        z6
    )


# =====================================================================
# Material fields
# =====================================================================

def make_material_fields(
    domain,
    cell_tags,
    a
):

    try:

        Q = fem.functionspace(
            domain,
            ("DG", 0)
        )

    except AttributeError:

        Q = fem.FunctionSpace(
            domain,
            ("DG", 0)
        )

    epsilon = fem.Function(Q)
    epsilon.name = "relative_permittivity"

    material_id = fem.Function(Q)
    material_id.name = "material_id"

    epsilon.x.array[:] = 0.0
    material_id.x.array[:] = 0.0

    materials = {
        MAT_AIR: a.eps_air,
        MAT_SI_TOP: a.eps_si,
        MAT_SIGE_28: a.eps_sige,
        MAT_SI_PROBE1: a.eps_si,
        MAT_SIGE_3: a.eps_sige,
        MAT_SI_PROBE2: a.eps_si,
        MAT_SIGE_BUFFER: a.eps_sige
    }

    tdim = domain.topology.dim

    root_print(
        domain.comm,
        ""
    )

    root_print(
        domain.comm,
        "Material tag integrity:"
    )

    for marker, eps_value in materials.items():

        cells = cell_tags.find(
            marker
        )

        count = domain.comm.allreduce(
            len(cells),
            op=MPI.SUM
        )

        root_print(
            domain.comm,
            f"  tag={marker:2d} "
            f"cells={count:,} "
            f"eps_r={eps_value}"
        )

        if count == 0:

            raise RuntimeError(
                f"Material tag {marker} is empty."
            )

        dofs = fem.locate_dofs_topological(
            Q,
            tdim,
            cells
        )

        epsilon.x.array[dofs] = eps_value

        material_id.x.array[dofs] = float(marker)

    epsilon.x.scatter_forward()
    material_id.x.scatter_forward()

    return (
        epsilon,
        material_id
    )


# =====================================================================
# Solve
# =====================================================================

def solve(
    domain,
    cell_tags,
    facet_tags,
    a
):

    tdim = domain.topology.dim
    fdim = tdim - 1

    try:

        V = fem.functionspace(
            domain,
            ("Lagrange", a.degree)
        )

    except AttributeError:

        V = fem.FunctionSpace(
            domain,
            ("CG", a.degree)
        )

    epsilon, material_id = make_material_fields(
        domain,
        cell_tags,
        a
    )

    def make_bc(
        marker,
        voltage
    ):

        facets = facet_tags.find(
            marker
        )

        count = domain.comm.allreduce(
            len(facets),
            op=MPI.SUM
        )

        if count == 0:

            raise RuntimeError(
                f"Facet tag {marker} is empty."
            )

        dofs = fem.locate_dofs_topological(
            V,
            fdim,
            facets
        )

        value = fem.Constant(
            domain,
            PETSc.ScalarType(voltage)
        )

        bc = fem.dirichletbc(
            value,
            dofs,
            V
        )

        return bc

    # ONLY imposed voltages
    bc_tip = make_bc(
        FACET_TIP,
        a.tip_voltage
    )

    bc_bottom = make_bc(
        FACET_BOTTOM,
        a.bottom_voltage
    )

    bcs = [
        bc_tip,
        bc_bottom
    ]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dx = ufl.Measure(
        "dx",
        domain=domain
    )

    zero = fem.Constant(
        domain,
        PETSc.ScalarType(0.0)
    )

    lhs = (
        epsilon
        * ufl.inner(
            ufl.grad(u),
            ufl.grad(v)
        )
        * dx
    )

    rhs = (
        zero
        * v
        * dx
    )

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 3000,
        "ksp_error_if_not_converged": True
    }

    try:

        problem = LinearProblem(
            lhs,
            rhs,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="sige_afm_"
        )

    except TypeError:

        problem = LinearProblem(
            lhs,
            rhs,
            bcs=bcs,
            petsc_options=petsc_options
        )

    root_print(
        domain.comm,
        ""
    )

    root_print(
        domain.comm,
        "Solving Laplace equation..."
    )

    phi = problem.solve()

    phi.name = "phi_V"

    phi.x.scatter_forward()

    return (
        phi,
        epsilon,
        material_id,
        V,
        problem
    )


# =====================================================================
# Probe statistics
# =====================================================================

def analyze_probe(
    domain,
    V,
    phi,
    facet_tags,
    marker,
    name
):

    fdim = domain.topology.dim - 1

    facets = facet_tags.find(
        marker
    )

    global_facets = domain.comm.allreduce(
        len(facets),
        op=MPI.SUM
    )

    if global_facets == 0:

        return {
            "name": name,
            "facets": 0,
            "dofs": 0,
            "min": np.nan,
            "max": np.nan,
            "mean_nodal": np.nan
        }

    dofs = fem.locate_dofs_topological(
        V,
        fdim,
        facets
    )

    values = np.real(
        phi.x.array[dofs]
    )

    if len(values) > 0:

        local_min = np.min(values)
        local_max = np.max(values)
        local_sum = np.sum(values)
        local_count = len(values)

    else:

        local_min = np.inf
        local_max = -np.inf
        local_sum = 0.0
        local_count = 0

    global_min = domain.comm.allreduce(
        local_min,
        op=MPI.MIN
    )

    global_max = domain.comm.allreduce(
        local_max,
        op=MPI.MAX
    )

    global_sum = domain.comm.allreduce(
        local_sum,
        op=MPI.SUM
    )

    global_count = domain.comm.allreduce(
        local_count,
        op=MPI.SUM
    )

    mean = (
        global_sum / global_count
        if global_count > 0
        else np.nan
    )

    return {
        "name": name,
        "facets": int(global_facets),
        "dofs": int(global_count),
        "min": float(global_min),
        "max": float(global_max),
        "mean_nodal": float(mean)
    }


# =====================================================================
# Main
# =====================================================================

def main():

    a = parse_args()

    comm = MPI.COMM_WORLD

    # -------------------------------------------------------------
    # Automatically assign run directory
    # -------------------------------------------------------------

    if comm.rank == 0:

        if a.output is None:

            output = next_run_directory(
                a.results_root
            )

        else:

            output = a.output

        os.makedirs(
            output,
            exist_ok=False
        )

    else:

        output = None

    output = comm.bcast(
        output,
        root=0
    )

    a.output = output

    comm.barrier()

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "=" * 78
    )

    root_print(
        comm,
        "AFM-TIP DRIVEN Si/SiGe ELECTROSTATICS"
    )

    root_print(
        comm,
        "=" * 78
    )

    root_print(
        comm,
        f"OUTPUT RUN          : {a.output}"
    )

    root_print(
        comm,
        f"AFM voltage         : "
        f"{a.tip_voltage:.6f} V"
    )

    root_print(
        comm,
        f"Bottom/back gate    : "
        f"{a.bottom_voltage:.6f} V"
    )

    root_print(
        comm,
        "Probe 1 voltage     : FREE / not prescribed"
    )

    root_print(
        comm,
        "Probe 2 voltage     : FREE / not prescribed"
    )

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "Stack:"
    )

    root_print(
        comm,
        "   0 ->    2 nm : Si"
    )

    root_print(
        comm,
        "   2 ->   30 nm : SiGe"
    )

    root_print(
        comm,
        "  30 ->   40 nm : Si"
    )

    root_print(
        comm,
        "                 Probe 1 at z=35 nm"
    )

    root_print(
        comm,
        "  40 ->   43 nm : SiGe"
    )

    root_print(
        comm,
        "  43 ->   53 nm : Si"
    )

    root_print(
        comm,
        "                 Probe 2 at z=48 nm"
    )

    root_print(
        comm,
        "  53 -> 2053 nm : SiGe buffer"
    )

    (
        domain,
        cell_tags,
        facet_tags,
        z_probe1,
        z_probe2,
        z_bottom
    ) = build_geometry(
        a,
        comm
    )

    if cell_tags is None:

        raise RuntimeError(
            "Cell tags did not survive Gmsh conversion."
        )

    if facet_tags is None:

        raise RuntimeError(
            "Facet tags did not survive Gmsh conversion."
        )

    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(
        fdim,
        tdim
    )

    domain.topology.create_connectivity(
        tdim,
        fdim
    )

    # -------------------------------------------------------------
    # Facet tag check
    # -------------------------------------------------------------

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "Facet tag integrity:"
    )

    facet_groups = {
        "AFM tip": FACET_TIP,
        "Bottom": FACET_BOTTOM,
        "Probe 1": FACET_PROBE1,
        "Probe 2": FACET_PROBE2,
        "Outer": FACET_OUTER
    }

    for name, marker in facet_groups.items():

        facets = facet_tags.find(
            marker
        )

        count = comm.allreduce(
            len(facets),
            op=MPI.SUM
        )

        root_print(
            comm,
            f"  {name:10s} "
            f"tag={marker:3d} "
            f"facets={count:,}"
        )

    # -------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------

    (
        phi,
        epsilon,
        material_id,
        V,
        problem
    ) = solve(
        domain,
        cell_tags,
        facet_tags,
        a
    )

    values = np.real(
        phi.x.array
    )

    local_min = (
        np.min(values)
        if len(values)
        else np.inf
    )

    local_max = (
        np.max(values)
        if len(values)
        else -np.inf
    )

    phi_min = comm.allreduce(
        local_min,
        op=MPI.MIN
    )

    phi_max = comm.allreduce(
        local_max,
        op=MPI.MAX
    )

    # -------------------------------------------------------------
    # Probe potentials
    # -------------------------------------------------------------

    probe1 = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE1,
        "Probe 1"
    )

    probe2 = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE2,
        "Probe 2"
    )

    global_cells = (
        domain.topology.index_map(
            tdim
        ).size_global
    )

    global_dofs = (
        V.dofmap.index_map.size_global
        * V.dofmap.index_map_bs
    )

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "=" * 78
    )

    root_print(
        comm,
        "SOLVER RESULTS"
    )

    root_print(
        comm,
        "=" * 78
    )

    root_print(
        comm,
        f"Cells              : {global_cells:,}"
    )

    root_print(
        comm,
        f"DOFs               : {global_dofs:,}"
    )

    root_print(
        comm,
        f"phi min            : {phi_min:.12e} V"
    )

    root_print(
        comm,
        f"phi max            : {phi_max:.12e} V"
    )

    try:

        root_print(
            comm,
            f"KSP iterations     : "
            f"{problem.solver.getIterationNumber()}"
        )

        root_print(
            comm,
            f"KSP reason         : "
            f"{problem.solver.getConvergedReason()}"
        )

    except Exception:

        pass

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "INDUCED PROBE POTENTIALS"
    )

    root_print(
        comm,
        "-" * 78
    )

    for result, z in [
        (probe1, z_probe1),
        (probe2, z_probe2)
    ]:

        root_print(
            comm,
            f"{result['name']} at z={z:.1f} nm"
        )

        root_print(
            comm,
            f"  min phi          : "
            f"{result['min']:.12e} V"
        )

        root_print(
            comm,
            f"  max phi          : "
            f"{result['max']:.12e} V"
        )

        root_print(
            comm,
            f"  mean nodal phi   : "
            f"{result['mean_nodal']:.12e} V"
        )

    # -------------------------------------------------------------
    # Outputs: ONE combined XDMF/H5 per run, so ParaView opens a
    # single file and lets you choose to color by phi_V, material_id,
    # relative_permittivity, cell_tags, or facet_tags -- instead of
    # three separate files with no built-in link between them.
    # -------------------------------------------------------------

    combined_path = os.path.join(
        a.output,
        "sige_afm_tip.xdmf"
    )

    cell_tags.name = "cell_tags"
    facet_tags.name = "facet_tags"

    with io.XDMFFile(
        comm,
        combined_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        xdmf.write_function(
            phi
        )

        xdmf.write_function(
            epsilon
        )

        xdmf.write_function(
            material_id
        )

        try:

            xdmf.write_meshtags(
                cell_tags,
                domain.geometry
            )

            xdmf.write_meshtags(
                facet_tags,
                domain.geometry
            )

        except TypeError:

            xdmf.write_meshtags(
                cell_tags
            )

            xdmf.write_meshtags(
                facet_tags
            )

    root_print(
        comm,
        f"Wrote {combined_path} "
        "(phi_V, relative_permittivity, material_id, cell_tags, facet_tags)"
    )

    # -------------------------------------------------------------
    # Centerline: phi(0, 0, z) from the AFM tip apex down through the
    # air gap, both probe depths, and into the buffer/back gate --
    # exactly the "how does the tip's potential penetrate through the
    # stack" profile that matters for this AFM-tip case.
    # -------------------------------------------------------------

    z_fine = np.arange(-a.air_height, 100.0, 1.0)
    z_coarse = np.arange(100.0, z_bottom, 20.0)
    z_line = np.concatenate([z_fine, z_coarse, [z_bottom]])

    points = np.zeros((len(z_line), 3))
    points[:, 2] = z_line

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    # Each rank only owns a local mesh partition, so under mpiexec -n>1
    # a given centerline point is typically found on only ONE rank.
    # Accumulate a (value, found-count) pair per point and allreduce-sum
    # both across ranks -- summing is safe here since at most one rank
    # contributes a nonzero value per point (rather than reducing with
    # rank-0-only local data, which would silently drop most points).
    phi_line_local = np.zeros(len(z_line))
    found_local = np.zeros(len(z_line))
    local_cells, local_idx = [], []

    for i in range(len(z_line)):
        links_i = colliding.links(i)
        if len(links_i) > 0:
            local_idx.append(i)
            local_cells.append(links_i[0])

    if local_idx:
        vals = phi.eval(
            points[local_idx],
            np.array(local_cells, dtype=np.int32)
        )[:, 0]
        for k, gi in enumerate(local_idx):
            phi_line_local[gi] = np.real(vals[k])
            found_local[gi] = 1.0

    phi_line_sum = comm.allreduce(phi_line_local, op=MPI.SUM)
    found_sum = comm.allreduce(found_local, op=MPI.SUM)

    phi_line = np.where(
        found_sum > 0,
        phi_line_sum / np.maximum(found_sum, 1.0),
        np.nan
    )

    if comm.rank == 0:

        centerline_path = os.path.join(
            a.output,
            "centerline.csv"
        )

        with open(centerline_path, "w") as f:
            f.write("z_nm,phi_V\n")
            for zval, pval in zip(z_line, phi_line):
                if np.isfinite(pval):
                    f.write(f"{zval:.4f},{pval:.6e}\n")

        root_print(
            comm,
            f"Wrote {centerline_path}"
        )

    # -------------------------------------------------------------
    # JSON metadata/results
    # -------------------------------------------------------------

    if comm.rank == 0:

        results = {

            "output": a.output,

            "geometry_nm": {
                "lx": a.lx,
                "ly": a.ly,
                "air_height": a.air_height,
                "tip_gap": a.gap,
                "tip_radius": a.tip_radius,
                "cone_height": a.cone_height,
                "shank_radius": a.shank_radius,
                "cone_half_angle_deg_input": a.cone_half_angle_deg,
                "cone_half_angle_deg_effective": float(np.degrees(np.arctan(
                    (a.shank_radius - a.tip_radius) / a.cone_height
                ))),
                "shaft_radius": a.shaft_radius,
                "shaft_height": a.shaft_height,
                "probe1_z": z_probe1,
                "probe2_z": z_probe2,
                "probe1_radius": a.probe1_radius,
                "probe2_radius": a.probe2_radius,
                "bottom_z": z_bottom
            },

            "voltages_V": {
                "AFM_tip": a.tip_voltage,
                "bottom_back_gate": a.bottom_voltage,
                "probe1": None,
                "probe2": None
            },

            "materials": {
                "eps_air": a.eps_air,
                "eps_si": a.eps_si,
                "eps_sige": a.eps_sige
            },

            "mesh": {
                "cells": int(global_cells),
                "dofs": int(global_dofs),
                "degree": a.degree,
                "h_apex_nm": a.h_apex,
                "h_device_nm": a.h_device,
                "h_near_nm": a.h_near,
                "h_bottom_nm": a.h_bottom
            },

            "solution": {
                "phi_min_V": float(phi_min),
                "phi_max_V": float(phi_max),
                "probe1": probe1,
                "probe2": probe2
            }
        }

        with open(
            os.path.join(
                a.output,
                "run_results.json"
            ),
            "w"
        ) as f:

            json.dump(
                results,
                f,
                indent=2
            )

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "Output files:"
    )

    root_print(
        comm,
        f"  {a.output}/sige_afm_tip.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/sige_afm_tip.h5"
    )

    root_print(
        comm,
        f"  {a.output}/run_results.json"
    )

    root_print(
        comm,
        f"  {a.output}/centerline.csv"
    )

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        f"RUN COMPLETE: {a.output}"
    )


if __name__ == "__main__":
    main()
