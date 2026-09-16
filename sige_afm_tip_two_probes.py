#!/usr/bin/env python3

"""
AFM-tip driven Si/SiGe electrostatics.

Coordinate convention
---------------------
z = 0 is the top semiconductor surface.
Air is at z < 0.
The semiconductor stack extends toward positive z.

AFM tip
-------
Conductive AFM tip in air.
Tip voltage = +1 V by default.

The tip consists of:
    spherical apex
    conical shank

The lowest point of the tip is --gap nm above z=0.

Semiconductor stack
-------------------
Layer 1:
    Si
    z = 0 to 2 nm
    eps_r = 11.7

Layer 2:
    SiGe
    z = 2 to 30 nm
    eps_r = 12.0

Layer 3:
    Si
    z = 30 to 40 nm
    eps_r = 11.7

    Probe disk 1:
        z = 35 nm
        radius = 10 nm

Layer 4:
    SiGe
    z = 40 to 43 nm
    eps_r = 12.0

Layer 5:
    Si
    z = 43 to 53 nm
    eps_r = 11.7

    Probe disk 2:
        z = 48 nm
        radius = 10 nm

Layer 6:
    SiGe buffer
    z = 53 to 2053 nm
    eps_r = 12.0

Bottom/back gate:
    z = 2053 nm
    phi = 0 V

IMPORTANT
---------
Probe 1 and Probe 2 DO NOT receive Dirichlet boundary conditions.

They are passive internal surfaces used to inspect the potential induced
by the AFM tip.

Equation:
    div(eps_r grad(phi)) = 0

rho = 0.

Coordinates are in nm.
Potential is in V.
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import json
import os

import gmsh
import numpy as np
import ufl

from dolfinx import fem, io


# =====================================================================
# DOLFINx compatibility
# =====================================================================

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

    # ---------------------------------------------------------------
    # Lateral domain
    # ---------------------------------------------------------------

    p.add_argument(
        "--lx",
        type=float,
        default=300.0,
        help="Domain width x [nm]"
    )

    p.add_argument(
        "--ly",
        type=float,
        default=300.0,
        help="Domain width y [nm]"
    )

    # ---------------------------------------------------------------
    # Air
    # ---------------------------------------------------------------

    p.add_argument(
        "--air-height",
        type=float,
        default=200.0,
        help="Air region height above sample [nm]"
    )

    p.add_argument(
        "--eps-air",
        type=float,
        default=1.0
    )

    # ---------------------------------------------------------------
    # Semiconductor permittivity
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # AFM geometry
    # ---------------------------------------------------------------

    p.add_argument(
        "--gap",
        type=float,
        default=10.0,
        help="Tip-surface gap [nm]"
    )

    p.add_argument(
        "--tip-radius",
        type=float,
        default=20.0,
        help="Spherical tip radius [nm]"
    )

    p.add_argument(
        "--cone-height",
        type=float,
        default=100.0,
        help="Cone height [nm]"
    )

    p.add_argument(
        "--shank-radius",
        type=float,
        default=60.0,
        help="Cone radius at upper end [nm]"
    )

    # ---------------------------------------------------------------
    # Voltages
    # ---------------------------------------------------------------

    p.add_argument(
        "--tip-voltage",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--bottom-voltage",
        type=float,
        default=0.0
    )

    # ---------------------------------------------------------------
    # Probe geometry
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Mesh
    # ---------------------------------------------------------------

    p.add_argument(
        "--h-apex",
        type=float,
        default=1.0,
        help="Mesh size near AFM apex [nm]"
    )

    p.add_argument(
        "--h-device",
        type=float,
        default=2.0,
        help="Mesh size in central device region [nm]"
    )

    p.add_argument(
        "--h-near",
        type=float,
        default=5.0,
        help="Mesh size in near field [nm]"
    )

    p.add_argument(
        "--h-bottom",
        type=float,
        default=100.0,
        help="Mesh size deep in buffer [nm]"
    )

    p.add_argument(
        "--degree",
        type=int,
        default=1
    )

    p.add_argument(
        "--output",
        type=str,
        default="results/sige_afm_tip"
    )

    return p.parse_args()


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
            f"Physical group {name} is empty."
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

    # ----------------------------------------------------------------
    # Fixed semiconductor stack
    # ----------------------------------------------------------------

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

        # ============================================================
        # Semiconductor volumes
        # ============================================================

        v_si_top = occ.addBox(
            xmin,
            ymin,
            z0,
            a.lx,
            a.ly,
            z1 - z0
        )

        v_sige28 = occ.addBox(
            xmin,
            ymin,
            z1,
            a.lx,
            a.ly,
            z2 - z1
        )

        v_si_probe1 = occ.addBox(
            xmin,
            ymin,
            z2,
            a.lx,
            a.ly,
            z3 - z2
        )

        v_sige3 = occ.addBox(
            xmin,
            ymin,
            z3,
            a.lx,
            a.ly,
            z4 - z3
        )

        v_si_probe2 = occ.addBox(
            xmin,
            ymin,
            z4,
            a.lx,
            a.ly,
            z5 - z4
        )

        v_buffer = occ.addBox(
            xmin,
            ymin,
            z5,
            a.lx,
            a.ly,
            z6 - z5
        )

        # ============================================================
        # Air box
        #
        # Air extends from -air_height to z=0.
        # ============================================================

        air = occ.addBox(
            xmin,
            ymin,
            -a.air_height,
            a.lx,
            a.ly,
            a.air_height
        )

        # ============================================================
        # AFM tip
        #
        # Lowest point:
        #
        #     z = -gap
        #
        # Sphere center:
        #
        #     z = -(gap + radius)
        #
        # Cone extends upward toward more-negative z.
        # ============================================================

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

        tip_objects, _ = occ.fuse(
            [(3, sphere)],
            [(3, cone)],
            removeObject=True,
            removeTool=True
        )

        # Remove conducting tip volume from air.
        air_cut, _ = occ.cut(
            [(3, air)],
            tip_objects,
            removeObject=True,
            removeTool=True
        )

        # ============================================================
        # Fragment all dielectric volumes together
        # ============================================================

        objects = [
            (3, v_si_top)
        ]

        tools = [
            (3, v_sige28),
            (3, v_si_probe1),
            (3, v_sige3),
            (3, v_si_probe2),
            (3, v_buffer)
        ]

        tools.extend(
            air_cut
        )

        occ.fragment(
            objects,
            tools,
            removeObject=True,
            removeTool=True
        )

        occ.synchronize()

        # ============================================================
        # Classify volumes by center-of-mass z
        # ============================================================

        air_vols = []
        si_top_vols = []
        sige28_vols = []
        si_probe1_vols = []
        sige3_vols = []
        si_probe2_vols = []
        buffer_vols = []

        for dim, tag in gmsh.model.getEntities(3):

            _, _, cz = occ.getCenterOfMass(
                dim,
                tag
            )

            if cz < 0.0:

                air_vols.append(tag)

            elif cz < z1:

                si_top_vols.append(tag)

            elif cz < z2:

                sige28_vols.append(tag)

            elif cz < z3:

                si_probe1_vols.append(tag)

            elif cz < z4:

                sige3_vols.append(tag)

            elif cz < z5:

                si_probe2_vols.append(tag)

            else:

                buffer_vols.append(tag)

        volume_groups = [
            (
                MAT_AIR,
                air_vols,
                "air"
            ),
            (
                MAT_SI_TOP,
                si_top_vols,
                "Si_2nm"
            ),
            (
                MAT_SIGE_28,
                sige28_vols,
                "SiGe_28nm"
            ),
            (
                MAT_SI_PROBE1,
                si_probe1_vols,
                "Si_10nm_upper"
            ),
            (
                MAT_SIGE_3,
                sige3_vols,
                "SiGe_3nm"
            ),
            (
                MAT_SI_PROBE2,
                si_probe2_vols,
                "Si_10nm_lower"
            ),
            (
                MAT_SIGE_BUFFER,
                buffer_vols,
                "SiGe_buffer_2000nm"
            )
        ]

        for tag, entities, name in volume_groups:

            physical(
                3,
                entities,
                tag,
                name
            )

        # ============================================================
        # Passive circular probe surfaces
        # ============================================================

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

        # Embed into appropriate Si volumes.
        for volume in si_probe1_vols:

            gmsh.model.mesh.embed(
                2,
                [probe1],
                3,
                volume
            )

        for volume in si_probe2_vols:

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

        # ============================================================
        # Find AFM-tip cavity facets
        # ============================================================

        tip_surfaces = []

        # Air should normally be one volume.
        for air_tag in air_vols:

            boundary = gmsh.model.getBoundary(
                [(3, air_tag)],
                oriented=False,
                recursive=False
            )

            for dim, surface_tag in boundary:

                if dim != 2:
                    continue

                bbox = gmsh.model.getBoundingBox(
                    2,
                    surface_tag
                )

                sxmin, symin, szmin, sxmax, symax, szmax = bbox

                tol = 1.0e-6

                on_box = (
                    abs(sxmin - xmin) < tol
                    and abs(sxmax - xmin) < tol
                ) or (
                    abs(sxmin - xmax) < tol
                    and abs(sxmax - xmax) < tol
                ) or (
                    abs(symin - ymin) < tol
                    and abs(symax - ymin) < tol
                ) or (
                    abs(symin - ymax) < tol
                    and abs(symax - ymax) < tol
                ) or (
                    abs(szmin + a.air_height) < tol
                    and abs(szmax + a.air_height) < tol
                ) or (
                    abs(szmin - 0.0) < tol
                    and abs(szmax - 0.0) < tol
                )

                if not on_box:

                    tip_surfaces.append(
                        surface_tag
                    )

        tip_surfaces = sorted(
            set(tip_surfaces)
        )

        physical(
            2,
            tip_surfaces,
            FACET_TIP,
            "AFM_tip_1V"
        )

        # ============================================================
        # Bottom boundary
        # ============================================================

        bottom_surfaces = []

        outer_surfaces = []

        for dim, tag in gmsh.model.getEntities(2):

            if tag in [
                probe1,
                probe2
            ]:
                continue

            bbox = gmsh.model.getBoundingBox(
                dim,
                tag
            )

            sxmin, symin, szmin, sxmax, symax, szmax = bbox

            tol = 1.0e-6

            if (
                abs(szmin - z6) < tol
                and
                abs(szmax - z6) < tol
            ):

                bottom_surfaces.append(
                    tag
                )

            elif (
                (
                    abs(sxmin - xmin) < tol
                    and
                    abs(sxmax - xmin) < tol
                )
                or
                (
                    abs(sxmin - xmax) < tol
                    and
                    abs(sxmax - xmax) < tol
                )
                or
                (
                    abs(symin - ymin) < tol
                    and
                    abs(symax - ymin) < tol
                )
                or
                (
                    abs(symin - ymax) < tol
                    and
                    abs(symax - ymax) < tol
                )
                or
                (
                    abs(szmin + a.air_height) < tol
                    and
                    abs(szmax + a.air_height) < tol
                )
            ):

                outer_surfaces.append(
                    tag
                )

        physical(
            2,
            bottom_surfaces,
            FACET_BOTTOM,
            "bottom_back_gate_0V"
        )

        physical(
            2,
            sorted(set(outer_surfaces)),
            FACET_OUTER,
            "outer"
        )

        # ============================================================
        # Mesh refinement
        #
        # 1. Fine near AFM tip.
        # 2. Fine in central active semiconductor region.
        # 3. Coarse deep in 2000 nm buffer.
        # ============================================================

        # Distance from tip.
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

        tip_threshold = gmsh.model.mesh.field.add(
            "Threshold"
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "InField",
            distance_tip
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "SizeMin",
            a.h_apex
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "SizeMax",
            a.h_near
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "DistMin",
            10.0
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "DistMax",
            100.0
        )

        # Central device refinement.
        central = gmsh.model.mesh.field.add(
            "Box"
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "VIn",
            a.h_device
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "VOut",
            a.h_bottom
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "XMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "XMax",
            +60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "YMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "YMax",
            +60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "ZMin",
            -30.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "ZMax",
            80.0
        )

        # Depth grading.
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
                tip_threshold,
                central,
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
            "Generating AFM + Si/SiGe tetrahedral mesh..."
        )

        gmsh.model.mesh.generate(3)

    # ----------------------------------------------------------------
    # Transfer to DOLFINx
    # ----------------------------------------------------------------

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
# Material field
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

    epsilon.name = (
        "relative_permittivity"
    )

    material_id = fem.Function(Q)

    material_id.name = (
        "material_id"
    )

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

    for marker, eps_value in materials.items():

        cells = cell_tags.find(
            marker
        )

        count = domain.comm.allreduce(
            len(cells),
            op=MPI.SUM
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

        epsilon.x.array[dofs] = (
            eps_value
        )

        material_id.x.array[dofs] = (
            float(marker)
        )

    epsilon.x.scatter_forward()
    material_id.x.scatter_forward()

    return epsilon, material_id


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

    epsilon, material_id = (
        make_material_fields(
            domain,
            cell_tags,
            a
        )
    )

    # ----------------------------------------------------------------
    # BC helper
    # ----------------------------------------------------------------

    def make_bc(marker, voltage):

        facets = facet_tags.find(
            marker
        )

        global_facets = (
            domain.comm.allreduce(
                len(facets),
                op=MPI.SUM
            )
        )

        if global_facets == 0:

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

        return (
            fem.dirichletbc(
                value,
                dofs,
                V
            ),
            facets,
            dofs
        )

    # ================================================================
    # ONLY TWO ELECTRICAL DIRICHLET CONDITIONS
    #
    # 1. AFM tip = 1 V
    # 2. Bottom/back gate = 0 V
    #
    # Probe disks receive NO BC.
    # ================================================================

    bc_tip, tip_facets, tip_dofs = make_bc(
        FACET_TIP,
        a.tip_voltage
    )

    bc_bottom, bottom_facets, bottom_dofs = make_bc(
        FACET_BOTTOM,
        a.bottom_voltage
    )

    bcs = [
        bc_tip,
        bc_bottom
    ]

    # ----------------------------------------------------------------
    # Weak problem
    # ----------------------------------------------------------------

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

    a_form = (
        epsilon
        * ufl.inner(
            ufl.grad(u),
            ufl.grad(v)
        )
        * dx
    )

    L_form = (
        zero
        * v
        * dx
    )

    options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 3000,
        "ksp_error_if_not_converged": True
    }

    try:

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=options,
            petsc_options_prefix="sige_afm_"
        )

    except TypeError:

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=options
        )

    root_print(
        domain.comm,
        ""
    )

    root_print(
        domain.comm,
        "Solving div(eps grad(phi)) = 0 ..."
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
# Probe analysis
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

    if len(values):

        local_min = np.min(values)
        local_max = np.max(values)

        local_sum = np.sum(values)
        local_n = len(values)

    else:

        local_min = np.inf
        local_max = -np.inf

        local_sum = 0.0
        local_n = 0

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

    global_n = domain.comm.allreduce(
        local_n,
        op=MPI.SUM
    )

    mean_nodal = (
        global_sum / global_n
        if global_n > 0
        else np.nan
    )

    return {
        "name": name,
        "facets": int(global_facets),
        "dofs": int(global_n),
        "min": float(global_min),
        "max": float(global_max),
        "mean_nodal": float(mean_nodal)
    }


# =====================================================================
# Main
# =====================================================================

def main():

    a = parse_args()

    comm = MPI.COMM_WORLD

    if comm.rank == 0:

        os.makedirs(
            a.output,
            exist_ok=True
        )

    comm.barrier()

    root_print(comm, "")
    root_print(comm, "=" * 78)

    root_print(
        comm,
        "AFM-TIP DRIVEN Si/SiGe ELECTROSTATICS"
    )

    root_print(comm, "=" * 78)

    root_print(
        comm,
        f"Domain width       : "
        f"{a.lx:.1f} x {a.ly:.1f} nm"
    )

    root_print(
        comm,
        f"Air height         : "
        f"{a.air_height:.1f} nm"
    )

    root_print(
        comm,
        f"AFM tip voltage    : "
        f"{a.tip_voltage:.6f} V"
    )

    root_print(
        comm,
        f"Bottom/back gate   : "
        f"{a.bottom_voltage:.6f} V"
    )

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "Gate/probe 1       : "
        "NO imposed voltage"
    )

    root_print(
        comm,
        "Gate/probe 2       : "
        "NO imposed voltage"
    )

    root_print(comm, "")
    root_print(comm, "Stack:")

    root_print(
        comm,
        "  0 ->    2 nm : Si"
    )

    root_print(
        comm,
        "  2 ->   30 nm : SiGe"
    )

    root_print(
        comm,
        " 30 ->   40 nm : Si"
    )

    root_print(
        comm,
        "              Probe 1 at z=35 nm"
    )

    root_print(
        comm,
        " 40 ->   43 nm : SiGe"
    )

    root_print(
        comm,
        " 43 ->   53 nm : Si"
    )

    root_print(
        comm,
        "              Probe 2 at z=48 nm"
    )

    root_print(
        comm,
        " 53 -> 2053 nm : SiGe buffer"
    )

    # ----------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------

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
            "Cell tags did not survive "
            "Gmsh -> DOLFINx conversion."
        )

    if facet_tags is None:

        raise RuntimeError(
            "Facet tags did not survive "
            "Gmsh -> DOLFINx conversion."
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

    # ----------------------------------------------------------------
    # Tag diagnostics
    # ----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "Facet tag integrity:")

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

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # Global potential statistics
    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # Probe statistics
    # ----------------------------------------------------------------

    probe1_results = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE1,
        "Probe 1"
    )

    probe2_results = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE2,
        "Probe 2"
    )

    # ----------------------------------------------------------------
    # Mesh statistics
    # ----------------------------------------------------------------

    global_cells = (
        domain.topology.index_map(
            tdim
        ).size_global
    )

    global_dofs = (
        V.dofmap.index_map.size_global
        * V.dofmap.index_map_bs
    )

    # ----------------------------------------------------------------
    # Output report
    # ----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "=" * 78)
    root_print(comm, "SOLVER RESULTS")
    root_print(comm, "=" * 78)

    root_print(
        comm,
        f"Cells              : "
        f"{global_cells:,}"
    )

    root_print(
        comm,
        f"Potential DOFs     : "
        f"{global_dofs:,}"
    )

    root_print(
        comm,
        f"phi min            : "
        f"{phi_min:.12e} V"
    )

    root_print(
        comm,
        f"phi max            : "
        f"{phi_max:.12e} V"
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

    root_print(comm, "")
    root_print(comm, "INDUCED PROBE POTENTIALS")
    root_print(comm, "-" * 78)

    for result, z in [
        (
            probe1_results,
            z_probe1
        ),
        (
            probe2_results,
            z_probe2
        )
    ]:

        root_print(
            comm,
            f"{result['name']} at z={z:.1f} nm"
        )

        root_print(
            comm,
            f"  facets           : "
            f"{result['facets']:,}"
        )

        root_print(
            comm,
            f"  probe DOFs       : "
            f"{result['dofs']:,}"
        )

        root_print(
            comm,
            f"  minimum phi      : "
            f"{result['min']:.12e} V"
        )

        root_print(
            comm,
            f"  maximum phi      : "
            f"{result['max']:.12e} V"
        )

        root_print(
            comm,
            f"  mean nodal phi   : "
            f"{result['mean_nodal']:.12e} V"
        )

    # ================================================================
    # XDMF + H5 outputs
    # ================================================================

    potential_path = os.path.join(
        a.output,
        "potential.xdmf"
    )

    materials_path = os.path.join(
        a.output,
        "materials.xdmf"
    )

    tags_path = os.path.join(
        a.output,
        "facet_tags.xdmf"
    )

    with io.XDMFFile(
        comm,
        potential_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        xdmf.write_function(
            phi
        )

    with io.XDMFFile(
        comm,
        materials_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        xdmf.write_function(
            epsilon
        )

        xdmf.write_function(
            material_id
        )

    facet_tags.name = "facet_tags"

    with io.XDMFFile(
        comm,
        tags_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        try:

            xdmf.write_meshtags(
                facet_tags,
                domain.geometry
            )

        except TypeError:

            xdmf.write_meshtags(
                facet_tags
            )

    # ----------------------------------------------------------------
    # JSON results
    # ----------------------------------------------------------------

    if comm.rank == 0:

        results = {
            "geometry_nm": {
                "lx": a.lx,
                "ly": a.ly,
                "air_height": a.air_height,
                "bottom_z": z_bottom,
                "probe1_z": z_probe1,
                "probe2_z": z_probe2,
                "probe1_radius": a.probe1_radius,
                "probe2_radius": a.probe2_radius,
                "tip_gap": a.gap,
                "tip_radius": a.tip_radius,
                "cone_height": a.cone_height,
                "shank_radius": a.shank_radius
            },

            "voltages_V": {
                "AFM_tip": a.tip_voltage,
                "bottom_back_gate": a.bottom_voltage,
                "probe1": None,
                "probe2": None
            },

            "potential_V": {
                "global_min": float(phi_min),
                "global_max": float(phi_max),
                "probe1": probe1_results,
                "probe2": probe2_results
            },

            "mesh": {
                "cells": int(global_cells),
                "potential_dofs": int(global_dofs),
                "h_apex_nm": a.h_apex,
                "h_device_nm": a.h_device,
                "h_near_nm": a.h_near,
                "h_bottom_nm": a.h_bottom,
                "degree": a.degree
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

    root_print(comm, "")
    root_print(comm, "Wrote:")

    root_print(
        comm,
        f"  {a.output}/potential.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/potential.h5"
    )

    root_print(
        comm,
        f"  {a.output}/materials.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/materials.h5"
    )

    root_print(
        comm,
        f"  {a.output}/facet_tags.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/facet_tags.h5"
    )

    root_print(
        comm,
        f"  {a.output}/run_results.json"
    )

    root_print(comm, "")
    root_print(
        comm,
        "Done."
    )


if __name__ == "__main__":
    main()
PYcat > sige_afm_tip_two_probes.py <<'PY'
#!/usr/bin/env python3

"""
AFM-tip driven Si/SiGe electrostatics.

Coordinate convention
---------------------
z = 0 is the top semiconductor surface.
Air is at z < 0.
The semiconductor stack extends toward positive z.

AFM tip
-------
Conductive AFM tip in air.
Tip voltage = +1 V by default.

The tip consists of:
    spherical apex
    conical shank

The lowest point of the tip is --gap nm above z=0.

Semiconductor stack
-------------------
Layer 1:
    Si
    z = 0 to 2 nm
    eps_r = 11.7

Layer 2:
    SiGe
    z = 2 to 30 nm
    eps_r = 12.0

Layer 3:
    Si
    z = 30 to 40 nm
    eps_r = 11.7

    Probe disk 1:
        z = 35 nm
        radius = 10 nm

Layer 4:
    SiGe
    z = 40 to 43 nm
    eps_r = 12.0

Layer 5:
    Si
    z = 43 to 53 nm
    eps_r = 11.7

    Probe disk 2:
        z = 48 nm
        radius = 10 nm

Layer 6:
    SiGe buffer
    z = 53 to 2053 nm
    eps_r = 12.0

Bottom/back gate:
    z = 2053 nm
    phi = 0 V

IMPORTANT
---------
Probe 1 and Probe 2 DO NOT receive Dirichlet boundary conditions.

They are passive internal surfaces used to inspect the potential induced
by the AFM tip.

Equation:
    div(eps_r grad(phi)) = 0

rho = 0.

Coordinates are in nm.
Potential is in V.
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import json
import os

import gmsh
import numpy as np
import ufl

from dolfinx import fem, io


# =====================================================================
# DOLFINx compatibility
# =====================================================================

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

    # ---------------------------------------------------------------
    # Lateral domain
    # ---------------------------------------------------------------

    p.add_argument(
        "--lx",
        type=float,
        default=300.0,
        help="Domain width x [nm]"
    )

    p.add_argument(
        "--ly",
        type=float,
        default=300.0,
        help="Domain width y [nm]"
    )

    # ---------------------------------------------------------------
    # Air
    # ---------------------------------------------------------------

    p.add_argument(
        "--air-height",
        type=float,
        default=200.0,
        help="Air region height above sample [nm]"
    )

    p.add_argument(
        "--eps-air",
        type=float,
        default=1.0
    )

    # ---------------------------------------------------------------
    # Semiconductor permittivity
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # AFM geometry
    # ---------------------------------------------------------------

    p.add_argument(
        "--gap",
        type=float,
        default=10.0,
        help="Tip-surface gap [nm]"
    )

    p.add_argument(
        "--tip-radius",
        type=float,
        default=20.0,
        help="Spherical tip radius [nm]"
    )

    p.add_argument(
        "--cone-height",
        type=float,
        default=100.0,
        help="Cone height [nm]"
    )

    p.add_argument(
        "--shank-radius",
        type=float,
        default=60.0,
        help="Cone radius at upper end [nm]"
    )

    # ---------------------------------------------------------------
    # Voltages
    # ---------------------------------------------------------------

    p.add_argument(
        "--tip-voltage",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--bottom-voltage",
        type=float,
        default=0.0
    )

    # ---------------------------------------------------------------
    # Probe geometry
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Mesh
    # ---------------------------------------------------------------

    p.add_argument(
        "--h-apex",
        type=float,
        default=1.0,
        help="Mesh size near AFM apex [nm]"
    )

    p.add_argument(
        "--h-device",
        type=float,
        default=2.0,
        help="Mesh size in central device region [nm]"
    )

    p.add_argument(
        "--h-near",
        type=float,
        default=5.0,
        help="Mesh size in near field [nm]"
    )

    p.add_argument(
        "--h-bottom",
        type=float,
        default=100.0,
        help="Mesh size deep in buffer [nm]"
    )

    p.add_argument(
        "--degree",
        type=int,
        default=1
    )

    p.add_argument(
        "--output",
        type=str,
        default="results/sige_afm_tip"
    )

    return p.parse_args()


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
            f"Physical group {name} is empty."
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

    # ----------------------------------------------------------------
    # Fixed semiconductor stack
    # ----------------------------------------------------------------

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

        # ============================================================
        # Semiconductor volumes
        # ============================================================

        v_si_top = occ.addBox(
            xmin,
            ymin,
            z0,
            a.lx,
            a.ly,
            z1 - z0
        )

        v_sige28 = occ.addBox(
            xmin,
            ymin,
            z1,
            a.lx,
            a.ly,
            z2 - z1
        )

        v_si_probe1 = occ.addBox(
            xmin,
            ymin,
            z2,
            a.lx,
            a.ly,
            z3 - z2
        )

        v_sige3 = occ.addBox(
            xmin,
            ymin,
            z3,
            a.lx,
            a.ly,
            z4 - z3
        )

        v_si_probe2 = occ.addBox(
            xmin,
            ymin,
            z4,
            a.lx,
            a.ly,
            z5 - z4
        )

        v_buffer = occ.addBox(
            xmin,
            ymin,
            z5,
            a.lx,
            a.ly,
            z6 - z5
        )

        # ============================================================
        # Air box
        #
        # Air extends from -air_height to z=0.
        # ============================================================

        air = occ.addBox(
            xmin,
            ymin,
            -a.air_height,
            a.lx,
            a.ly,
            a.air_height
        )

        # ============================================================
        # AFM tip
        #
        # Lowest point:
        #
        #     z = -gap
        #
        # Sphere center:
        #
        #     z = -(gap + radius)
        #
        # Cone extends upward toward more-negative z.
        # ============================================================

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

        tip_objects, _ = occ.fuse(
            [(3, sphere)],
            [(3, cone)],
            removeObject=True,
            removeTool=True
        )

        # Remove conducting tip volume from air.
        air_cut, _ = occ.cut(
            [(3, air)],
            tip_objects,
            removeObject=True,
            removeTool=True
        )

        # ============================================================
        # Fragment all dielectric volumes together
        # ============================================================

        objects = [
            (3, v_si_top)
        ]

        tools = [
            (3, v_sige28),
            (3, v_si_probe1),
            (3, v_sige3),
            (3, v_si_probe2),
            (3, v_buffer)
        ]

        tools.extend(
            air_cut
        )

        occ.fragment(
            objects,
            tools,
            removeObject=True,
            removeTool=True
        )

        occ.synchronize()

        # ============================================================
        # Classify volumes by center-of-mass z
        # ============================================================

        air_vols = []
        si_top_vols = []
        sige28_vols = []
        si_probe1_vols = []
        sige3_vols = []
        si_probe2_vols = []
        buffer_vols = []

        for dim, tag in gmsh.model.getEntities(3):

            _, _, cz = occ.getCenterOfMass(
                dim,
                tag
            )

            if cz < 0.0:

                air_vols.append(tag)

            elif cz < z1:

                si_top_vols.append(tag)

            elif cz < z2:

                sige28_vols.append(tag)

            elif cz < z3:

                si_probe1_vols.append(tag)

            elif cz < z4:

                sige3_vols.append(tag)

            elif cz < z5:

                si_probe2_vols.append(tag)

            else:

                buffer_vols.append(tag)

        volume_groups = [
            (
                MAT_AIR,
                air_vols,
                "air"
            ),
            (
                MAT_SI_TOP,
                si_top_vols,
                "Si_2nm"
            ),
            (
                MAT_SIGE_28,
                sige28_vols,
                "SiGe_28nm"
            ),
            (
                MAT_SI_PROBE1,
                si_probe1_vols,
                "Si_10nm_upper"
            ),
            (
                MAT_SIGE_3,
                sige3_vols,
                "SiGe_3nm"
            ),
            (
                MAT_SI_PROBE2,
                si_probe2_vols,
                "Si_10nm_lower"
            ),
            (
                MAT_SIGE_BUFFER,
                buffer_vols,
                "SiGe_buffer_2000nm"
            )
        ]

        for tag, entities, name in volume_groups:

            physical(
                3,
                entities,
                tag,
                name
            )

        # ============================================================
        # Passive circular probe surfaces
        # ============================================================

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

        # Embed into appropriate Si volumes.
        for volume in si_probe1_vols:

            gmsh.model.mesh.embed(
                2,
                [probe1],
                3,
                volume
            )

        for volume in si_probe2_vols:

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

        # ============================================================
        # Find AFM-tip cavity facets
        # ============================================================

        tip_surfaces = []

        # Air should normally be one volume.
        for air_tag in air_vols:

            boundary = gmsh.model.getBoundary(
                [(3, air_tag)],
                oriented=False,
                recursive=False
            )

            for dim, surface_tag in boundary:

                if dim != 2:
                    continue

                bbox = gmsh.model.getBoundingBox(
                    2,
                    surface_tag
                )

                sxmin, symin, szmin, sxmax, symax, szmax = bbox

                tol = 1.0e-6

                on_box = (
                    abs(sxmin - xmin) < tol
                    and abs(sxmax - xmin) < tol
                ) or (
                    abs(sxmin - xmax) < tol
                    and abs(sxmax - xmax) < tol
                ) or (
                    abs(symin - ymin) < tol
                    and abs(symax - ymin) < tol
                ) or (
                    abs(symin - ymax) < tol
                    and abs(symax - ymax) < tol
                ) or (
                    abs(szmin + a.air_height) < tol
                    and abs(szmax + a.air_height) < tol
                ) or (
                    abs(szmin - 0.0) < tol
                    and abs(szmax - 0.0) < tol
                )

                if not on_box:

                    tip_surfaces.append(
                        surface_tag
                    )

        tip_surfaces = sorted(
            set(tip_surfaces)
        )

        physical(
            2,
            tip_surfaces,
            FACET_TIP,
            "AFM_tip_1V"
        )

        # ============================================================
        # Bottom boundary
        # ============================================================

        bottom_surfaces = []

        outer_surfaces = []

        for dim, tag in gmsh.model.getEntities(2):

            if tag in [
                probe1,
                probe2
            ]:
                continue

            bbox = gmsh.model.getBoundingBox(
                dim,
                tag
            )

            sxmin, symin, szmin, sxmax, symax, szmax = bbox

            tol = 1.0e-6

            if (
                abs(szmin - z6) < tol
                and
                abs(szmax - z6) < tol
            ):

                bottom_surfaces.append(
                    tag
                )

            elif (
                (
                    abs(sxmin - xmin) < tol
                    and
                    abs(sxmax - xmin) < tol
                )
                or
                (
                    abs(sxmin - xmax) < tol
                    and
                    abs(sxmax - xmax) < tol
                )
                or
                (
                    abs(symin - ymin) < tol
                    and
                    abs(symax - ymin) < tol
                )
                or
                (
                    abs(symin - ymax) < tol
                    and
                    abs(symax - ymax) < tol
                )
                or
                (
                    abs(szmin + a.air_height) < tol
                    and
                    abs(szmax + a.air_height) < tol
                )
            ):

                outer_surfaces.append(
                    tag
                )

        physical(
            2,
            bottom_surfaces,
            FACET_BOTTOM,
            "bottom_back_gate_0V"
        )

        physical(
            2,
            sorted(set(outer_surfaces)),
            FACET_OUTER,
            "outer"
        )

        # ============================================================
        # Mesh refinement
        #
        # 1. Fine near AFM tip.
        # 2. Fine in central active semiconductor region.
        # 3. Coarse deep in 2000 nm buffer.
        # ============================================================

        # Distance from tip.
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

        tip_threshold = gmsh.model.mesh.field.add(
            "Threshold"
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "InField",
            distance_tip
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "SizeMin",
            a.h_apex
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "SizeMax",
            a.h_near
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "DistMin",
            10.0
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "DistMax",
            100.0
        )

        # Central device refinement.
        central = gmsh.model.mesh.field.add(
            "Box"
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "VIn",
            a.h_device
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "VOut",
            a.h_bottom
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "XMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "XMax",
            +60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "YMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "YMax",
            +60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "ZMin",
            -30.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "ZMax",
            80.0
        )

        # Depth grading.
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
                tip_threshold,
                central,
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
            "Generating AFM + Si/SiGe tetrahedral mesh..."
        )

        gmsh.model.mesh.generate(3)

    # ----------------------------------------------------------------
    # Transfer to DOLFINx
    # ----------------------------------------------------------------

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
# Material field
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

    epsilon.name = (
        "relative_permittivity"
    )

    material_id = fem.Function(Q)

    material_id.name = (
        "material_id"
    )

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

    for marker, eps_value in materials.items():

        cells = cell_tags.find(
            marker
        )

        count = domain.comm.allreduce(
            len(cells),
            op=MPI.SUM
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

        epsilon.x.array[dofs] = (
            eps_value
        )

        material_id.x.array[dofs] = (
            float(marker)
        )

    epsilon.x.scatter_forward()
    material_id.x.scatter_forward()

    return epsilon, material_id


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

    epsilon, material_id = (
        make_material_fields(
            domain,
            cell_tags,
            a
        )
    )

    # ----------------------------------------------------------------
    # BC helper
    # ----------------------------------------------------------------

    def make_bc(marker, voltage):

        facets = facet_tags.find(
            marker
        )

        global_facets = (
            domain.comm.allreduce(
                len(facets),
                op=MPI.SUM
            )
        )

        if global_facets == 0:

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

        return (
            fem.dirichletbc(
                value,
                dofs,
                V
            ),
            facets,
            dofs
        )

    # ================================================================
    # ONLY TWO ELECTRICAL DIRICHLET CONDITIONS
    #
    # 1. AFM tip = 1 V
    # 2. Bottom/back gate = 0 V
    #
    # Probe disks receive NO BC.
    # ================================================================

    bc_tip, tip_facets, tip_dofs = make_bc(
        FACET_TIP,
        a.tip_voltage
    )

    bc_bottom, bottom_facets, bottom_dofs = make_bc(
        FACET_BOTTOM,
        a.bottom_voltage
    )

    bcs = [
        bc_tip,
        bc_bottom
    ]

    # ----------------------------------------------------------------
    # Weak problem
    # ----------------------------------------------------------------

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

    a_form = (
        epsilon
        * ufl.inner(
            ufl.grad(u),
            ufl.grad(v)
        )
        * dx
    )

    L_form = (
        zero
        * v
        * dx
    )

    options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 3000,
        "ksp_error_if_not_converged": True
    }

    try:

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=options,
            petsc_options_prefix="sige_afm_"
        )

    except TypeError:

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=options
        )

    root_print(
        domain.comm,
        ""
    )

    root_print(
        domain.comm,
        "Solving div(eps grad(phi)) = 0 ..."
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
# Probe analysis
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

    if len(values):

        local_min = np.min(values)
        local_max = np.max(values)

        local_sum = np.sum(values)
        local_n = len(values)

    else:

        local_min = np.inf
        local_max = -np.inf

        local_sum = 0.0
        local_n = 0

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

    global_n = domain.comm.allreduce(
        local_n,
        op=MPI.SUM
    )

    mean_nodal = (
        global_sum / global_n
        if global_n > 0
        else np.nan
    )

    return {
        "name": name,
        "facets": int(global_facets),
        "dofs": int(global_n),
        "min": float(global_min),
        "max": float(global_max),
        "mean_nodal": float(mean_nodal)
    }


# =====================================================================
# Main
# =====================================================================

def main():

    a = parse_args()

    comm = MPI.COMM_WORLD

    if comm.rank == 0:

        os.makedirs(
            a.output,
            exist_ok=True
        )

    comm.barrier()

    root_print(comm, "")
    root_print(comm, "=" * 78)

    root_print(
        comm,
        "AFM-TIP DRIVEN Si/SiGe ELECTROSTATICS"
    )

    root_print(comm, "=" * 78)

    root_print(
        comm,
        f"Domain width       : "
        f"{a.lx:.1f} x {a.ly:.1f} nm"
    )

    root_print(
        comm,
        f"Air height         : "
        f"{a.air_height:.1f} nm"
    )

    root_print(
        comm,
        f"AFM tip voltage    : "
        f"{a.tip_voltage:.6f} V"
    )

    root_print(
        comm,
        f"Bottom/back gate   : "
        f"{a.bottom_voltage:.6f} V"
    )

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "Gate/probe 1       : "
        "NO imposed voltage"
    )

    root_print(
        comm,
        "Gate/probe 2       : "
        "NO imposed voltage"
    )

    root_print(comm, "")
    root_print(comm, "Stack:")

    root_print(
        comm,
        "  0 ->    2 nm : Si"
    )

    root_print(
        comm,
        "  2 ->   30 nm : SiGe"
    )

    root_print(
        comm,
        " 30 ->   40 nm : Si"
    )

    root_print(
        comm,
        "              Probe 1 at z=35 nm"
    )

    root_print(
        comm,
        " 40 ->   43 nm : SiGe"
    )

    root_print(
        comm,
        " 43 ->   53 nm : Si"
    )

    root_print(
        comm,
        "              Probe 2 at z=48 nm"
    )

    root_print(
        comm,
        " 53 -> 2053 nm : SiGe buffer"
    )

    # ----------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------

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
            "Cell tags did not survive "
            "Gmsh -> DOLFINx conversion."
        )

    if facet_tags is None:

        raise RuntimeError(
            "Facet tags did not survive "
            "Gmsh -> DOLFINx conversion."
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

    # ----------------------------------------------------------------
    # Tag diagnostics
    # ----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "Facet tag integrity:")

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

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # Global potential statistics
    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # Probe statistics
    # ----------------------------------------------------------------

    probe1_results = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE1,
        "Probe 1"
    )

    probe2_results = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE2,
        "Probe 2"
    )

    # ----------------------------------------------------------------
    # Mesh statistics
    # ----------------------------------------------------------------

    global_cells = (
        domain.topology.index_map(
            tdim
        ).size_global
    )

    global_dofs = (
        V.dofmap.index_map.size_global
        * V.dofmap.index_map_bs
    )

    # ----------------------------------------------------------------
    # Output report
    # ----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "=" * 78)
    root_print(comm, "SOLVER RESULTS")
    root_print(comm, "=" * 78)

    root_print(
        comm,
        f"Cells              : "
        f"{global_cells:,}"
    )

    root_print(
        comm,
        f"Potential DOFs     : "
        f"{global_dofs:,}"
    )

    root_print(
        comm,
        f"phi min            : "
        f"{phi_min:.12e} V"
    )

    root_print(
        comm,
        f"phi max            : "
        f"{phi_max:.12e} V"
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

    root_print(comm, "")
    root_print(comm, "INDUCED PROBE POTENTIALS")
    root_print(comm, "-" * 78)

    for result, z in [
        (
            probe1_results,
            z_probe1
        ),
        (
            probe2_results,
            z_probe2
        )
    ]:

        root_print(
            comm,
            f"{result['name']} at z={z:.1f} nm"
        )

        root_print(
            comm,
            f"  facets           : "
            f"{result['facets']:,}"
        )

        root_print(
            comm,
            f"  probe DOFs       : "
            f"{result['dofs']:,}"
        )

        root_print(
            comm,
            f"  minimum phi      : "
            f"{result['min']:.12e} V"
        )

        root_print(
            comm,
            f"  maximum phi      : "
            f"{result['max']:.12e} V"
        )

        root_print(
            comm,
            f"  mean nodal phi   : "
            f"{result['mean_nodal']:.12e} V"
        )

    # ================================================================
    # XDMF + H5 outputs
    # ================================================================

    potential_path = os.path.join(
        a.output,
        "potential.xdmf"
    )

    materials_path = os.path.join(
        a.output,
        "materials.xdmf"
    )

    tags_path = os.path.join(
        a.output,
        "facet_tags.xdmf"
    )

    with io.XDMFFile(
        comm,
        potential_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        xdmf.write_function(
            phi
        )

    with io.XDMFFile(
        comm,
        materials_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        xdmf.write_function(
            epsilon
        )

        xdmf.write_function(
            material_id
        )

    facet_tags.name = "facet_tags"

    with io.XDMFFile(
        comm,
        tags_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        try:

            xdmf.write_meshtags(
                facet_tags,
                domain.geometry
            )

        except TypeError:

            xdmf.write_meshtags(
                facet_tags
            )

    # ----------------------------------------------------------------
    # JSON results
    # ----------------------------------------------------------------

    if comm.rank == 0:

        results = {
            "geometry_nm": {
                "lx": a.lx,
                "ly": a.ly,
                "air_height": a.air_height,
                "bottom_z": z_bottom,
                "probe1_z": z_probe1,
                "probe2_z": z_probe2,
                "probe1_radius": a.probe1_radius,
                "probe2_radius": a.probe2_radius,
                "tip_gap": a.gap,
                "tip_radius": a.tip_radius,
                "cone_height": a.cone_height,
                "shank_radius": a.shank_radius
            },

            "voltages_V": {
                "AFM_tip": a.tip_voltage,
                "bottom_back_gate": a.bottom_voltage,
                "probe1": None,
                "probe2": None
            },

            "potential_V": {
                "global_min": float(phi_min),
                "global_max": float(phi_max),
                "probe1": probe1_results,
                "probe2": probe2_results
            },

            "mesh": {
                "cells": int(global_cells),
                "potential_dofs": int(global_dofs),
                "h_apex_nm": a.h_apex,
                "h_device_nm": a.h_device,
                "h_near_nm": a.h_near,
                "h_bottom_nm": a.h_bottom,
                "degree": a.degree
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

    root_print(comm, "")
    root_print(comm, "Wrote:")

    root_print(
        comm,
        f"  {a.output}/potential.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/potential.h5"
    )

    root_print(
        comm,
        f"  {a.output}/materials.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/materials.h5"
    )

    root_print(
        comm,
        f"  {a.output}/facet_tags.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/facet_tags.h5"
    )

    root_print(
        comm,
        f"  {a.output}/run_results.json"
    )

    root_print(comm, "")
    root_print(
        comm,
        "Done."
    )


if __name__ == "__main__":
    main()
PYcat > sige_afm_tip_two_probes.py <<'PY'
#!/usr/bin/env python3

"""
AFM-tip driven Si/SiGe electrostatics.

Coordinate convention
---------------------
z = 0 is the top semiconductor surface.
Air is at z < 0.
The semiconductor stack extends toward positive z.

AFM tip
-------
Conductive AFM tip in air.
Tip voltage = +1 V by default.

The tip consists of:
    spherical apex
    conical shank

The lowest point of the tip is --gap nm above z=0.

Semiconductor stack
-------------------
Layer 1:
    Si
    z = 0 to 2 nm
    eps_r = 11.7

Layer 2:
    SiGe
    z = 2 to 30 nm
    eps_r = 12.0

Layer 3:
    Si
    z = 30 to 40 nm
    eps_r = 11.7

    Probe disk 1:
        z = 35 nm
        radius = 10 nm

Layer 4:
    SiGe
    z = 40 to 43 nm
    eps_r = 12.0

Layer 5:
    Si
    z = 43 to 53 nm
    eps_r = 11.7

    Probe disk 2:
        z = 48 nm
        radius = 10 nm

Layer 6:
    SiGe buffer
    z = 53 to 2053 nm
    eps_r = 12.0

Bottom/back gate:
    z = 2053 nm
    phi = 0 V

IMPORTANT
---------
Probe 1 and Probe 2 DO NOT receive Dirichlet boundary conditions.

They are passive internal surfaces used to inspect the potential induced
by the AFM tip.

Equation:
    div(eps_r grad(phi)) = 0

rho = 0.

Coordinates are in nm.
Potential is in V.
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import json
import os

import gmsh
import numpy as np
import ufl

from dolfinx import fem, io


# =====================================================================
# DOLFINx compatibility
# =====================================================================

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

    # ---------------------------------------------------------------
    # Lateral domain
    # ---------------------------------------------------------------

    p.add_argument(
        "--lx",
        type=float,
        default=300.0,
        help="Domain width x [nm]"
    )

    p.add_argument(
        "--ly",
        type=float,
        default=300.0,
        help="Domain width y [nm]"
    )

    # ---------------------------------------------------------------
    # Air
    # ---------------------------------------------------------------

    p.add_argument(
        "--air-height",
        type=float,
        default=200.0,
        help="Air region height above sample [nm]"
    )

    p.add_argument(
        "--eps-air",
        type=float,
        default=1.0
    )

    # ---------------------------------------------------------------
    # Semiconductor permittivity
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # AFM geometry
    # ---------------------------------------------------------------

    p.add_argument(
        "--gap",
        type=float,
        default=10.0,
        help="Tip-surface gap [nm]"
    )

    p.add_argument(
        "--tip-radius",
        type=float,
        default=20.0,
        help="Spherical tip radius [nm]"
    )

    p.add_argument(
        "--cone-height",
        type=float,
        default=100.0,
        help="Cone height [nm]"
    )

    p.add_argument(
        "--shank-radius",
        type=float,
        default=60.0,
        help="Cone radius at upper end [nm]"
    )

    # ---------------------------------------------------------------
    # Voltages
    # ---------------------------------------------------------------

    p.add_argument(
        "--tip-voltage",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--bottom-voltage",
        type=float,
        default=0.0
    )

    # ---------------------------------------------------------------
    # Probe geometry
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Mesh
    # ---------------------------------------------------------------

    p.add_argument(
        "--h-apex",
        type=float,
        default=1.0,
        help="Mesh size near AFM apex [nm]"
    )

    p.add_argument(
        "--h-device",
        type=float,
        default=2.0,
        help="Mesh size in central device region [nm]"
    )

    p.add_argument(
        "--h-near",
        type=float,
        default=5.0,
        help="Mesh size in near field [nm]"
    )

    p.add_argument(
        "--h-bottom",
        type=float,
        default=100.0,
        help="Mesh size deep in buffer [nm]"
    )

    p.add_argument(
        "--degree",
        type=int,
        default=1
    )

    p.add_argument(
        "--output",
        type=str,
        default="results/sige_afm_tip"
    )

    return p.parse_args()


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
            f"Physical group {name} is empty."
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

    # ----------------------------------------------------------------
    # Fixed semiconductor stack
    # ----------------------------------------------------------------

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

        # ============================================================
        # Semiconductor volumes
        # ============================================================

        v_si_top = occ.addBox(
            xmin,
            ymin,
            z0,
            a.lx,
            a.ly,
            z1 - z0
        )

        v_sige28 = occ.addBox(
            xmin,
            ymin,
            z1,
            a.lx,
            a.ly,
            z2 - z1
        )

        v_si_probe1 = occ.addBox(
            xmin,
            ymin,
            z2,
            a.lx,
            a.ly,
            z3 - z2
        )

        v_sige3 = occ.addBox(
            xmin,
            ymin,
            z3,
            a.lx,
            a.ly,
            z4 - z3
        )

        v_si_probe2 = occ.addBox(
            xmin,
            ymin,
            z4,
            a.lx,
            a.ly,
            z5 - z4
        )

        v_buffer = occ.addBox(
            xmin,
            ymin,
            z5,
            a.lx,
            a.ly,
            z6 - z5
        )

        # ============================================================
        # Air box
        #
        # Air extends from -air_height to z=0.
        # ============================================================

        air = occ.addBox(
            xmin,
            ymin,
            -a.air_height,
            a.lx,
            a.ly,
            a.air_height
        )

        # ============================================================
        # AFM tip
        #
        # Lowest point:
        #
        #     z = -gap
        #
        # Sphere center:
        #
        #     z = -(gap + radius)
        #
        # Cone extends upward toward more-negative z.
        # ============================================================

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

        tip_objects, _ = occ.fuse(
            [(3, sphere)],
            [(3, cone)],
            removeObject=True,
            removeTool=True
        )

        # Remove conducting tip volume from air.
        air_cut, _ = occ.cut(
            [(3, air)],
            tip_objects,
            removeObject=True,
            removeTool=True
        )

        # ============================================================
        # Fragment all dielectric volumes together
        # ============================================================

        objects = [
            (3, v_si_top)
        ]

        tools = [
            (3, v_sige28),
            (3, v_si_probe1),
            (3, v_sige3),
            (3, v_si_probe2),
            (3, v_buffer)
        ]

        tools.extend(
            air_cut
        )

        occ.fragment(
            objects,
            tools,
            removeObject=True,
            removeTool=True
        )

        occ.synchronize()

        # ============================================================
        # Classify volumes by center-of-mass z
        # ============================================================

        air_vols = []
        si_top_vols = []
        sige28_vols = []
        si_probe1_vols = []
        sige3_vols = []
        si_probe2_vols = []
        buffer_vols = []

        for dim, tag in gmsh.model.getEntities(3):

            _, _, cz = occ.getCenterOfMass(
                dim,
                tag
            )

            if cz < 0.0:

                air_vols.append(tag)

            elif cz < z1:

                si_top_vols.append(tag)

            elif cz < z2:

                sige28_vols.append(tag)

            elif cz < z3:

                si_probe1_vols.append(tag)

            elif cz < z4:

                sige3_vols.append(tag)

            elif cz < z5:

                si_probe2_vols.append(tag)

            else:

                buffer_vols.append(tag)

        volume_groups = [
            (
                MAT_AIR,
                air_vols,
                "air"
            ),
            (
                MAT_SI_TOP,
                si_top_vols,
                "Si_2nm"
            ),
            (
                MAT_SIGE_28,
                sige28_vols,
                "SiGe_28nm"
            ),
            (
                MAT_SI_PROBE1,
                si_probe1_vols,
                "Si_10nm_upper"
            ),
            (
                MAT_SIGE_3,
                sige3_vols,
                "SiGe_3nm"
            ),
            (
                MAT_SI_PROBE2,
                si_probe2_vols,
                "Si_10nm_lower"
            ),
            (
                MAT_SIGE_BUFFER,
                buffer_vols,
                "SiGe_buffer_2000nm"
            )
        ]

        for tag, entities, name in volume_groups:

            physical(
                3,
                entities,
                tag,
                name
            )

        # ============================================================
        # Passive circular probe surfaces
        # ============================================================

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

        # Embed into appropriate Si volumes.
        for volume in si_probe1_vols:

            gmsh.model.mesh.embed(
                2,
                [probe1],
                3,
                volume
            )

        for volume in si_probe2_vols:

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

        # ============================================================
        # Find AFM-tip cavity facets
        # ============================================================

        tip_surfaces = []

        # Air should normally be one volume.
        for air_tag in air_vols:

            boundary = gmsh.model.getBoundary(
                [(3, air_tag)],
                oriented=False,
                recursive=False
            )

            for dim, surface_tag in boundary:

                if dim != 2:
                    continue

                bbox = gmsh.model.getBoundingBox(
                    2,
                    surface_tag
                )

                sxmin, symin, szmin, sxmax, symax, szmax = bbox

                tol = 1.0e-6

                on_box = (
                    abs(sxmin - xmin) < tol
                    and abs(sxmax - xmin) < tol
                ) or (
                    abs(sxmin - xmax) < tol
                    and abs(sxmax - xmax) < tol
                ) or (
                    abs(symin - ymin) < tol
                    and abs(symax - ymin) < tol
                ) or (
                    abs(symin - ymax) < tol
                    and abs(symax - ymax) < tol
                ) or (
                    abs(szmin + a.air_height) < tol
                    and abs(szmax + a.air_height) < tol
                ) or (
                    abs(szmin - 0.0) < tol
                    and abs(szmax - 0.0) < tol
                )

                if not on_box:

                    tip_surfaces.append(
                        surface_tag
                    )

        tip_surfaces = sorted(
            set(tip_surfaces)
        )

        physical(
            2,
            tip_surfaces,
            FACET_TIP,
            "AFM_tip_1V"
        )

        # ============================================================
        # Bottom boundary
        # ============================================================

        bottom_surfaces = []

        outer_surfaces = []

        for dim, tag in gmsh.model.getEntities(2):

            if tag in [
                probe1,
                probe2
            ]:
                continue

            bbox = gmsh.model.getBoundingBox(
                dim,
                tag
            )

            sxmin, symin, szmin, sxmax, symax, szmax = bbox

            tol = 1.0e-6

            if (
                abs(szmin - z6) < tol
                and
                abs(szmax - z6) < tol
            ):

                bottom_surfaces.append(
                    tag
                )

            elif (
                (
                    abs(sxmin - xmin) < tol
                    and
                    abs(sxmax - xmin) < tol
                )
                or
                (
                    abs(sxmin - xmax) < tol
                    and
                    abs(sxmax - xmax) < tol
                )
                or
                (
                    abs(symin - ymin) < tol
                    and
                    abs(symax - ymin) < tol
                )
                or
                (
                    abs(symin - ymax) < tol
                    and
                    abs(symax - ymax) < tol
                )
                or
                (
                    abs(szmin + a.air_height) < tol
                    and
                    abs(szmax + a.air_height) < tol
                )
            ):

                outer_surfaces.append(
                    tag
                )

        physical(
            2,
            bottom_surfaces,
            FACET_BOTTOM,
            "bottom_back_gate_0V"
        )

        physical(
            2,
            sorted(set(outer_surfaces)),
            FACET_OUTER,
            "outer"
        )

        # ============================================================
        # Mesh refinement
        #
        # 1. Fine near AFM tip.
        # 2. Fine in central active semiconductor region.
        # 3. Coarse deep in 2000 nm buffer.
        # ============================================================

        # Distance from tip.
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

        tip_threshold = gmsh.model.mesh.field.add(
            "Threshold"
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "InField",
            distance_tip
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "SizeMin",
            a.h_apex
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "SizeMax",
            a.h_near
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "DistMin",
            10.0
        )

        gmsh.model.mesh.field.setNumber(
            tip_threshold,
            "DistMax",
            100.0
        )

        # Central device refinement.
        central = gmsh.model.mesh.field.add(
            "Box"
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "VIn",
            a.h_device
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "VOut",
            a.h_bottom
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "XMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "XMax",
            +60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "YMin",
            -60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "YMax",
            +60.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "ZMin",
            -30.0
        )

        gmsh.model.mesh.field.setNumber(
            central,
            "ZMax",
            80.0
        )

        # Depth grading.
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
                tip_threshold,
                central,
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
            "Generating AFM + Si/SiGe tetrahedral mesh..."
        )

        gmsh.model.mesh.generate(3)

    # ----------------------------------------------------------------
    # Transfer to DOLFINx
    # ----------------------------------------------------------------

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
# Material field
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

    epsilon.name = (
        "relative_permittivity"
    )

    material_id = fem.Function(Q)

    material_id.name = (
        "material_id"
    )

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

    for marker, eps_value in materials.items():

        cells = cell_tags.find(
            marker
        )

        count = domain.comm.allreduce(
            len(cells),
            op=MPI.SUM
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

        epsilon.x.array[dofs] = (
            eps_value
        )

        material_id.x.array[dofs] = (
            float(marker)
        )

    epsilon.x.scatter_forward()
    material_id.x.scatter_forward()

    return epsilon, material_id


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

    epsilon, material_id = (
        make_material_fields(
            domain,
            cell_tags,
            a
        )
    )

    # ----------------------------------------------------------------
    # BC helper
    # ----------------------------------------------------------------

    def make_bc(marker, voltage):

        facets = facet_tags.find(
            marker
        )

        global_facets = (
            domain.comm.allreduce(
                len(facets),
                op=MPI.SUM
            )
        )

        if global_facets == 0:

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

        return (
            fem.dirichletbc(
                value,
                dofs,
                V
            ),
            facets,
            dofs
        )

    # ================================================================
    # ONLY TWO ELECTRICAL DIRICHLET CONDITIONS
    #
    # 1. AFM tip = 1 V
    # 2. Bottom/back gate = 0 V
    #
    # Probe disks receive NO BC.
    # ================================================================

    bc_tip, tip_facets, tip_dofs = make_bc(
        FACET_TIP,
        a.tip_voltage
    )

    bc_bottom, bottom_facets, bottom_dofs = make_bc(
        FACET_BOTTOM,
        a.bottom_voltage
    )

    bcs = [
        bc_tip,
        bc_bottom
    ]

    # ----------------------------------------------------------------
    # Weak problem
    # ----------------------------------------------------------------

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

    a_form = (
        epsilon
        * ufl.inner(
            ufl.grad(u),
            ufl.grad(v)
        )
        * dx
    )

    L_form = (
        zero
        * v
        * dx
    )

    options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 3000,
        "ksp_error_if_not_converged": True
    }

    try:

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=options,
            petsc_options_prefix="sige_afm_"
        )

    except TypeError:

        problem = LinearProblem(
            a_form,
            L_form,
            bcs=bcs,
            petsc_options=options
        )

    root_print(
        domain.comm,
        ""
    )

    root_print(
        domain.comm,
        "Solving div(eps grad(phi)) = 0 ..."
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
# Probe analysis
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

    if len(values):

        local_min = np.min(values)
        local_max = np.max(values)

        local_sum = np.sum(values)
        local_n = len(values)

    else:

        local_min = np.inf
        local_max = -np.inf

        local_sum = 0.0
        local_n = 0

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

    global_n = domain.comm.allreduce(
        local_n,
        op=MPI.SUM
    )

    mean_nodal = (
        global_sum / global_n
        if global_n > 0
        else np.nan
    )

    return {
        "name": name,
        "facets": int(global_facets),
        "dofs": int(global_n),
        "min": float(global_min),
        "max": float(global_max),
        "mean_nodal": float(mean_nodal)
    }


# =====================================================================
# Main
# =====================================================================

def main():

    a = parse_args()

    comm = MPI.COMM_WORLD

    if comm.rank == 0:

        os.makedirs(
            a.output,
            exist_ok=True
        )

    comm.barrier()

    root_print(comm, "")
    root_print(comm, "=" * 78)

    root_print(
        comm,
        "AFM-TIP DRIVEN Si/SiGe ELECTROSTATICS"
    )

    root_print(comm, "=" * 78)

    root_print(
        comm,
        f"Domain width       : "
        f"{a.lx:.1f} x {a.ly:.1f} nm"
    )

    root_print(
        comm,
        f"Air height         : "
        f"{a.air_height:.1f} nm"
    )

    root_print(
        comm,
        f"AFM tip voltage    : "
        f"{a.tip_voltage:.6f} V"
    )

    root_print(
        comm,
        f"Bottom/back gate   : "
        f"{a.bottom_voltage:.6f} V"
    )

    root_print(
        comm,
        ""
    )

    root_print(
        comm,
        "Gate/probe 1       : "
        "NO imposed voltage"
    )

    root_print(
        comm,
        "Gate/probe 2       : "
        "NO imposed voltage"
    )

    root_print(comm, "")
    root_print(comm, "Stack:")

    root_print(
        comm,
        "  0 ->    2 nm : Si"
    )

    root_print(
        comm,
        "  2 ->   30 nm : SiGe"
    )

    root_print(
        comm,
        " 30 ->   40 nm : Si"
    )

    root_print(
        comm,
        "              Probe 1 at z=35 nm"
    )

    root_print(
        comm,
        " 40 ->   43 nm : SiGe"
    )

    root_print(
        comm,
        " 43 ->   53 nm : Si"
    )

    root_print(
        comm,
        "              Probe 2 at z=48 nm"
    )

    root_print(
        comm,
        " 53 -> 2053 nm : SiGe buffer"
    )

    # ----------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------

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
            "Cell tags did not survive "
            "Gmsh -> DOLFINx conversion."
        )

    if facet_tags is None:

        raise RuntimeError(
            "Facet tags did not survive "
            "Gmsh -> DOLFINx conversion."
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

    # ----------------------------------------------------------------
    # Tag diagnostics
    # ----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "Facet tag integrity:")

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

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # Global potential statistics
    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # Probe statistics
    # ----------------------------------------------------------------

    probe1_results = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE1,
        "Probe 1"
    )

    probe2_results = analyze_probe(
        domain,
        V,
        phi,
        facet_tags,
        FACET_PROBE2,
        "Probe 2"
    )

    # ----------------------------------------------------------------
    # Mesh statistics
    # ----------------------------------------------------------------

    global_cells = (
        domain.topology.index_map(
            tdim
        ).size_global
    )

    global_dofs = (
        V.dofmap.index_map.size_global
        * V.dofmap.index_map_bs
    )

    # ----------------------------------------------------------------
    # Output report
    # ----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "=" * 78)
    root_print(comm, "SOLVER RESULTS")
    root_print(comm, "=" * 78)

    root_print(
        comm,
        f"Cells              : "
        f"{global_cells:,}"
    )

    root_print(
        comm,
        f"Potential DOFs     : "
        f"{global_dofs:,}"
    )

    root_print(
        comm,
        f"phi min            : "
        f"{phi_min:.12e} V"
    )

    root_print(
        comm,
        f"phi max            : "
        f"{phi_max:.12e} V"
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

    root_print(comm, "")
    root_print(comm, "INDUCED PROBE POTENTIALS")
    root_print(comm, "-" * 78)

    for result, z in [
        (
            probe1_results,
            z_probe1
        ),
        (
            probe2_results,
            z_probe2
        )
    ]:

        root_print(
            comm,
            f"{result['name']} at z={z:.1f} nm"
        )

        root_print(
            comm,
            f"  facets           : "
            f"{result['facets']:,}"
        )

        root_print(
            comm,
            f"  probe DOFs       : "
            f"{result['dofs']:,}"
        )

        root_print(
            comm,
            f"  minimum phi      : "
            f"{result['min']:.12e} V"
        )

        root_print(
            comm,
            f"  maximum phi      : "
            f"{result['max']:.12e} V"
        )

        root_print(
            comm,
            f"  mean nodal phi   : "
            f"{result['mean_nodal']:.12e} V"
        )

    # ================================================================
    # XDMF + H5 outputs
    # ================================================================

    potential_path = os.path.join(
        a.output,
        "potential.xdmf"
    )

    materials_path = os.path.join(
        a.output,
        "materials.xdmf"
    )

    tags_path = os.path.join(
        a.output,
        "facet_tags.xdmf"
    )

    with io.XDMFFile(
        comm,
        potential_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        xdmf.write_function(
            phi
        )

    with io.XDMFFile(
        comm,
        materials_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        xdmf.write_function(
            epsilon
        )

        xdmf.write_function(
            material_id
        )

    facet_tags.name = "facet_tags"

    with io.XDMFFile(
        comm,
        tags_path,
        "w"
    ) as xdmf:

        xdmf.write_mesh(
            domain
        )

        try:

            xdmf.write_meshtags(
                facet_tags,
                domain.geometry
            )

        except TypeError:

            xdmf.write_meshtags(
                facet_tags
            )

    # ----------------------------------------------------------------
    # JSON results
    # ----------------------------------------------------------------

    if comm.rank == 0:

        results = {
            "geometry_nm": {
                "lx": a.lx,
                "ly": a.ly,
                "air_height": a.air_height,
                "bottom_z": z_bottom,
                "probe1_z": z_probe1,
                "probe2_z": z_probe2,
                "probe1_radius": a.probe1_radius,
                "probe2_radius": a.probe2_radius,
                "tip_gap": a.gap,
                "tip_radius": a.tip_radius,
                "cone_height": a.cone_height,
                "shank_radius": a.shank_radius
            },

            "voltages_V": {
                "AFM_tip": a.tip_voltage,
                "bottom_back_gate": a.bottom_voltage,
                "probe1": None,
                "probe2": None
            },

            "potential_V": {
                "global_min": float(phi_min),
                "global_max": float(phi_max),
                "probe1": probe1_results,
                "probe2": probe2_results
            },

            "mesh": {
                "cells": int(global_cells),
                "potential_dofs": int(global_dofs),
                "h_apex_nm": a.h_apex,
                "h_device_nm": a.h_device,
                "h_near_nm": a.h_near,
                "h_bottom_nm": a.h_bottom,
                "degree": a.degree
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

    root_print(comm, "")
    root_print(comm, "Wrote:")

    root_print(
        comm,
        f"  {a.output}/potential.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/potential.h5"
    )

    root_print(
        comm,
        f"  {a.output}/materials.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/materials.h5"
    )

    root_print(
        comm,
        f"  {a.output}/facet_tags.xdmf"
    )

    root_print(
        comm,
        f"  {a.output}/facet_tags.h5"
    )

    root_print(
        comm,
        f"  {a.output}/run_results.json"
    )

    root_print(comm, "")
    root_print(
        comm,
        "Done."
    )


if __name__ == "__main__":
    main()
