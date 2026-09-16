#!/usr/bin/env python3

"""
Vertically stacked Si/SiGe electrostatics benchmark.

Geometry, from top to bottom:

    Layer 1: Si
        z = 0 to 2 nm
        thickness = 2 nm
        eps_r = 11.7

    Layer 2: SiGe
        z = 2 to 30 nm
        thickness = 28 nm
        eps_r = 12.0

    Layer 3: Si
        z = 30 to 40 nm
        thickness = 10 nm
        eps_r = 11.7

        Gate 1:
            circular internal Dirichlet surface
            centered at x = y = 0
            z = 35 nm
            radius = 10 nm
            V = 1 V

    Layer 4: SiGe
        z = 40 to 43 nm
        thickness = 3 nm
        eps_r = 12.0

    Layer 5: Si
        z = 43 to 53 nm
        thickness = 10 nm
        eps_r = 11.7

        Gate 2:
            circular internal Dirichlet surface
            centered at x = y = 0
            z = 48 nm
            radius = 10 nm
            V = 0 V

    Layer 6: SiGe buffer
        z = 53 to 2053 nm
        thickness = 2000 nm
        eps_r = 12.0

    Bottom surface:
        z = 2053 nm
        V = 0 V

Equation:

    div(eps_r * grad(phi)) = 0

rho = 0.

Coordinates are in nm.
Potential is in volts.

Because rho = 0, nm coordinates are acceptable for the charge-free
Laplace problem. When physical charge density is added later, units
must be converted consistently to SI.
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import os
import numpy as np
import gmsh
import ufl

from dolfinx import fem, io


# =====================================================================
# DOLFINx compatibility imports
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
# Physical IDs
# =====================================================================

MAT_SI_TOP = 1
MAT_SIGE_28 = 2
MAT_SI_GATE1 = 3
MAT_SIGE_3 = 4
MAT_SI_GATE2 = 5
MAT_SIGE_BUFFER = 6

FACET_GATE1 = 101
FACET_GATE2 = 102
FACET_BOTTOM = 103


# =====================================================================
# Arguments
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Vertical Si/SiGe two circular gate electrostatics"
    )

    # Domain
    parser.add_argument(
        "--lx",
        type=float,
        default=200.0,
        help="Domain size in x [nm]"
    )

    parser.add_argument(
        "--ly",
        type=float,
        default=200.0,
        help="Domain size in y [nm]"
    )

    # Layer thicknesses
    parser.add_argument("--t-si-top", type=float, default=2.0)
    parser.add_argument("--t-sige-upper", type=float, default=28.0)
    parser.add_argument("--t-si-gate1", type=float, default=10.0)
    parser.add_argument("--t-sige-spacer", type=float, default=3.0)
    parser.add_argument("--t-si-gate2", type=float, default=10.0)
    parser.add_argument("--t-buffer", type=float, default=2000.0)

    # Dielectric constants
    parser.add_argument(
        "--eps-si",
        type=float,
        default=11.7
    )

    parser.add_argument(
        "--eps-sige",
        type=float,
        default=12.0
    )

    # Gates
    parser.add_argument(
        "--gate1-radius",
        type=float,
        default=10.0,
        help="Gate 1 disk radius [nm]"
    )

    parser.add_argument(
        "--gate2-radius",
        type=float,
        default=10.0,
        help="Gate 2 disk radius [nm]"
    )

    parser.add_argument(
        "--gate1-voltage",
        type=float,
        default=1.0,
        help="Gate 1 voltage [V]"
    )

    parser.add_argument(
        "--gate2-voltage",
        type=float,
        default=0.0,
        help="Gate 2 voltage [V]"
    )

    parser.add_argument(
        "--bottom-voltage",
        type=float,
        default=0.0,
        help="Bottom surface voltage [V]"
    )

    # FEM / mesh
    parser.add_argument(
        "--h-device",
        type=float,
        default=3.0,
        help="Target mesh size in upper device region [nm]"
    )

    parser.add_argument(
        "--h-bottom",
        type=float,
        default=100.0,
        help="Target mesh size near bottom [nm]"
    )

    parser.add_argument(
        "--p",
        type=int,
        default=1,
        help="Finite element degree"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="sige_vertical_2_28_10_3_10_2000"
    )

    return parser.parse_args()


# =====================================================================
# Helper
# =====================================================================

def root_print(comm, *args):

    if comm.rank == 0:
        print(*args, flush=True)


# =====================================================================
# Build Gmsh geometry
# =====================================================================

def build_mesh(args, comm):

    # -----------------------------------------------------------------
    # Vertical coordinates
    # -----------------------------------------------------------------

    z0 = 0.0

    z1 = z0 + args.t_si_top
    z2 = z1 + args.t_sige_upper
    z3 = z2 + args.t_si_gate1
    z4 = z3 + args.t_sige_spacer
    z5 = z4 + args.t_si_gate2
    z6 = z5 + args.t_buffer

    z_gate1 = z2 + args.t_si_gate1 / 2.0
    z_gate2 = z4 + args.t_si_gate2 / 2.0

    if comm.rank == 0:

        gmsh.initialize()

        gmsh.option.setNumber(
            "General.Terminal",
            1
        )

        gmsh.model.add(
            "sige_vertical_two_gate"
        )

        occ = gmsh.model.occ

        xmin = -args.lx / 2.0
        ymin = -args.ly / 2.0

        # =============================================================
        # Dielectric volumes
        # =============================================================

        v1 = occ.addBox(
            xmin, ymin, z0,
            args.lx, args.ly, args.t_si_top
        )

        v2 = occ.addBox(
            xmin, ymin, z1,
            args.lx, args.ly, args.t_sige_upper
        )

        v3 = occ.addBox(
            xmin, ymin, z2,
            args.lx, args.ly, args.t_si_gate1
        )

        v4 = occ.addBox(
            xmin, ymin, z3,
            args.lx, args.ly, args.t_sige_spacer
        )

        v5 = occ.addBox(
            xmin, ymin, z4,
            args.lx, args.ly, args.t_si_gate2
        )

        v6 = occ.addBox(
            xmin, ymin, z5,
            args.lx, args.ly, args.t_buffer
        )

        # Fragment volumes so all material interfaces are conforming.
        occ.fragment(
            [(3, v1)],
            [
                (3, v2),
                (3, v3),
                (3, v4),
                (3, v5),
                (3, v6)
            ]
        )

        occ.synchronize()

        # =============================================================
        # Find resulting volumes by center-of-mass z
        # =============================================================

        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []

        for dim, tag in gmsh.model.getEntities(3):

            _, _, cz = occ.getCenterOfMass(
                dim,
                tag
            )

            if cz < z1:
                layer1.append(tag)

            elif cz < z2:
                layer2.append(tag)

            elif cz < z3:
                layer3.append(tag)

            elif cz < z4:
                layer4.append(tag)

            elif cz < z5:
                layer5.append(tag)

            else:
                layer6.append(tag)

        layer_sets = [
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6
        ]

        for i, vols in enumerate(
            layer_sets,
            start=1
        ):
            if len(vols) == 0:
                raise RuntimeError(
                    f"Could not identify material layer {i}"
                )

        # =============================================================
        # Material physical groups
        # =============================================================

        material_groups = [
            (
                MAT_SI_TOP,
                layer1,
                "Si_top_2nm"
            ),
            (
                MAT_SIGE_28,
                layer2,
                "SiGe_28nm"
            ),
            (
                MAT_SI_GATE1,
                layer3,
                "Si_gate1_10nm"
            ),
            (
                MAT_SIGE_3,
                layer4,
                "SiGe_3nm"
            ),
            (
                MAT_SI_GATE2,
                layer5,
                "Si_gate2_10nm"
            ),
            (
                MAT_SIGE_BUFFER,
                layer6,
                "SiGe_buffer_2000nm"
            )
        ]

        for physical_id, volumes, name in material_groups:

            gmsh.model.addPhysicalGroup(
                3,
                volumes,
                physical_id
            )

            gmsh.model.setPhysicalName(
                3,
                physical_id,
                name
            )

        # =============================================================
        # Circular internal gates
        # =============================================================

        gate1 = occ.addDisk(
            0.0,
            0.0,
            z_gate1,
            args.gate1_radius,
            args.gate1_radius
        )

        gate2 = occ.addDisk(
            0.0,
            0.0,
            z_gate2,
            args.gate2_radius,
            args.gate2_radius
        )

        occ.synchronize()

        # Embed Gate 1 into the first 10 nm Si layer.
        for volume_tag in layer3:

            gmsh.model.mesh.embed(
                2,
                [gate1],
                3,
                volume_tag
            )

        # Embed Gate 2 into the second 10 nm Si layer.
        for volume_tag in layer5:

            gmsh.model.mesh.embed(
                2,
                [gate2],
                3,
                volume_tag
            )

        # Gate physical groups
        gmsh.model.addPhysicalGroup(
            2,
            [gate1],
            FACET_GATE1
        )

        gmsh.model.setPhysicalName(
            2,
            FACET_GATE1,
            "Gate1_1V"
        )

        gmsh.model.addPhysicalGroup(
            2,
            [gate2],
            FACET_GATE2
        )

        gmsh.model.setPhysicalName(
            2,
            FACET_GATE2,
            "Gate2_0V"
        )

        # =============================================================
        # Bottom external surface
        # =============================================================

        bottom_surfaces = []

        tolerance = 1.0e-6

        for dim, tag in gmsh.model.getEntities(2):

            if tag == gate1 or tag == gate2:
                continue

            bbox = gmsh.model.getBoundingBox(
                dim,
                tag
            )

            zmin = bbox[2]
            zmax = bbox[5]

            if (
                abs(zmin - z6) < tolerance
                and
                abs(zmax - z6) < tolerance
            ):
                bottom_surfaces.append(tag)

        if len(bottom_surfaces) == 0:

            raise RuntimeError(
                "Could not identify bottom boundary."
            )

        gmsh.model.addPhysicalGroup(
            2,
            bottom_surfaces,
            FACET_BOTTOM
        )

        gmsh.model.setPhysicalName(
            2,
            FACET_BOTTOM,
            "Bottom_0V"
        )

        # =============================================================
        # Mesh field
        #
        # Keep upper device fine and gradually coarsen throughout
        # the 2000 nm lower buffer.
        # =============================================================

        z_device_end = 70.0

        # Base grading throughout domain.
        graded_field = gmsh.model.mesh.field.add(
            "MathEval"
        )

        expression = (
            f"{args.h_device}"
            f"+({args.h_bottom}-{args.h_device})"
            f"*(z/{z6})"
        )

        gmsh.model.mesh.field.setString(
            graded_field,
            "F",
            expression
        )

        # Force device region z=0...70 nm to remain fine.
        device_field = gmsh.model.mesh.field.add(
            "Box"
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "VIn",
            args.h_device
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "VOut",
            args.h_bottom
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "XMin",
            xmin
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "XMax",
            -xmin
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "YMin",
            ymin
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "YMax",
            -ymin
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "ZMin",
            0.0
        )

        gmsh.model.mesh.field.setNumber(
            device_field,
            "ZMax",
            z_device_end
        )

        # Minimum of both mesh fields.
        background_field = gmsh.model.mesh.field.add(
            "Min"
        )

        gmsh.model.mesh.field.setNumbers(
            background_field,
            "FieldsList",
            [
                graded_field,
                device_field
            ]
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(
            background_field
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

        root_print(comm, "")
        root_print(
            comm,
            "Generating Gmsh tetrahedral mesh..."
        )

        gmsh.model.mesh.generate(3)

    # =============================================================
    # Gmsh -> DOLFINx
    # =============================================================

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
        z0,
        z1,
        z2,
        z3,
        z4,
        z5,
        z6,
        z_gate1,
        z_gate2
    )


# =====================================================================
# Main
# =====================================================================

def main():

    args = parse_args()

    comm = MPI.COMM_WORLD

    if comm.rank == 0:

        os.makedirs(
            args.output,
            exist_ok=True
        )

    comm.barrier()

    # -----------------------------------------------------------------
    # Build z positions before mesh for reporting
    # -----------------------------------------------------------------

    z0 = 0.0
    z1 = z0 + args.t_si_top
    z2 = z1 + args.t_sige_upper
    z3 = z2 + args.t_si_gate1
    z4 = z3 + args.t_sige_spacer
    z5 = z4 + args.t_si_gate2
    z6 = z5 + args.t_buffer

    z_gate1 = z2 + args.t_si_gate1 / 2.0
    z_gate2 = z4 + args.t_si_gate2 / 2.0

    root_print(comm, "")
    root_print(comm, "=" * 76)
    root_print(
        comm,
        "Si/SiGe VERTICAL TWO-CIRCULAR-GATE ELECTROSTATICS"
    )
    root_print(comm, "=" * 76)

    root_print(
        comm,
        f"Domain: {args.lx:.1f} x "
        f"{args.ly:.1f} x {z6:.1f} nm"
    )

    root_print(comm, "")
    root_print(comm, "VERTICAL STACK")

    root_print(
        comm,
        f"  Layer 1  Si          "
        f"{z0:7.1f} -> {z1:7.1f} nm "
        f"({args.t_si_top:.1f} nm)"
    )

    root_print(
        comm,
        f"  Layer 2  SiGe        "
        f"{z1:7.1f} -> {z2:7.1f} nm "
        f"({args.t_sige_upper:.1f} nm)"
    )

    root_print(
        comm,
        f"  Layer 3  Si          "
        f"{z2:7.1f} -> {z3:7.1f} nm "
        f"({args.t_si_gate1:.1f} nm)"
    )

    root_print(
        comm,
        f"           Gate 1      "
        f"z = {z_gate1:.1f} nm, "
        f"R = {args.gate1_radius:.1f} nm, "
        f"V = {args.gate1_voltage:.3f} V"
    )

    root_print(
        comm,
        f"  Layer 4  SiGe        "
        f"{z3:7.1f} -> {z4:7.1f} nm "
        f"({args.t_sige_spacer:.1f} nm)"
    )

    root_print(
        comm,
        f"  Layer 5  Si          "
        f"{z4:7.1f} -> {z5:7.1f} nm "
        f"({args.t_si_gate2:.1f} nm)"
    )

    root_print(
        comm,
        f"           Gate 2      "
        f"z = {z_gate2:.1f} nm, "
        f"R = {args.gate2_radius:.1f} nm, "
        f"V = {args.gate2_voltage:.3f} V"
    )

    root_print(
        comm,
        f"  Layer 6  SiGe buffer "
        f"{z5:7.1f} -> {z6:7.1f} nm "
        f"({args.t_buffer:.1f} nm)"
    )

    root_print(
        comm,
        f"  Bottom gate surface  "
        f"z = {z6:.1f} nm, "
        f"V = {args.bottom_voltage:.3f} V"
    )

    root_print(comm, "")
    root_print(
        comm,
        f"eps_r(Si)   = {args.eps_si}"
    )

    root_print(
        comm,
        f"eps_r(SiGe) = {args.eps_sige}"
    )

    root_print(
        comm,
        f"Mesh: h_device={args.h_device} nm, "
        f"h_bottom={args.h_bottom} nm, "
        f"p={args.p}"
    )

    # -----------------------------------------------------------------
    # Mesh
    # -----------------------------------------------------------------

    (
        domain,
        cell_tags,
        facet_tags,
        z0,
        z1,
        z2,
        z3,
        z4,
        z5,
        z6,
        z_gate1,
        z_gate2
    ) = build_mesh(
        args,
        comm
    )

    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(
        fdim,
        tdim
    )

    # -----------------------------------------------------------------
    # Function spaces
    # -----------------------------------------------------------------

    try:

        V = fem.functionspace(
            domain,
            ("Lagrange", args.p)
        )

        Q = fem.functionspace(
            domain,
            ("DG", 0)
        )

    except AttributeError:

        V = fem.FunctionSpace(
            domain,
            ("CG", args.p)
        )

        Q = fem.FunctionSpace(
            domain,
            ("DG", 0)
        )

    # -----------------------------------------------------------------
    # Material fields
    # -----------------------------------------------------------------

    eps_r = fem.Function(Q)
    eps_r.name = "epsilon_r"

    material_id = fem.Function(Q)
    material_id.name = "material_id"

    eps_r.x.array[:] = 0.0
    material_id.x.array[:] = 0.0

    materials = {
        MAT_SI_TOP: args.eps_si,
        MAT_SIGE_28: args.eps_sige,
        MAT_SI_GATE1: args.eps_si,
        MAT_SIGE_3: args.eps_sige,
        MAT_SI_GATE2: args.eps_si,
        MAT_SIGE_BUFFER: args.eps_sige
    }

    root_print(comm, "")
    root_print(comm, "Material tag check:")

    for material_tag, epsilon_value in materials.items():

        cells = cell_tags.find(
            material_tag
        )

        global_count = comm.allreduce(
            len(cells),
            op=MPI.SUM
        )

        root_print(
            comm,
            f"  material tag {material_tag}: "
            f"{global_count:,} cells"
        )

        if global_count == 0:

            raise RuntimeError(
                f"Material tag {material_tag} has zero cells."
            )

        dofs = fem.locate_dofs_topological(
            Q,
            tdim,
            cells
        )

        eps_r.x.array[dofs] = epsilon_value

        material_id.x.array[dofs] = float(
            material_tag
        )

    eps_r.x.scatter_forward()
    material_id.x.scatter_forward()

    # -----------------------------------------------------------------
    # Internal gate and bottom facet tags
    # -----------------------------------------------------------------

    if facet_tags is None:

        raise RuntimeError(
            "No facet tags were transferred from Gmsh."
        )

    facet_dict = {
        "Gate 1": FACET_GATE1,
        "Gate 2": FACET_GATE2,
        "Bottom": FACET_BOTTOM
    }

    tagged_facets = {}

    root_print(comm, "")
    root_print(comm, "Facet tag integrity check:")

    for name, tag in facet_dict.items():

        facets = facet_tags.find(
            tag
        )

        tagged_facets[tag] = facets

        global_count = comm.allreduce(
            len(facets),
            op=MPI.SUM
        )

        root_print(
            comm,
            f"  {name:8s} | "
            f"tag={tag:3d} | "
            f"facets={global_count:,}"
        )

        if global_count == 0:

            raise RuntimeError(
                f"{name} tag has zero facets."
            )

    # -----------------------------------------------------------------
    # Dirichlet DOFs
    # -----------------------------------------------------------------

    gate1_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        tagged_facets[FACET_GATE1]
    )

    gate2_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        tagged_facets[FACET_GATE2]
    )

    bottom_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        tagged_facets[FACET_BOTTOM]
    )

    root_print(comm, "")
    root_print(comm, "Dirichlet DOF counts:")

    for name, dofs in [
        ("Gate 1", gate1_dofs),
        ("Gate 2", gate2_dofs),
        ("Bottom", bottom_dofs)
    ]:

        count = comm.allreduce(
            len(dofs),
            op=MPI.SUM
        )

        root_print(
            comm,
            f"  {name:8s}: {count:,}"
        )

    # -----------------------------------------------------------------
    # Dirichlet values
    # -----------------------------------------------------------------

    gate1_value = fem.Constant(
        domain,
        PETSc.ScalarType(
            args.gate1_voltage
        )
    )

    gate2_value = fem.Constant(
        domain,
        PETSc.ScalarType(
            args.gate2_voltage
        )
    )

    bottom_value = fem.Constant(
        domain,
        PETSc.ScalarType(
            args.bottom_voltage
        )
    )

    bc_gate1 = fem.dirichletbc(
        gate1_value,
        gate1_dofs,
        V
    )

    bc_gate2 = fem.dirichletbc(
        gate2_value,
        gate2_dofs,
        V
    )

    bc_bottom = fem.dirichletbc(
        bottom_value,
        bottom_dofs,
        V
    )

    bcs = [
        bc_gate1,
        bc_gate2,
        bc_bottom
    ]

    # -----------------------------------------------------------------
    # Weak problem
    # -----------------------------------------------------------------

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dx = ufl.Measure(
        "dx",
        domain=domain
    )

    a = (
        eps_r
        * ufl.inner(
            ufl.grad(u),
            ufl.grad(v)
        )
        * dx
    )

    zero = fem.Constant(
        domain,
        PETSc.ScalarType(0.0)
    )

    L = zero * v * dx

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu"
    }

    try:

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="sige_vertical_"
        )

    except TypeError:

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options
        )

    root_print(comm, "")
    root_print(comm, "Solving Laplace equation...")

    phi = problem.solve()

    phi.name = "phi"

    phi.x.scatter_forward()

    # -----------------------------------------------------------------
    # Result statistics
    # -----------------------------------------------------------------

    phi_array = np.real(
        phi.x.array
    )

    local_min = np.min(
        phi_array
    )

    local_max = np.max(
        phi_array
    )

    phi_min = comm.allreduce(
        local_min,
        op=MPI.MIN
    )

    phi_max = comm.allreduce(
        local_max,
        op=MPI.MAX
    )

    local_cells = domain.topology.index_map(
        tdim
    ).size_local

    global_cells = comm.allreduce(
        local_cells,
        op=MPI.SUM
    )

    local_dofs = V.dofmap.index_map.size_local

    global_dofs = comm.allreduce(
        local_dofs,
        op=MPI.SUM
    )

    # -----------------------------------------------------------------
    # Maximum principle sanity check
    # -----------------------------------------------------------------

    expected_min = min(
        args.gate1_voltage,
        args.gate2_voltage,
        args.bottom_voltage
    )

    expected_max = max(
        args.gate1_voltage,
        args.gate2_voltage,
        args.bottom_voltage
    )

    tolerance = 1.0e-8

    maximum_principle_ok = (
        phi_min >= expected_min - tolerance
        and
        phi_max <= expected_max + tolerance
    )

    # -----------------------------------------------------------------
    # Print result
    # -----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "=" * 76)
    root_print(comm, "SOLVER RESULTS")
    root_print(comm, "=" * 76)

    root_print(
        comm,
        f"Cells         : {global_cells:,}"
    )

    root_print(
        comm,
        f"DOFs          : {global_dofs:,}"
    )

    root_print(
        comm,
        f"Minimum phi   : {phi_min:.12e} V"
    )

    root_print(
        comm,
        f"Maximum phi   : {phi_max:.12e} V"
    )

    try:

        root_print(
            comm,
            f"KSP reason    : "
            f"{problem.solver.getConvergedReason()}"
        )

        root_print(
            comm,
            f"KSP iterations: "
            f"{problem.solver.getIterationNumber()}"
        )

    except Exception:

        root_print(
            comm,
            "KSP details unavailable."
        )

    root_print(
        comm,
        "Maximum principle:",
        "PASS"
        if maximum_principle_ok
        else "CHECK"
    )

    root_print(comm, "")
    root_print(comm, "Important z positions:")

    root_print(
        comm,
        f"  Top                     : {z0:.1f} nm"
    )

    root_print(
        comm,
        f"  Si / SiGe               : {z1:.1f} nm"
    )

    root_print(
        comm,
        f"  SiGe / Si               : {z2:.1f} nm"
    )

    root_print(
        comm,
        f"  Gate 1                   : {z_gate1:.1f} nm"
    )

    root_print(
        comm,
        f"  Si / SiGe               : {z3:.1f} nm"
    )

    root_print(
        comm,
        f"  SiGe / Si               : {z4:.1f} nm"
    )

    root_print(
        comm,
        f"  Gate 2                   : {z_gate2:.1f} nm"
    )

    root_print(
        comm,
        f"  Si / buffer              : {z5:.1f} nm"
    )

    root_print(
        comm,
        f"  Bottom                   : {z6:.1f} nm"
    )

    # =================================================================
    # Output
    #
    # XDMFFile writes companion .h5 files automatically.
    # =================================================================

    potential_path = os.path.join(
        args.output,
        "potential.xdmf"
    )

    materials_path = os.path.join(
        args.output,
        "materials.xdmf"
    )

    facet_path = os.path.join(
        args.output,
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
            eps_r
        )

        xdmf.write_function(
            material_id
        )

    with io.XDMFFile(
        comm,
        facet_path,
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

    # -----------------------------------------------------------------
    # Text run summary
    # -----------------------------------------------------------------

    if comm.rank == 0:

        summary_path = os.path.join(
            args.output,
            "run_summary.txt"
        )

        with open(
            summary_path,
            "w"
        ) as f:

            f.write(
                "Si/SiGe vertical two-gate electrostatics\n"
            )

            f.write(
                "=" * 65 + "\n\n"
            )

            f.write(
                f"Domain = {args.lx} x "
                f"{args.ly} x {z6} nm\n\n"
            )

            f.write(
                "VERTICAL STACK\n"
            )

            f.write(
                f"Si       : {z0} -> {z1} nm\n"
            )

            f.write(
                f"SiGe     : {z1} -> {z2} nm\n"
            )

            f.write(
                f"Si       : {z2} -> {z3} nm\n"
            )

            f.write(
                f"SiGe     : {z3} -> {z4} nm\n"
            )

            f.write(
                f"Si       : {z4} -> {z5} nm\n"
            )

            f.write(
                f"SiGe buf : {z5} -> {z6} nm\n\n"
            )

            f.write(
                f"Gate 1 center = "
                f"(0, 0, {z_gate1}) nm\n"
            )

            f.write(
                f"Gate 1 radius = "
                f"{args.gate1_radius} nm\n"
            )

            f.write(
                f"Gate 1 voltage = "
                f"{args.gate1_voltage} V\n\n"
            )

            f.write(
                f"Gate 2 center = "
                f"(0, 0, {z_gate2}) nm\n"
            )

            f.write(
                f"Gate 2 radius = "
                f"{args.gate2_radius} nm\n"
            )

            f.write(
                f"Gate 2 voltage = "
                f"{args.gate2_voltage} V\n\n"
            )

            f.write(
                f"Bottom voltage = "
                f"{args.bottom_voltage} V\n\n"
            )

            f.write(
                f"eps_r Si = {args.eps_si}\n"
            )

            f.write(
                f"eps_r SiGe = {args.eps_sige}\n\n"
            )

            f.write(
                f"h_device = {args.h_device} nm\n"
            )

            f.write(
                f"h_bottom = {args.h_bottom} nm\n"
            )

            f.write(
                f"p = {args.p}\n\n"
            )

            f.write(
                f"Cells = {global_cells}\n"
            )

            f.write(
                f"DOFs = {global_dofs}\n"
            )

            f.write(
                f"phi_min = {phi_min:.12e} V\n"
            )

            f.write(
                f"phi_max = {phi_max:.12e} V\n"
            )

            f.write(
                "Maximum principle = "
                + (
                    "PASS\n"
                    if maximum_principle_ok
                    else "CHECK\n"
                )
            )

    # -----------------------------------------------------------------
    # Final output listing
    # -----------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "Output files:")

    root_print(
        comm,
        f"  {args.output}/potential.xdmf"
    )

    root_print(
        comm,
        f"  {args.output}/potential.h5"
    )

    root_print(
        comm,
        f"  {args.output}/materials.xdmf"
    )

    root_print(
        comm,
        f"  {args.output}/materials.h5"
    )

    root_print(
        comm,
        f"  {args.output}/facet_tags.xdmf"
    )

    root_print(
        comm,
        f"  {args.output}/facet_tags.h5"
    )

    root_print(
        comm,
        f"  {args.output}/run_summary.txt"
    )

    root_print(comm, "")
    root_print(
        comm,
        "Simulation complete."
    )

    root_print(
        comm,
        "=" * 76
    )


if __name__ == "__main__":
    main()
