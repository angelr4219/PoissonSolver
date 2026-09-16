#!/usr/bin/env python3

"""
Four-layer Si/SiGe two-gate electrostatic benchmark.

Stack, from top z=0 downward:

    Layer 1: SiGe     thickness = 10, 30, or 50 nm
    Layer 2: Si       thickness = 3 nm
    Layer 3: SiGe     thickness = 10 nm
    Layer 4: SiGe     thickness = 2000 nm

Boundary conditions:

    Gate 1, upper gate in +y direction: 1 V
    Gate 2, lower gate in -y direction: 0 V
    Bottom surface:                     0 V

All remaining surfaces:
    homogeneous natural Neumann BC

Equation:
    div(eps_r grad(phi)) = 0

Coordinates are expressed in nm.
Potential is expressed in volts.

Because rho=0, using nm for the length coordinates is harmless here.
When physical charge density is added later, the unit handling must be
changed to SI consistently.
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import os
import sys
import numpy as np
import gmsh
import ufl

from dolfinx import fem, io, mesh

# ---------------------------------------------------------------------
# DOLFINx compatibility imports
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
# Command line arguments
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Four-layer Si/SiGe two-gate Laplace simulation"
    )

    # Stack
    parser.add_argument(
        "--t1",
        type=float,
        default=10.0,
        choices=[10.0, 30.0, 50.0],
        help="Layer 1 SiGe thickness in nm"
    )

    parser.add_argument("--t2", type=float, default=3.0)
    parser.add_argument("--t3", type=float, default=10.0)
    parser.add_argument("--t4", type=float, default=2000.0)

    # Dielectric constants
    parser.add_argument("--eps1", type=float, default=12.0)
    parser.add_argument("--eps2", type=float, default=11.7)
    parser.add_argument("--eps3", type=float, default=12.0)
    parser.add_argument("--eps4", type=float, default=12.0)

    # Gate voltages
    parser.add_argument("--gate1-voltage", type=float, default=1.0)
    parser.add_argument("--gate2-voltage", type=float, default=0.0)
    parser.add_argument("--bottom-voltage", type=float, default=0.0)

    # Gate geometry
    parser.add_argument("--gate-width", type=float, default=60.0)
    parser.add_argument("--gate-height", type=float, default=60.0)
    parser.add_argument("--gate-gap", type=float, default=20.0)

    # Domain
    parser.add_argument("--lx", type=float, default=200.0)
    parser.add_argument("--ly", type=float, default=200.0)

    # Mesh
    parser.add_argument(
        "--h-top",
        type=float,
        default=3.0,
        help="Target mesh size near top surface in nm"
    )

    parser.add_argument(
        "--h-bottom",
        type=float,
        default=100.0,
        help="Target mesh size near bottom in nm"
    )

    parser.add_argument(
        "--p",
        type=int,
        default=1,
        help="Finite-element polynomial degree"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory"
    )

    return parser.parse_args()


# =====================================================================
# Utility printing
# =====================================================================

def root_print(comm, *args):
    if comm.rank == 0:
        print(*args, flush=True)


# =====================================================================
# Build gmsh geometry
# =====================================================================

def build_mesh(args, comm):

    t1 = args.t1
    t2 = args.t2
    t3 = args.t3
    t4 = args.t4

    z0 = 0.0
    z1 = t1
    z2 = t1 + t2
    z3 = t1 + t2 + t3
    z4 = t1 + t2 + t3 + t4

    total_depth = z4

    if comm.rank == 0:

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.model.add("sige_two_gate_4layer")

        occ = gmsh.model.occ

        xmin = -args.lx / 2.0
        ymin = -args.ly / 2.0

        # -------------------------------------------------------------
        # Create the four stacked regions
        # -------------------------------------------------------------

        box1 = occ.addBox(
            xmin, ymin, z0,
            args.lx, args.ly, t1
        )

        box2 = occ.addBox(
            xmin, ymin, z1,
            args.lx, args.ly, t2
        )

        box3 = occ.addBox(
            xmin, ymin, z2,
            args.lx, args.ly, t3
        )

        box4 = occ.addBox(
            xmin, ymin, z3,
            args.lx, args.ly, t4
        )

        # Fragment them so interfaces share the same mesh.
        occ.fragment(
            [(3, box1)],
            [(3, box2), (3, box3), (3, box4)]
        )

        occ.synchronize()

        # -------------------------------------------------------------
        # Identify resulting volumes based on center-of-mass z
        # -------------------------------------------------------------

        volumes = gmsh.model.getEntities(3)

        layer1_volumes = []
        layer2_volumes = []
        layer3_volumes = []
        layer4_volumes = []

        for dim, tag in volumes:

            cx, cy, cz = occ.getCenterOfMass(dim, tag)

            if cz < z1:
                layer1_volumes.append(tag)

            elif cz < z2:
                layer2_volumes.append(tag)

            elif cz < z3:
                layer3_volumes.append(tag)

            else:
                layer4_volumes.append(tag)

        if len(layer1_volumes) == 0:
            raise RuntimeError("Could not identify Layer 1 volume.")

        if len(layer2_volumes) == 0:
            raise RuntimeError("Could not identify Layer 2 volume.")

        if len(layer3_volumes) == 0:
            raise RuntimeError("Could not identify Layer 3 volume.")

        if len(layer4_volumes) == 0:
            raise RuntimeError("Could not identify Layer 4 volume.")

        # -------------------------------------------------------------
        # Physical volume IDs
        # -------------------------------------------------------------

        MAT_LAYER1 = 1
        MAT_LAYER2 = 2
        MAT_LAYER3 = 3
        MAT_LAYER4 = 4

        gmsh.model.addPhysicalGroup(
            3, layer1_volumes, MAT_LAYER1
        )
        gmsh.model.setPhysicalName(
            3, MAT_LAYER1, "Layer1_SiGe"
        )

        gmsh.model.addPhysicalGroup(
            3, layer2_volumes, MAT_LAYER2
        )
        gmsh.model.setPhysicalName(
            3, MAT_LAYER2, "Layer2_Si"
        )

        gmsh.model.addPhysicalGroup(
            3, layer3_volumes, MAT_LAYER3
        )
        gmsh.model.setPhysicalName(
            3, MAT_LAYER3, "Layer3_SiGe"
        )

        gmsh.model.addPhysicalGroup(
            3, layer4_volumes, MAT_LAYER4
        )
        gmsh.model.setPhysicalName(
            3, MAT_LAYER4, "Layer4_Buffer"
        )

        # -------------------------------------------------------------
        # Graded mesh
        #
        # Small near z=0.
        # Gradually increases toward bottom of 2000 nm buffer.
        # -------------------------------------------------------------

        field = gmsh.model.mesh.field.add("MathEval")

        h_top = args.h_top
        h_bottom = args.h_bottom

        mesh_expression = (
            f"{h_top}"
            f" + ({h_bottom}-{h_top})"
            f"*(z/{total_depth})"
        )

        gmsh.model.mesh.field.setString(
            field,
            "F",
            mesh_expression
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(field)

        gmsh.option.setNumber(
            "Mesh.MeshSizeExtendFromBoundary", 0
        )
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromPoints", 0
        )
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", 0
        )

        gmsh.option.setNumber(
            "Mesh.Algorithm3D",
            10
        )

        root_print(
            comm,
            "\nGenerating Gmsh 3D tetrahedral mesh..."
        )

        gmsh.model.mesh.generate(3)

    # -------------------------------------------------------------
    # Transfer Gmsh mesh to DOLFINx
    # -------------------------------------------------------------

    mesh_data = gmshio.model_to_mesh(
        gmsh.model,
        comm,
        0,
        gdim=3
    )

    # Compatibility with different DOLFINx versions
    if hasattr(mesh_data, "mesh"):
        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
    else:
        domain, cell_tags, _ = mesh_data

    if comm.rank == 0:
        gmsh.finalize()

    return domain, cell_tags, (z0, z1, z2, z3, z4)


# =====================================================================
# Main solver
# =====================================================================

def main():

    args = parse_args()

    comm = MPI.COMM_WORLD

    if args.output is None:
        output_dir = f"sige_t1_{int(args.t1)}nm"
    else:
        output_dir = args.output

    if comm.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    comm.barrier()

    # -------------------------------------------------------------
    # Print configuration
    # -------------------------------------------------------------

    total_depth = (
        args.t1
        + args.t2
        + args.t3
        + args.t4
    )

    root_print(comm, "")
    root_print(comm, "=" * 72)
    root_print(comm, "Si/SiGe TWO-GATE FOUR-LAYER ELECTROSTATICS")
    root_print(comm, "=" * 72)

    root_print(
        comm,
        f"Domain: {args.lx:.1f} x {args.ly:.1f} x "
        f"{total_depth:.1f} nm"
    )

    root_print(comm, "")
    root_print(comm, "Layer stack:")
    root_print(
        comm,
        f"  Layer 1 SiGe : {args.t1:8.1f} nm   "
        f"eps_r = {args.eps1}"
    )

    root_print(
        comm,
        f"  Layer 2 Si   : {args.t2:8.1f} nm   "
        f"eps_r = {args.eps2}"
    )

    root_print(
        comm,
        f"  Layer 3 SiGe : {args.t3:8.1f} nm   "
        f"eps_r = {args.eps3}"
    )

    root_print(
        comm,
        f"  Layer 4      : {args.t4:8.1f} nm   "
        f"eps_r = {args.eps4}"
    )

    root_print(comm, "")
    root_print(comm, "Voltages:")

    root_print(
        comm,
        f"  Gate 1, +y = {args.gate1_voltage:.4f} V"
    )

    root_print(
        comm,
        f"  Gate 2, -y = {args.gate2_voltage:.4f} V"
    )

    root_print(
        comm,
        f"  Bottom       = {args.bottom_voltage:.4f} V"
    )

    root_print(comm, "")
    root_print(
        comm,
        f"Mesh: h_top={args.h_top} nm, "
        f"h_bottom={args.h_bottom} nm"
    )

    root_print(
        comm,
        f"Finite element degree p={args.p}"
    )

    # -------------------------------------------------------------
    # Mesh
    # -------------------------------------------------------------

    domain, cell_tags, z_interfaces = build_mesh(
        args,
        comm
    )

    z0, z1, z2, z3, z4 = z_interfaces

    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(
        fdim,
        tdim
    )

    # -------------------------------------------------------------
    # Function space
    # -------------------------------------------------------------

    try:
        V = fem.functionspace(
            domain,
            ("Lagrange", args.p)
        )
    except AttributeError:
        V = fem.FunctionSpace(
            domain,
            ("CG", args.p)
        )

    # -------------------------------------------------------------
    # DG0 dielectric field
    # -------------------------------------------------------------

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

    eps_r = fem.Function(Q)
    eps_r.name = "epsilon_r"

    material_id = fem.Function(Q)
    material_id.name = "material_id"

    eps_r.x.array[:] = 0.0
    material_id.x.array[:] = 0.0

    materials = {
        1: args.eps1,
        2: args.eps2,
        3: args.eps3,
        4: args.eps4,
    }

    for mat_id, eps_value in materials.items():

        cells = cell_tags.find(mat_id)

        dofs = fem.locate_dofs_topological(
            Q,
            tdim,
            cells
        )

        eps_r.x.array[dofs] = eps_value
        material_id.x.array[dofs] = float(mat_id)

    eps_r.x.scatter_forward()
    material_id.x.scatter_forward()

    # -------------------------------------------------------------
    # Gate geometry
    #
    # Gate 1 is ABOVE Gate 2.
    #
    # "Above" means +y in the xy plane.
    # -------------------------------------------------------------

    gw = args.gate_width
    gh = args.gate_height
    gap = args.gate_gap

    x_min_gate = -gw / 2.0
    x_max_gate = +gw / 2.0

    gate1_ymin = gap / 2.0
    gate1_ymax = gap / 2.0 + gh

    gate2_ymax = -gap / 2.0
    gate2_ymin = -gap / 2.0 - gh

    atol = 1.0e-8

    # -------------------------------------------------------------
    # Boundary locator functions
    # -------------------------------------------------------------

    def gate1_marker(x):
        return (
            np.isclose(x[2], 0.0, atol=atol)
            & (x[0] >= x_min_gate - atol)
            & (x[0] <= x_max_gate + atol)
            & (x[1] >= gate1_ymin - atol)
            & (x[1] <= gate1_ymax + atol)
        )

    def gate2_marker(x):
        return (
            np.isclose(x[2], 0.0, atol=atol)
            & (x[0] >= x_min_gate - atol)
            & (x[0] <= x_max_gate + atol)
            & (x[1] >= gate2_ymin - atol)
            & (x[1] <= gate2_ymax + atol)
        )

    def bottom_marker(x):
        return np.isclose(
            x[2],
            z4,
            atol=atol
        )

    gate1_facets = mesh.locate_entities_boundary(
        domain,
        fdim,
        gate1_marker
    )

    gate2_facets = mesh.locate_entities_boundary(
        domain,
        fdim,
        gate2_marker
    )

    bottom_facets = mesh.locate_entities_boundary(
        domain,
        fdim,
        bottom_marker
    )

    # -------------------------------------------------------------
    # Check gate tagging
    # -------------------------------------------------------------

    n_gate1_local = len(gate1_facets)
    n_gate2_local = len(gate2_facets)
    n_bottom_local = len(bottom_facets)

    n_gate1 = comm.allreduce(
        n_gate1_local,
        op=MPI.SUM
    )

    n_gate2 = comm.allreduce(
        n_gate2_local,
        op=MPI.SUM
    )

    n_bottom = comm.allreduce(
        n_bottom_local,
        op=MPI.SUM
    )

    root_print(comm, "")
    root_print(comm, "Boundary facet counts:")
    root_print(comm, f"  Gate 1 : {n_gate1}")
    root_print(comm, f"  Gate 2 : {n_gate2}")
    root_print(comm, f"  Bottom : {n_bottom}")

    if n_gate1 == 0:
        raise RuntimeError(
            "ERROR: Gate 1 has zero boundary facets."
        )

    if n_gate2 == 0:
        raise RuntimeError(
            "ERROR: Gate 2 has zero boundary facets."
        )

    if n_bottom == 0:
        raise RuntimeError(
            "ERROR: Bottom has zero boundary facets."
        )

    # -------------------------------------------------------------
    # Dirichlet DOFs
    # -------------------------------------------------------------

    gate1_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        gate1_facets
    )

    gate2_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        gate2_facets
    )

    bottom_dofs = fem.locate_dofs_topological(
        V,
        fdim,
        bottom_facets
    )

    gate1_value = fem.Constant(
        domain,
        PETSc.ScalarType(args.gate1_voltage)
    )

    gate2_value = fem.Constant(
        domain,
        PETSc.ScalarType(args.gate2_voltage)
    )

    bottom_value = fem.Constant(
        domain,
        PETSc.ScalarType(args.bottom_voltage)
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

    # -------------------------------------------------------------
    # Weak formulation
    #
    # div(eps_r grad(phi)) = 0
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    try:
        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="sige_"
        )

    except TypeError:
        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options
        )

    root_print(comm, "")
    root_print(comm, "Solving Laplace problem...")

    phi = problem.solve()
    phi.name = "phi"

    phi.x.scatter_forward()

    # -------------------------------------------------------------
    # Solution statistics
    # -------------------------------------------------------------

    local_min = np.min(
        np.real(phi.x.array)
    )

    local_max = np.max(
        np.real(phi.x.array)
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

    root_print(comm, "")
    root_print(comm, "=" * 72)
    root_print(comm, "SOLVER RESULTS")
    root_print(comm, "=" * 72)

    root_print(
        comm,
        f"Cells        : {global_cells:,}"
    )

    root_print(
        comm,
        f"DOFs         : {global_dofs:,}"
    )

    root_print(
        comm,
        f"Minimum phi  : {phi_min:.10e} V"
    )

    root_print(
        comm,
        f"Maximum phi  : {phi_max:.10e} V"
    )

    # -------------------------------------------------------------
    # PETSc solver information
    # -------------------------------------------------------------

    try:
        ksp = problem.solver

        reason = ksp.getConvergedReason()
        iterations = ksp.getIterationNumber()

        root_print(
            comm,
            f"KSP reason   : {reason}"
        )

        root_print(
            comm,
            f"KSP iterations: {iterations}"
        )

    except Exception:
        root_print(
            comm,
            "KSP information unavailable "
            "with this DOLFINx version."
        )

    # -------------------------------------------------------------
    # Sanity check using maximum principle
    # -------------------------------------------------------------

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

    max_principle_ok = (
        phi_min >= expected_min - tolerance
        and
        phi_max <= expected_max + tolerance
    )

    root_print(comm, "")
    root_print(
        comm,
        "Maximum-principle check:",
        "PASS" if max_principle_ok else "CHECK RESULT"
    )

    # -------------------------------------------------------------
    # Print z positions
    # -------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "Interface positions:")

    root_print(
        comm,
        f"  Top                         z = {z0:.1f} nm"
    )

    root_print(
        comm,
        f"  Layer 1 / Layer 2          z = {z1:.1f} nm"
    )

    root_print(
        comm,
        f"  Layer 2 / Layer 3          z = {z2:.1f} nm"
    )

    root_print(
        comm,
        f"  Layer 3 / Layer 4          z = {z3:.1f} nm"
    )

    root_print(
        comm,
        f"  Bottom gate                z = {z4:.1f} nm"
    )

    qw_midpoint = z1 + args.t2 / 2.0

    root_print(
        comm,
        f"  Layer 2 midpoint           z = {qw_midpoint:.1f} nm"
    )

    # -------------------------------------------------------------
    # Print gate coordinates
    # -------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "Gate coordinates:")

    root_print(
        comm,
        "  Gate 1:"
    )

    root_print(
        comm,
        f"    x = [{x_min_gate:.1f}, {x_max_gate:.1f}] nm"
    )

    root_print(
        comm,
        f"    y = [{gate1_ymin:.1f}, {gate1_ymax:.1f}] nm"
    )

    root_print(
        comm,
        f"    V = {args.gate1_voltage:.3f} V"
    )

    root_print(
        comm,
        "  Gate 2:"
    )

    root_print(
        comm,
        f"    x = [{x_min_gate:.1f}, {x_max_gate:.1f}] nm"
    )

    root_print(
        comm,
        f"    y = [{gate2_ymin:.1f}, {gate2_ymax:.1f}] nm"
    )

    root_print(
        comm,
        f"    V = {args.gate2_voltage:.3f} V"
    )

    # -------------------------------------------------------------
    # Write XDMF
    # -------------------------------------------------------------

    potential_file = os.path.join(
        output_dir,
        "potential.xdmf"
    )

    materials_file = os.path.join(
        output_dir,
        "materials.xdmf"
    )

    with io.XDMFFile(
        comm,
        potential_file,
        "w"
    ) as xdmf:

        xdmf.write_mesh(domain)
        xdmf.write_function(phi)

    with io.XDMFFile(
        comm,
        materials_file,
        "w"
    ) as xdmf:

        xdmf.write_mesh(domain)
        xdmf.write_function(eps_r)
        xdmf.write_function(material_id)

    # -------------------------------------------------------------
    # Run summary
    # -------------------------------------------------------------

    if comm.rank == 0:

        summary_path = os.path.join(
            output_dir,
            "run_summary.txt"
        )

        with open(summary_path, "w") as f:

            f.write(
                "Si/SiGe two-gate four-layer simulation\n"
            )

            f.write("=" * 60 + "\n\n")

            f.write(
                f"Layer 1 thickness = {args.t1} nm\n"
            )

            f.write(
                f"Layer 2 thickness = {args.t2} nm\n"
            )

            f.write(
                f"Layer 3 thickness = {args.t3} nm\n"
            )

            f.write(
                f"Layer 4 thickness = {args.t4} nm\n"
            )

            f.write(
                f"Total depth = {total_depth} nm\n\n"
            )

            f.write(
                f"epsilon_r = "
                f"{args.eps1}, "
                f"{args.eps2}, "
                f"{args.eps3}, "
                f"{args.eps4}\n\n"
            )

            f.write(
                f"Gate 1 voltage = "
                f"{args.gate1_voltage} V\n"
            )

            f.write(
                f"Gate 2 voltage = "
                f"{args.gate2_voltage} V\n"
            )

            f.write(
                f"Bottom voltage = "
                f"{args.bottom_voltage} V\n\n"
            )

            f.write(
                f"h_top = {args.h_top} nm\n"
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
                f"Layer 2 midpoint z = "
                f"{qw_midpoint} nm\n"
            )

    root_print(comm, "")
    root_print(
        comm,
        f"Potential written to: {potential_file}"
    )

    root_print(
        comm,
        f"Materials written to: {materials_file}"
    )

    root_print(
        comm,
        f"Summary written to: "
        f"{os.path.join(output_dir, 'run_summary.txt')}"
    )

    root_print(comm, "")
    root_print(comm, "Run complete.")
    root_print(comm, "=" * 72)


if __name__ == "__main__":
    main()
