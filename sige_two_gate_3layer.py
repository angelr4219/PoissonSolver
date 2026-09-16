#!/usr/bin/env python3

"""
Three-layer Si/SiGe electrostatic benchmark in DOLFINx.

Stack, top to bottom:
    0  - 10 nm : SiGe
    10 - 13 nm : Si
    13 - 23 nm : SiGe

Top gates:
    Gate 1 = 1 V
    Gate 2 = 0 V

Bottom:
    0 V

All other boundaries:
    Natural Neumann, n . eps grad(phi) = 0

PDE:
    div(eps_r grad(phi)) = 0

Coordinates are expressed in nm.
Potential is expressed in volts.
"""

import argparse
import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh, io


def parse_args():
    parser = argparse.ArgumentParser(
        description="3-layer Si/SiGe two-gate Laplace benchmark"
    )

    # Domain
    parser.add_argument("--Lx", type=float, default=200.0,
                        help="Domain size in x [nm]")
    parser.add_argument("--Ly", type=float, default=120.0,
                        help="Domain size in y [nm]")

    # Layer thicknesses
    parser.add_argument("--t1", type=float, default=10.0,
                        help="Top SiGe thickness [nm]")
    parser.add_argument("--t2", type=float, default=3.0,
                        help="Si layer thickness [nm]")
    parser.add_argument("--t3", type=float, default=10.0,
                        help="Bottom SiGe thickness [nm]")

    # Relative permittivities
    parser.add_argument("--eps-sige-top", type=float, default=12.0)
    parser.add_argument("--eps-si", type=float, default=11.7)
    parser.add_argument("--eps-sige-bottom", type=float, default=12.0)

    # Gate geometry
    parser.add_argument("--gate-width", type=float, default=60.0,
                        help="Gate width in x [nm]")
    parser.add_argument("--gate-height", type=float, default=60.0,
                        help="Gate height in y [nm]")
    parser.add_argument("--gate-gap", type=float, default=20.0,
                        help="Gap between gates [nm]")

    # Gate voltages
    parser.add_argument("--V1", type=float, default=1.0,
                        help="Gate 1 voltage [V]")
    parser.add_argument("--V2", type=float, default=0.0,
                        help="Gate 2 voltage [V]")
    parser.add_argument("--Vbottom", type=float, default=0.0,
                        help="Bottom voltage [V]")

    # Mesh
    parser.add_argument("--hx", type=float, default=5.0,
                        help="Approximate x/y mesh spacing [nm]")
    parser.add_argument("--nz", type=int, default=23,
                        help="Number of mesh divisions through 23 nm stack")

    parser.add_argument("--p", type=int, default=1,
                        help="Lagrange polynomial degree")

    parser.add_argument("--output", type=str,
                        default="sige_two_gate_3layer",
                        help="Output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    t_total = args.t1 + args.t2 + args.t3

    # ---------------------------------------------------------------------
    # Run summary
    # ---------------------------------------------------------------------

    if rank == 0:
        print("=" * 78)
        print("3-LAYER Si/SiGe TWO-GATE ELECTROSTATIC SIMULATION")
        print("=" * 78)
        print()
        print("DOMAIN")
        print(f"  Lx                  = {args.Lx:.3f} nm")
        print(f"  Ly                  = {args.Ly:.3f} nm")
        print(f"  Lz                  = {t_total:.3f} nm")
        print()
        print("LAYER STACK")
        print(
            f"  Layer 1: SiGe   z = 0 -> {args.t1:.3f} nm"
            f"   eps_r = {args.eps_sige_top}"
        )
        print(
            f"  Layer 2: Si     z = {args.t1:.3f}"
            f" -> {args.t1 + args.t2:.3f} nm"
            f"   eps_r = {args.eps_si}"
        )
        print(
            f"  Layer 3: SiGe   z = {args.t1 + args.t2:.3f}"
            f" -> {t_total:.3f} nm"
            f"   eps_r = {args.eps_sige_bottom}"
        )
        print()
        print("BOUNDARY CONDITIONS")
        print(f"  Gate 1              = {args.V1:.6f} V")
        print(f"  Gate 2              = {args.V2:.6f} V")
        print(f"  Bottom              = {args.Vbottom:.6f} V")
        print("  Remaining surfaces  = zero normal electric flux")
        print()
        print("PHYSICS")
        print("  rho                  = 0")
        print("  equation             = div(eps_r grad(phi)) = 0")
        print()

    # ---------------------------------------------------------------------
    # Mesh
    # ---------------------------------------------------------------------

    nx = max(2, int(np.ceil(args.Lx / args.hx)))
    ny = max(2, int(np.ceil(args.Ly / args.hx)))
    nz = args.nz

    domain = mesh.create_box(
        comm,
        [
            np.array([-args.Lx / 2.0, -args.Ly / 2.0, 0.0]),
            np.array([ args.Lx / 2.0,  args.Ly / 2.0, t_total]),
        ],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )

    tdim = domain.topology.dim
    fdim = tdim - 1

    # ---------------------------------------------------------------------
    # Check interface alignment
    # ---------------------------------------------------------------------

    dz = t_total / nz
    z_interface_1 = args.t1
    z_interface_2 = args.t1 + args.t2

    align1 = abs(z_interface_1 / dz - round(z_interface_1 / dz))
    align2 = abs(z_interface_2 / dz - round(z_interface_2 / dz))

    if rank == 0:
        print("MESH")
        print(f"  nx, ny, nz          = {nx}, {ny}, {nz}")
        print(f"  nominal dx          = {args.Lx / nx:.4f} nm")
        print(f"  nominal dy          = {args.Ly / ny:.4f} nm")
        print(f"  dz                  = {dz:.4f} nm")
        print(f"  polynomial degree p = {args.p}")

        if align1 > 1e-10 or align2 > 1e-10:
            print()
            print("WARNING:")
            print("  One or more dielectric interfaces do not coincide")
            print("  exactly with horizontal mesh planes.")
            print("  For this benchmark, use nz=23, 46, 69, ...")
        print()

    # ---------------------------------------------------------------------
    # Function spaces
    # ---------------------------------------------------------------------

    try:
        V = fem.functionspace(domain, ("Lagrange", args.p))
        Q = fem.functionspace(domain, ("DG", 0))
    except AttributeError:
        # Compatibility with some older DOLFINx releases
        V = fem.FunctionSpace(domain, ("Lagrange", args.p))
        Q = fem.FunctionSpace(domain, ("DG", 0))

    # ---------------------------------------------------------------------
    # Material fields
    # ---------------------------------------------------------------------

    eps_r = fem.Function(Q)
    eps_r.name = "epsilon_r"

    material_id = fem.Function(Q)
    material_id.name = "material_id"

    z1 = args.t1
    z2 = args.t1 + args.t2

    def epsilon_expression(x):
        values = np.full(x.shape[1], args.eps_sige_bottom, dtype=PETSc.ScalarType)
        values[x[2] < z2] = args.eps_si
        values[x[2] < z1] = args.eps_sige_top
        return values

    def material_expression(x):
        # 1 = top SiGe
        # 2 = Si
        # 3 = bottom SiGe
        values = np.full(x.shape[1], 3.0, dtype=PETSc.ScalarType)
        values[x[2] < z2] = 2.0
        values[x[2] < z1] = 1.0
        return values

    eps_r.interpolate(epsilon_expression)
    material_id.interpolate(material_expression)

    # ---------------------------------------------------------------------
    # Gate geometry
    #
    # Two gates centered symmetrically around x = 0.
    #
    # gate 1:
    #     x = -(gap/2 + width) ... -gap/2
    #
    # gate 2:
    #     x = +gap/2 ... +(gap/2 + width)
    #
    # both:
    #     |y| <= gate_height/2
    # ---------------------------------------------------------------------

    half_gap = args.gate_gap / 2.0

    gate1_xmin = -(half_gap + args.gate_width)
    gate1_xmax = -half_gap

    gate2_xmin = half_gap
    gate2_xmax = half_gap + args.gate_width

    gate_ymin = -args.gate_height / 2.0
    gate_ymax =  args.gate_height / 2.0

    tol = 1e-8

    def gate1_marker(x):
        return (
            np.isclose(x[2], 0.0, atol=tol)
            & (x[0] >= gate1_xmin - tol)
            & (x[0] <= gate1_xmax + tol)
            & (x[1] >= gate_ymin - tol)
            & (x[1] <= gate_ymax + tol)
        )

    def gate2_marker(x):
        return (
            np.isclose(x[2], 0.0, atol=tol)
            & (x[0] >= gate2_xmin - tol)
            & (x[0] <= gate2_xmax + tol)
            & (x[1] >= gate_ymin - tol)
            & (x[1] <= gate_ymax + tol)
        )

    def bottom_marker(x):
        return np.isclose(x[2], t_total, atol=tol)

    gate1_facets = mesh.locate_entities_boundary(domain, fdim, gate1_marker)
    gate2_facets = mesh.locate_entities_boundary(domain, fdim, gate2_marker)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_marker)

    n_g1 = comm.allreduce(len(gate1_facets), op=MPI.SUM)
    n_g2 = comm.allreduce(len(gate2_facets), op=MPI.SUM)
    n_bot = comm.allreduce(len(bottom_facets), op=MPI.SUM)

    if rank == 0:
        print("GATE GEOMETRY")
        print(
            f"  Gate 1 x-range      = "
            f"[{gate1_xmin:.1f}, {gate1_xmax:.1f}] nm"
        )
        print(
            f"  Gate 2 x-range      = "
            f"[{gate2_xmin:.1f}, {gate2_xmax:.1f}] nm"
        )
        print(
            f"  Gate y-range        = "
            f"[{gate_ymin:.1f}, {gate_ymax:.1f}] nm"
        )
        print()
        print("BOUNDARY FACET CHECK")
        print(f"  Gate 1 facets       = {n_g1}")
        print(f"  Gate 2 facets       = {n_g2}")
        print(f"  Bottom facets       = {n_bot}")
        print()

    if n_g1 == 0:
        raise RuntimeError("Gate 1 has zero boundary facets.")

    if n_g2 == 0:
        raise RuntimeError("Gate 2 has zero boundary facets.")

    if n_bot == 0:
        raise RuntimeError("Bottom boundary has zero facets.")

    # ---------------------------------------------------------------------
    # Dirichlet BCs
    # ---------------------------------------------------------------------

    dofs_g1 = fem.locate_dofs_topological(V, fdim, gate1_facets)
    dofs_g2 = fem.locate_dofs_topological(V, fdim, gate2_facets)
    dofs_bot = fem.locate_dofs_topological(V, fdim, bottom_facets)

    bc_g1 = fem.dirichletbc(
        PETSc.ScalarType(args.V1),
        dofs_g1,
        V
    )

    bc_g2 = fem.dirichletbc(
        PETSc.ScalarType(args.V2),
        dofs_g2,
        V
    )

    bc_bot = fem.dirichletbc(
        PETSc.ScalarType(args.Vbottom),
        dofs_bot,
        V
    )

    bcs = [bc_g1, bc_g2, bc_bot]

    # ---------------------------------------------------------------------
    # Weak form
    #
    # div(eps_r grad(phi)) = 0
    #
    # Weak form:
    #
    # integral eps_r grad(phi).grad(v) dx = 0
    # ---------------------------------------------------------------------

    phi_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (
        eps_r
        * ufl.inner(ufl.grad(phi_trial), ufl.grad(v))
        * ufl.dx
    )

    L = PETSc.ScalarType(0.0) * v * ufl.dx

    # ---------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    try:
        problem = fem.petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="sige_laplace_",
        )
    except TypeError:
        # Compatibility with DOLFINx releases where prefix is not required
        problem = fem.petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options=petsc_options,
        )

    phi = problem.solve()
    phi.name = "phi"

    phi.x.scatter_forward()

    # ---------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------

    local_min = np.min(np.real(phi.x.array))
    local_max = np.max(np.real(phi.x.array))

    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)

    local_dofs = V.dofmap.index_map.size_local
    global_dofs = comm.allreduce(local_dofs, op=MPI.SUM)

    iterations = -1
    reason = None

    if hasattr(problem, "solver"):
        try:
            iterations = problem.solver.getIterationNumber()
            reason = problem.solver.getConvergedReason()
        except Exception:
            pass

    if rank == 0:
        print("=" * 78)
        print("SOLVER RESULTS")
        print("=" * 78)
        print(f"  Global DOFs         = {global_dofs:,}")
        print(f"  min(phi)            = {global_min:.12e} V")
        print(f"  max(phi)            = {global_max:.12e} V")

        if iterations >= 0:
            print(f"  KSP iterations      = {iterations}")

        if reason is not None:
            print(f"  PETSc reason        = {reason}")

        print()
        print("Expected sanity check:")
        print(
            f"  potential should remain approximately between "
            f"{min(args.V1, args.V2, args.Vbottom):.3f} and "
            f"{max(args.V1, args.V2, args.Vbottom):.3f} V"
        )
        print()

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------

    if rank == 0:
        os.makedirs(args.output, exist_ok=True)

    comm.barrier()

    solution_file = os.path.join(args.output, "potential.xdmf")

    with io.XDMFFile(comm, solution_file, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)

    material_file = os.path.join(args.output, "materials.xdmf")

    with io.XDMFFile(comm, material_file, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(material_id)
        xdmf.write_function(eps_r)

    # ---------------------------------------------------------------------
    # Save run parameters
    # ---------------------------------------------------------------------

    if rank == 0:
        summary_file = os.path.join(args.output, "run_summary.txt")

        with open(summary_file, "w") as f:
            f.write("3-layer Si/SiGe two-gate electrostatic benchmark\n")
            f.write("=" * 70 + "\n\n")

            f.write("Domain\n")
            f.write(f"Lx_nm = {args.Lx}\n")
            f.write(f"Ly_nm = {args.Ly}\n")
            f.write(f"Lz_nm = {t_total}\n\n")

            f.write("Layers\n")
            f.write(
                f"1 SiGe: 0 to {args.t1} nm, "
                f"eps_r={args.eps_sige_top}\n"
            )
            f.write(
                f"2 Si: {args.t1} to {args.t1 + args.t2} nm, "
                f"eps_r={args.eps_si}\n"
            )
            f.write(
                f"3 SiGe: {args.t1 + args.t2} to {t_total} nm, "
                f"eps_r={args.eps_sige_bottom}\n\n"
            )

            f.write("Gates\n")
            f.write(
                f"Gate1: V={args.V1} V, "
                f"x=[{gate1_xmin},{gate1_xmax}] nm, "
                f"y=[{gate_ymin},{gate_ymax}] nm\n"
            )
            f.write(
                f"Gate2: V={args.V2} V, "
                f"x=[{gate2_xmin},{gate2_xmax}] nm, "
                f"y=[{gate_ymin},{gate_ymax}] nm\n\n"
            )

            f.write("Boundary conditions\n")
            f.write(f"Bottom = {args.Vbottom} V\n")
            f.write("Other surfaces = homogeneous Neumann\n\n")

            f.write("Mesh\n")
            f.write(f"nx = {nx}\n")
            f.write(f"ny = {ny}\n")
            f.write(f"nz = {nz}\n")
            f.write(f"p = {args.p}\n")
            f.write(f"dz_nm = {dz}\n\n")

            f.write("Boundary facet counts\n")
            f.write(f"Gate1 = {n_g1}\n")
            f.write(f"Gate2 = {n_g2}\n")
            f.write(f"Bottom = {n_bot}\n\n")

            f.write("Results\n")
            f.write(f"DOFs = {global_dofs}\n")
            f.write(f"phi_min_V = {global_min:.16e}\n")
            f.write(f"phi_max_V = {global_max:.16e}\n")

        print("OUTPUT")
        print(f"  Potential           = {solution_file}")
        print(f"  Materials           = {material_file}")
        print(f"  Run summary         = {summary_file}")
        print()
        print("Simulation complete.")
        print("=" * 78)


if __name__ == "__main__":
    main()
