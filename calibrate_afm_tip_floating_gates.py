#!/usr/bin/env python3

"""
AFM-tip driven Si/SiGe electrostatics -- FLOATING GATES variant.

Same geometry/materials/solver as calibrate_afm_tip_vs_leah.py (that script is
imported from, not duplicated, and is NOT modified by this file), but with one
physical change: Probe1/Probe2 (the two internal circular disks at z=35nm and
z=48nm, facet tags FACET_PROBE1=103 / FACET_PROBE2=104) are now treated as REAL
FLOATING CONDUCTORS -- self-consistent equipotential surfaces carrying ZERO NET
CHARGE -- instead of being left as plain passive/Neumann surfaces (electrically
inert, indistinguishable from the surrounding Si). This is what an unbiased
metal gate actually does physically.

Why superposition instead of iteration
---------------------------------------
This is linear electrostatics (div(eps_r grad(phi)) = 0, no volume charge), so
a floating conductor's self-consistent voltage can be obtained EXACTLY via
superposition instead of guess-and-check iteration. For FIXED geometry/mesh/
materials and FIXED (V_tip, V_bottom), solve three auxiliary problems on the
SAME mesh/function-space (only the Dirichlet values differ):

    phi_A : tip=V_tip,  bottom=V_bottom, gate1=0, gate2=0   (gates grounded)
    phi_B : tip=0,      bottom=0,        gate1=1, gate2=0   (unit gate1 response)
    phi_C : tip=0,      bottom=0,        gate1=0, gate2=1   (unit gate2 response)

phi_B and phi_C do NOT depend on V_tip, so they are solved ONCE and reused
across every requested tip voltage. For any (V_g1, V_g2):

    phi = phi_A + V_g1 * phi_B + V_g2 * phi_C                          (exact, by linearity)

The net charge flux through a gate surface, Q = integral(eps_r * grad(phi).n dS),
is itself linear in the Dirichlet data, so with Q1_A, Q2_A (from phi_A) and the
mesh-only constants Q1_B, Q2_B, Q1_C, Q2_C (from phi_B, phi_C):

    Q1(V_g1, V_g2) = Q1_A + V_g1*Q1_B + V_g2*Q1_C
    Q2(V_g1, V_g2) = Q2_A + V_g1*Q2_B + V_g2*Q2_C

The floating condition is Q1=Q2=0 -- a trivial 2x2 linear system solved with
numpy.linalg.solve for the exact self-consistent (V_g1*, V_g2*). The final
combined field is re-checked directly (flux re-integrated on the ACTUAL
combined phi, not just algebraically) as an independent sanity check on the
whole superposition procedure.

Mesh/geometry is generated ONCE per script invocation and reused across every
tip-voltage case requested via --tip-voltage (accepts multiple values), so
running e.g. --tip-voltage 0.0 1.0 -1.0 does 2 shared unit-gate solves + 3
tip-voltage baseline solves = 5 total linear solves on ONE mesh, instead of
regenerating the (expensive, ~467k-cell) mesh three times.

Output format matches calibrate_afm_tip_vs_leah.py: one combined
sige_afm_tip.xdmf/.h5 per run directory (phi_V, relative_permittivity,
material_id, cell_tags, facet_tags), plus run_results.json and centerline.csv,
with the fitted V_g1*/V_g2* and floating-condition sanity residuals recorded
in run_results.json under "floating_gates".
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import json
import os

import numpy as np
import ufl

from dolfinx import fem, geometry, io

from calibrate_afm_tip_vs_leah import (
    FACET_BOTTOM,
    FACET_PROBE1,
    FACET_PROBE2,
    FACET_TIP,
    build_geometry,
    make_material_fields,
    next_run_directory,
    root_print,
)

try:
    from dolfinx.fem.petsc import LinearProblem
except ImportError:
    from dolfinx.fem import petsc as fem_petsc
    LinearProblem = fem_petsc.LinearProblem


# =====================================================================
# Arguments
# =====================================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="AFM-tip driven Si/SiGe electrostatics -- floating gates"
    )

    p.add_argument("--lx", type=float, default=300.0)
    p.add_argument("--ly", type=float, default=300.0)
    p.add_argument("--air-height", type=float, default=260.0)

    p.add_argument("--eps-air", type=float, default=1.0)
    p.add_argument("--eps-si", type=float, default=11.7)
    p.add_argument("--eps-sige", type=float, default=12.0)

    p.add_argument("--gap", type=float, default=30.0)
    p.add_argument("--tip-radius", type=float, default=20.0)
    p.add_argument("--cone-height", type=float, default=100.0)
    p.add_argument("--shank-radius", type=float, default=60.0)
    p.add_argument("--cone-half-angle-deg", type=float, default=None)
    p.add_argument("--shaft-radius", type=float, default=60.0)
    p.add_argument("--shaft-height", type=float, default=60.0)

    p.add_argument(
        "--tip-voltage",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "One or more AFM tip Dirichlet voltages. The mesh/function-space "
            "is built ONCE and reused for every value given here, plus the 2 "
            "shared unit-gate auxiliary solves -- pass all polarity cases in "
            "one invocation to avoid redundant mesh generation."
        ),
    )

    p.add_argument("--bottom-voltage", type=float, default=-4.4)

    p.add_argument("--probe1-radius", type=float, default=10.0)
    p.add_argument("--probe2-radius", type=float, default=10.0)

    p.add_argument("--h-apex", type=float, default=1.0)
    p.add_argument("--h-device", type=float, default=2.0)
    p.add_argument("--h-near", type=float, default=5.0)
    p.add_argument("--h-bottom", type=float, default=100.0)
    p.add_argument("--degree", type=int, default=1)

    p.add_argument(
        "--results-root",
        type=str,
        default="notebooks/afm_vs_leah_compare",
    )

    p.add_argument(
        "--output",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One output folder per --tip-voltage value, same order. If "
            "omitted, next available runN folders are auto-assigned."
        ),
    )

    return p.parse_args()


# =====================================================================
# Solve helpers
# =====================================================================

def make_bc(domain, V, facet_tags, marker, voltage):

    fdim = domain.topology.dim - 1

    facets = facet_tags.find(marker)

    count = domain.comm.allreduce(len(facets), op=MPI.SUM)

    if count == 0:
        raise RuntimeError(f"Facet tag {marker} is empty.")

    dofs = fem.locate_dofs_topological(V, fdim, facets)

    value = fem.Constant(domain, PETSc.ScalarType(voltage))

    return fem.dirichletbc(value, dofs, V)


def solve_dirichlet(domain, V, epsilon, facet_tags, voltages, label):

    """voltages: dict {FACET_TIP: v, FACET_BOTTOM: v, FACET_PROBE1: v, FACET_PROBE2: v}"""

    bcs = [
        make_bc(domain, V, facet_tags, marker, v)
        for marker, v in voltages.items()
    ]

    u = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)

    dx = ufl.Measure("dx", domain=domain)

    zero = fem.Constant(domain, PETSc.ScalarType(0.0))

    lhs = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v_)) * dx
    rhs = zero * v_ * dx

    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 3000,
        "ksp_error_if_not_converged": True,
    }

    try:
        problem = LinearProblem(
            lhs, rhs, bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix=f"sige_afm_{label}_",
        )
    except TypeError:
        problem = LinearProblem(lhs, rhs, bcs=bcs, petsc_options=petsc_options)

    root_print(domain.comm, f"  solving '{label}' ...")

    phi = problem.solve()
    phi.x.scatter_forward()

    return phi


def gate_charge(domain, epsilon, phi, facet_tags, marker):

    """Net charge flux Q = integral(eps_r * grad(phi).n dS) through the gate
    surface tagged `marker`, via a UFL boundary integral (exterior facet
    integral tagged with the mesh's own facet_tags meshtag)."""

    n = ufl.FacetNormal(domain)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    form = fem.form(epsilon * ufl.dot(ufl.grad(phi), n) * ds(marker))

    local = fem.assemble_scalar(form)

    return domain.comm.allreduce(local, op=MPI.SUM)


# =====================================================================
# Probe statistics (mirrors calibrate_afm_tip_vs_leah.analyze_probe)
# =====================================================================

def analyze_probe(domain, V, phi, facet_tags, marker, name):

    fdim = domain.topology.dim - 1

    facets = facet_tags.find(marker)

    global_facets = domain.comm.allreduce(len(facets), op=MPI.SUM)

    if global_facets == 0:
        return {"name": name, "facets": 0, "dofs": 0, "min": np.nan, "max": np.nan, "mean_nodal": np.nan}

    dofs = fem.locate_dofs_topological(V, fdim, facets)

    values = np.real(phi.x.array[dofs])

    if len(values) > 0:
        local_min, local_max = np.min(values), np.max(values)
        local_sum, local_count = np.sum(values), len(values)
    else:
        local_min, local_max = np.inf, -np.inf
        local_sum, local_count = 0.0, 0

    global_min = domain.comm.allreduce(local_min, op=MPI.MIN)
    global_max = domain.comm.allreduce(local_max, op=MPI.MAX)
    global_sum = domain.comm.allreduce(local_sum, op=MPI.SUM)
    global_count = domain.comm.allreduce(local_count, op=MPI.SUM)

    mean = global_sum / global_count if global_count > 0 else np.nan

    return {
        "name": name,
        "facets": int(global_facets),
        "dofs": int(global_count),
        "min": float(global_min),
        "max": float(global_max),
        "mean_nodal": float(mean),
    }


# =====================================================================
# Output writer (mirrors calibrate_afm_tip_vs_leah.main's output block)
# =====================================================================

def write_case_outputs(
    output_dir, comm, domain, phi, epsilon, material_id, cell_tags, facet_tags,
    a, tip_voltage, z_probe1, z_probe2, z_bottom, floating_info,
):

    tdim = domain.topology.dim

    values = np.real(phi.x.array)
    local_min = np.min(values) if len(values) else np.inf
    local_max = np.max(values) if len(values) else -np.inf
    phi_min = comm.allreduce(local_min, op=MPI.MIN)
    phi_max = comm.allreduce(local_max, op=MPI.MAX)

    V = phi.function_space

    probe1 = analyze_probe(domain, V, phi, facet_tags, FACET_PROBE1, "Probe 1")
    probe2 = analyze_probe(domain, V, phi, facet_tags, FACET_PROBE2, "Probe 2")

    global_cells = domain.topology.index_map(tdim).size_global
    global_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    combined_path = os.path.join(output_dir, "sige_afm_tip.xdmf")

    with io.XDMFFile(comm, combined_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        xdmf.write_function(epsilon)
        xdmf.write_function(material_id)
        try:
            xdmf.write_meshtags(cell_tags, domain.geometry)
            xdmf.write_meshtags(facet_tags, domain.geometry)
        except TypeError:
            xdmf.write_meshtags(cell_tags)
            xdmf.write_meshtags(facet_tags)

    root_print(comm, f"Wrote {combined_path}")

    # ---------------- centerline ----------------

    z_fine = np.arange(-a.air_height, 100.0, 1.0)
    z_coarse = np.arange(100.0, z_bottom, 20.0)
    z_line = np.concatenate([z_fine, z_coarse, [z_bottom]])

    points = np.zeros((len(z_line), 3))
    points[:, 2] = z_line

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    phi_line_local = np.zeros(len(z_line))
    found_local = np.zeros(len(z_line))
    local_cells, local_idx = [], []

    for i in range(len(z_line)):
        links_i = colliding.links(i)
        if len(links_i) > 0:
            local_idx.append(i)
            local_cells.append(links_i[0])

    if local_idx:
        vals = phi.eval(points[local_idx], np.array(local_cells, dtype=np.int32))[:, 0]
        for k, gi in enumerate(local_idx):
            phi_line_local[gi] = np.real(vals[k])
            found_local[gi] = 1.0

    phi_line_sum = comm.allreduce(phi_line_local, op=MPI.SUM)
    found_sum = comm.allreduce(found_local, op=MPI.SUM)

    phi_line = np.where(found_sum > 0, phi_line_sum / np.maximum(found_sum, 1.0), np.nan)

    if comm.rank == 0:
        centerline_path = os.path.join(output_dir, "centerline.csv")
        with open(centerline_path, "w") as f:
            f.write("z_nm,phi_V\n")
            for zval, pval in zip(z_line, phi_line):
                if np.isfinite(pval):
                    f.write(f"{zval:.4f},{pval:.6e}\n")
        root_print(comm, f"Wrote {centerline_path}")

    # ---------------- run_results.json ----------------

    if comm.rank == 0:

        results = {
            "output": output_dir,
            "geometry_nm": {
                "lx": a.lx, "ly": a.ly, "air_height": a.air_height,
                "tip_gap": a.gap, "tip_radius": a.tip_radius,
                "cone_height": a.cone_height, "shank_radius": a.shank_radius,
                "cone_half_angle_deg_input": a.cone_half_angle_deg,
                "cone_half_angle_deg_effective": float(np.degrees(np.arctan(
                    (a.shank_radius - a.tip_radius) / a.cone_height
                ))),
                "shaft_radius": a.shaft_radius, "shaft_height": a.shaft_height,
                "probe1_z": z_probe1, "probe2_z": z_probe2,
                "probe1_radius": a.probe1_radius, "probe2_radius": a.probe2_radius,
                "bottom_z": z_bottom,
            },
            "voltages_V": {
                "AFM_tip": tip_voltage,
                "bottom_back_gate": a.bottom_voltage,
                "probe1": floating_info["V_g1_star"],
                "probe2": floating_info["V_g2_star"],
            },
            "floating_gates": floating_info,
            "materials": {"eps_air": a.eps_air, "eps_si": a.eps_si, "eps_sige": a.eps_sige},
            "mesh": {
                "cells": int(global_cells), "dofs": int(global_dofs), "degree": a.degree,
                "h_apex_nm": a.h_apex, "h_device_nm": a.h_device,
                "h_near_nm": a.h_near, "h_bottom_nm": a.h_bottom,
            },
            "solution": {
                "phi_min_V": float(phi_min), "phi_max_V": float(phi_max),
                "probe1": probe1, "probe2": probe2,
            },
        }

        with open(os.path.join(output_dir, "run_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        root_print(comm, f"Wrote {os.path.join(output_dir, 'run_results.json')}")


# =====================================================================
# Main
# =====================================================================

def main():

    a = parse_args()
    comm = MPI.COMM_WORLD

    tip_voltages = a.tip_voltage
    n_cases = len(tip_voltages)

    # -------------------------------------------------------------
    # Assign one output directory per case, up front (rank 0), so
    # numbering is deterministic and collision-free even though the
    # mesh/solves for all cases happen in this single process.
    # -------------------------------------------------------------

    if comm.rank == 0:

        if a.output is not None:
            if len(a.output) != n_cases:
                raise RuntimeError(
                    f"--output given {len(a.output)} values but --tip-voltage "
                    f"has {n_cases} values; they must match 1:1."
                )
            outputs = list(a.output)
        else:
            outputs = []
            for _ in range(n_cases):
                out = next_run_directory(a.results_root)
                os.makedirs(out, exist_ok=False)
                outputs.append(out)

        for out in outputs:
            os.makedirs(out, exist_ok=True)

    else:
        outputs = None

    outputs = comm.bcast(outputs, root=0)
    comm.barrier()

    root_print(comm, "")
    root_print(comm, "=" * 78)
    root_print(comm, "AFM-TIP DRIVEN Si/SiGe ELECTROSTATICS -- FLOATING GATES")
    root_print(comm, "=" * 78)
    root_print(comm, f"Tip voltages  : {tip_voltages}")
    root_print(comm, f"Output dirs   : {outputs}")
    root_print(comm, f"Bottom voltage: {a.bottom_voltage} V")
    root_print(comm, "Probe1/Probe2 : FLOATING (self-consistent, zero net charge)")

    # -------------------------------------------------------------
    # Build mesh ONCE
    # -------------------------------------------------------------

    (domain, cell_tags, facet_tags, z_probe1, z_probe2, z_bottom) = build_geometry(a, comm)

    if cell_tags is None:
        raise RuntimeError("Cell tags did not survive Gmsh conversion.")
    if facet_tags is None:
        raise RuntimeError("Facet tags did not survive Gmsh conversion.")

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)

    try:
        V = fem.functionspace(domain, ("Lagrange", a.degree))
    except AttributeError:
        V = fem.FunctionSpace(domain, ("CG", a.degree))

    epsilon, material_id = make_material_fields(domain, cell_tags, a)

    global_cells = domain.topology.index_map(tdim).size_global
    root_print(comm, f"Mesh cells: {global_cells:,}  (built once, reused for all {n_cases} cases)")

    # -------------------------------------------------------------
    # Shared unit-gate auxiliary solves (independent of V_tip) --
    # computed ONCE and reused across every tip-voltage case.
    # -------------------------------------------------------------

    root_print(comm, "")
    root_print(comm, "Solving shared unit-gate auxiliary problems (phi_B, phi_C)...")

    phi_B = solve_dirichlet(
        domain, V, epsilon, facet_tags,
        {FACET_TIP: 0.0, FACET_BOTTOM: 0.0, FACET_PROBE1: 1.0, FACET_PROBE2: 0.0},
        "phiB_unit_gate1",
    )

    phi_C = solve_dirichlet(
        domain, V, epsilon, facet_tags,
        {FACET_TIP: 0.0, FACET_BOTTOM: 0.0, FACET_PROBE1: 0.0, FACET_PROBE2: 1.0},
        "phiC_unit_gate2",
    )

    Q1_B = gate_charge(domain, epsilon, phi_B, facet_tags, FACET_PROBE1)
    Q2_B = gate_charge(domain, epsilon, phi_B, facet_tags, FACET_PROBE2)
    Q1_C = gate_charge(domain, epsilon, phi_C, facet_tags, FACET_PROBE1)
    Q2_C = gate_charge(domain, epsilon, phi_C, facet_tags, FACET_PROBE2)

    root_print(comm, f"  Q1_B={Q1_B:.6e}  Q2_B={Q2_B:.6e}")
    root_print(comm, f"  Q1_C={Q1_C:.6e}  Q2_C={Q2_C:.6e}")

    M = np.array([[Q1_B, Q1_C], [Q2_B, Q2_C]])

    # -------------------------------------------------------------
    # Per tip-voltage: baseline solve (gates grounded), fit floating
    # voltages via the 2x2 linear system, build the final field by
    # exact superposition, sanity-check, write outputs.
    # -------------------------------------------------------------

    for case_idx, (tip_voltage, output_dir) in enumerate(zip(tip_voltages, outputs)):

        root_print(comm, "")
        root_print(comm, "-" * 78)
        root_print(comm, f"CASE {case_idx + 1}/{n_cases}: V_tip = {tip_voltage} V  ->  {output_dir}")
        root_print(comm, "-" * 78)

        phi_A = solve_dirichlet(
            domain, V, epsilon, facet_tags,
            {FACET_TIP: tip_voltage, FACET_BOTTOM: a.bottom_voltage, FACET_PROBE1: 0.0, FACET_PROBE2: 0.0},
            f"phiA_case{case_idx}",
        )

        Q1_A = gate_charge(domain, epsilon, phi_A, facet_tags, FACET_PROBE1)
        Q2_A = gate_charge(domain, epsilon, phi_A, facet_tags, FACET_PROBE2)

        rhs = np.array([-Q1_A, -Q2_A])
        V_g1_star, V_g2_star = np.linalg.solve(M, rhs)

        root_print(comm, f"  Q1_A={Q1_A:.6e}  Q2_A={Q2_A:.6e}")
        root_print(comm, f"  fitted V_g1* = {V_g1_star:.8f} V   V_g2* = {V_g2_star:.8f} V")

        # Exact superposition
        phi_final = fem.Function(V)
        phi_final.x.array[:] = (
            phi_A.x.array + V_g1_star * phi_B.x.array + V_g2_star * phi_C.x.array
        )
        phi_final.name = "phi_V"
        phi_final.x.scatter_forward()

        # Independent sanity check: re-integrate flux on the ACTUAL combined
        # field directly (not via the superposition algebra) -- verifies the
        # linear-algebra step itself, not just that the 2x2 solve succeeded.
        Q1_check = gate_charge(domain, epsilon, phi_final, facet_tags, FACET_PROBE1)
        Q2_check = gate_charge(domain, epsilon, phi_final, facet_tags, FACET_PROBE2)

        root_print(comm, f"  SANITY CHECK (flux on final combined phi, should be ~0):")
        root_print(comm, f"    Q1(phi_final) = {Q1_check:.6e}")
        root_print(comm, f"    Q2(phi_final) = {Q2_check:.6e}")

        floating_info = {
            "V_g1_star": float(V_g1_star),
            "V_g2_star": float(V_g2_star),
            "Q1_A": float(Q1_A), "Q2_A": float(Q2_A),
            "Q1_B": float(Q1_B), "Q2_B": float(Q2_B),
            "Q1_C": float(Q1_C), "Q2_C": float(Q2_C),
            "sanity_Q1_residual": float(Q1_check),
            "sanity_Q2_residual": float(Q2_check),
            "method": "exact linear superposition (3 aux solves + 2x2 linalg.solve), "
                      "not iterative",
        }

        write_case_outputs(
            output_dir, comm, domain, phi_final, epsilon, material_id, cell_tags, facet_tags,
            a, tip_voltage, z_probe1, z_probe2, z_bottom, floating_info,
        )

        root_print(comm, f"CASE COMPLETE: {output_dir}")

    root_print(comm, "")
    root_print(comm, "=" * 78)
    root_print(comm, "ALL CASES COMPLETE")
    root_print(comm, "=" * 78)
    for tv, out in zip(tip_voltages, outputs):
        root_print(comm, f"  V_tip={tv:>6}  ->  {out}")


if __name__ == "__main__":
    main()
