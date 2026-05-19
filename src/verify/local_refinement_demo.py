#!/usr/bin/env python3
"""
Demo: solve Laplace's equation on a locally refined mesh.

Two refinement zones are defined:
  - Zone 1 (gate region): fine mesh near the top surface where gate BCs live
  - Zone 2 (interface): finer mesh at a dielectric interface mid-height

Run inside the DOLFINx Docker container:
    mpirun -n 4 python3 src/verify/local_refinement_demo.py

Outputs:
    local_refinement_demo.xdmf / .h5   — mesh + solution + region marker
"""
from __future__ import annotations
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
import ufl

from poisson.refinement import RefinementBox, build_refined_mesh_3d, mark_refinement_regions
from poisson.fem_solve import assemble_solve_poisson
from poisson.io_utils import write_field_xdmf

comm = MPI.COMM_WORLD

# ── Domain ────────────────────────────────────────────────────────────────────
Lx, Ly, Lz = 500e-9, 500e-9, 150e-9
h_coarse    = 20e-9

# ── Refinement zones ──────────────────────────────────────────────────────────
# Zone 1: gate area — very fine, centred near the top surface (z ≈ 0)
gate_box = RefinementBox(
    cx=0.0,  cy=0.0,  cz=15e-9,
    lx=80e-9, ly=80e-9, lz=30e-9,
    h_fine=4e-9,
)

# Zone 2: mid-height interface — moderately fine
interface_box = RefinementBox(
    cx=0.0,   cy=0.0,   cz=75e-9,
    lx=300e-9, ly=300e-9, lz=20e-9,
    h_fine=10e-9,
)

# ── Build mesh ────────────────────────────────────────────────────────────────
mesh = build_refined_mesh_3d(
    comm, Lx, Ly, Lz, h_coarse,
    boxes=[gate_box, interface_box],
)

region = mark_refinement_regions(mesh, [gate_box, interface_box])

# ── Function space & BCs ──────────────────────────────────────────────────────
try:
    V = fem.functionspace(mesh, ("Lagrange", 1))
except AttributeError:
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

Z = mesh.geometry.x[:, 2]
z_min = comm.allreduce(float(np.min(Z)), op=MPI.MIN)
z_max = comm.allreduce(float(np.max(Z)), op=MPI.MAX)

top_facets = fem.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[2], z_min, atol=1e-10)
)
bot_facets = fem.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[2], z_max, atol=1e-10)
)

dofs_top = fem.locate_dofs_topological(V, fdim, top_facets)
dofs_bot = fem.locate_dofs_topological(V, fdim, bot_facets)

bcs = [
    fem.dirichletbc(PETSc.ScalarType(1.0), dofs_top, V),
    fem.dirichletbc(PETSc.ScalarType(0.0), dofs_bot, V),
]

# ── Solve ─────────────────────────────────────────────────────────────────────
try:
    DG0 = fem.functionspace(mesh, ("DG", 0))
except AttributeError:
    DG0 = fem.FunctionSpace(mesh, ("DG", 0))

eps = fem.Function(DG0, name="epsilon")
eps.x.array[:] = 11.7 * 8.8541878128e-12   # silicon permittivity everywhere

rho = fem.Constant(mesh, PETSc.ScalarType(0.0))

phi, ksp = assemble_solve_poisson(mesh, V, eps, rho, bcs=bcs)

# ── Write output ──────────────────────────────────────────────────────────────
write_field_xdmf("local_refinement_demo.xdmf", mesh, phi, region)

if comm.rank == 0:
    tdim = mesh.topology.dim
    n_cells = comm.allreduce(mesh.topology.index_map(tdim).size_local, op=MPI.SUM)
    print(f"Total cells : {n_cells}")
    print(f"Gate cells  : {int(np.sum(region.x.array == 1.0))}")
    print(f"Iface cells : {int(np.sum(region.x.array == 2.0))}")
    print(f"Coarse cells: {int(np.sum(region.x.array == 0.0))}")
    print("Wrote: local_refinement_demo.xdmf")
    print("Open in ParaView → color by 'refinement_region' to see zones")
