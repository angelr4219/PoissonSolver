#!/usr/bin/env python3
"""
Embedded-gate Laplace/Poisson benchmark with spatially varying permittivity.

Primary mode (recommended):
- Load a gmsh .msh with:
    - gate cavity surfaces tagged 101, 102, ...
    - outer bottom surface tagged 10 (z=H) (optional but recommended)
- Apply Dirichlet BCs to gate cavity surfaces via facet_tags.find(tag).

Equation (default Laplace):
    -∇·(ε(x) ∇phi) = rho

Outputs:
    <basename>.xdmf        : phi (CG1) + eps_r (DG0)
    <basename>_tags.xdmf   : facet tags for debugging in ParaView
"""

import json
import argparse
import numpy as np
from mpi4py import MPI

from dolfinx import fem, io
from dolfinx.io import gmshio
from ufl import TrialFunction, TestFunction, grad, inner, dx

from petsc4py import PETSc


def load_msh(msh_path: str):
    domain, cell_tags, facet_tags = gmshio.read_from_msh(msh_path, MPI.COMM_WORLD, gdim=3)
    return domain, cell_tags, facet_tags


def tagged_dirichlet_bc(V: fem.FunctionSpace, facet_tags, tag: int, value: float):
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(int(tag))
    if facets.size == 0:
        if V.mesh.comm.rank == 0:
            print(f"[WARN] No facets found for tag={tag} (Dirichlet {value} V not applied)")
        return None
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(PETSc.ScalarType(value), dofs, V)


def gate_cavity_bcs(V: fem.FunctionSpace, facet_tags, gate_tag_to_V: dict):
    bcs = []
    for tag_str, Vg in gate_tag_to_V.items():
        bc = tagged_dirichlet_bc(V, facet_tags, int(tag_str), float(Vg))
        if bc is not None:
            bcs.append(bc)
    return bcs


def build_eps_r_dg0(domain, eps1r: float, eps2r: float, z_interface: float):
    """
    Piecewise eps_r(z):
        eps1r for z < z_interface
        eps2r for z >= z_interface

    If eps2r == eps1r, it's constant.
    """
    Q = fem.functionspace(domain, ("DG", 0))
    eps_r = fem.Function(Q, name="eps_r")

    # Evaluate at cell midpoints for DG0
    x = domain.geometry.x
    # For DG0, dolfinx stores one dof per cell; we can evaluate by using geometry + dof coords approach.
    # Easiest robust method: use Expression interpolation onto DG0 is not necessary; do manual fill via dof coords:
    dof_x = Q.tabulate_dof_coordinates().reshape(-1, domain.geometry.dim)
    z = dof_x[:, 2]
    vals = np.where(z < z_interface, eps1r, eps2r).astype(PETSc.ScalarType)
    eps_r.x.array[:] = vals
    return eps_r


def build_rho_dg0(domain, rho0: float):
    """
    Optional simple uniform rho in DG0 (for quick Poisson testing).
    """
    Q = fem.functionspace(domain, ("DG", 0))
    rho = fem.Function(Q, name="rho")
    rho.x.array[:] = PETSc.ScalarType(rho0)
    return rho


def solve_poisson(domain, facet_tags, deg: int,
                  gate_tag_to_V: dict,
                  eps1r: float, eps2r: float, z_interface: float,
                  rho0: float,
                  outprefix: str,
                  ground_bottom_tag: int = 10,
                  also_ground_sides: bool = False):
    V = fem.functionspace(domain, ("Lagrange", deg))

    # ---- BCs ----
    bcs = []

    # Gate cavity Dirichlet BCs (tags 101,102,...)
    bcs.extend(gate_cavity_bcs(V, facet_tags, gate_tag_to_V))

    # Ground bottom outer surface (tag 10 by default)
    bc_bottom = tagged_dirichlet_bc(V, facet_tags, ground_bottom_tag, 0.0)
    if bc_bottom is not None:
        bcs.append(bc_bottom)
    else:
        # If bottom tag missing, warn loudly (still solvable if gate BCs exist, but you usually want bottom grounded too)
        if domain.comm.rank == 0:
            print(f"[WARN] bottom tag {ground_bottom_tag} not found. Proceeding with only gate BCs.")

    # Optionally ground sides (tag 12 from gmsh script)
    if also_ground_sides:
        bc_sides = tagged_dirichlet_bc(V, facet_tags, 12, 0.0)
        if bc_sides is not None:
            bcs.append(bc_sides)

    if len(bcs) == 0:
        raise RuntimeError("No Dirichlet BCs were applied. The problem is not well-posed.")

    # ---- Fields ----
    eps_r = build_eps_r_dg0(domain, eps1r, eps2r, z_interface)
    rho = build_rho_dg0(domain, rho0) if rho0 != 0.0 else None

    # Convert eps_r (DG0) into a UFL coefficient
    # Use eps = eps0 * eps_r, but eps0 is constant scale; omit eps0 unless you need SI units in phi.
    eps = eps_r

    # ---- Variational form ----
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(eps * grad(u), grad(v)) * dx
    if rho is None:
        L = PETSc.ScalarType(0.0) * v * dx
    else:
        # sign convention: -div(eps grad u) = rho  ->  ∫ eps ∇u·∇v dx = ∫ rho v dx
        L = rho * v * dx

    problem = fem.petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14
        }
    )
    phi = problem.solve()
    phi.name = "phi"

    # ---- Debug: write facet tags ----
    with io.XDMFFile(domain.comm, f"{outprefix}_tags.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_meshtags(facet_tags, domain.geometry)

    # ---- Write solution + eps ----
    with io.XDMFFile(domain.comm, f"{outprefix}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(phi)
        xf.write_function(eps_r)

    return phi, eps_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--msh", required=True, help="Path to gmsh .msh with Physical Surface tags")
    p.add_argument("--deg", type=int, default=1, help="CG degree for phi")
    p.add_argument("--gate_voltages", type=str, required=True,
                   help='JSON dict mapping gate tags to voltages, e.g. \'{"101":0.2,"102":-0.1}\'')
    p.add_argument("--eps1r", type=float, default=11.7)
    p.add_argument("--eps2r", type=float, default=11.7)
    p.add_argument("--z_interface", type=float, default=0.0,
                   help="Layer interface at this z (in meters). Use 0 to make eps constant if eps1r=eps2r.")
    p.add_argument("--rho0", type=float, default=0.0, help="Uniform rho (DG0) for Poisson; 0 => Laplace")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--basename", type=str, default="embedded_gates")
    p.add_argument("--ground_bottom_tag", type=int, default=10, help="Physical Surface ID for bottom outer face")
    p.add_argument("--ground_sides", type=int, default=0, help="If 1, also ground Physical Surface 12")
    args = p.parse_args()

    gate_tag_to_V = json.loads(args.gate_voltages)

    domain, cell_tags, facet_tags = load_msh(args.msh)

    outprefix = f"{args.outdir}/{args.basename}"
    if domain.comm.rank == 0:
        import os
        os.makedirs(args.outdir, exist_ok=True)

    solve_poisson(
        domain=domain,
        facet_tags=facet_tags,
        deg=args.deg,
        gate_tag_to_V=gate_tag_to_V,
        eps1r=args.eps1r,
        eps2r=args.eps2r,
        z_interface=args.z_interface,
        rho0=args.rho0,
        outprefix=outprefix,
        ground_bottom_tag=args.ground_bottom_tag,
        also_ground_sides=bool(args.ground_sides)
    )

    if domain.comm.rank == 0:
        print(f"[OK] Wrote: {outprefix}.xdmf and {outprefix}_tags.xdmf")


if __name__ == "__main__":
    main()

