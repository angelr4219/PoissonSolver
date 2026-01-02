#!/usr/bin/env python3
"""
3D rectangular top-gates + 3D box-charge regions in a rectangular domain.

Domain:
  x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2], z in [-Hz/2, Hz/2]

BCs (Dirichlet):
  - bottom (z = -Hz/2): 0 V
  - top (z = +Hz/2): 0 V EXCEPT rectangular gate patches with specified voltages

PDE:
  -Δ phi = rho   (units arbitrary; intended as a geometric/BC benchmark)

Outputs:
  results/.../<basename>.xdmf         : phi (CG) and rho (DG0)
  results/.../<basename>_facets.xdmf  : facet tags (10 bottom, 11 top-rest, 101+ gates)
"""

import os, json, argparse
import numpy as np
from mpi4py import MPI

from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem


def _mk_facet_tags(domain, Lx, Ly, Hz, gates):
    """
    Create facet tags:
      10 = bottom (z=-Hz/2)
      11 = top (z=+Hz/2) excluding gate patches
      101,102,... = gate patches on the top surface (in order given)
    """
    fdim = domain.topology.dim - 1
    z_top = +Hz / 2.0
    z_bot = -Hz / 2.0
    eps = 1e-14 * max(Lx, Ly, Hz)

    # Locate top/bottom boundary facets
    top_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_top, atol=eps)
    )
    bot_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_bot, atol=eps)
    )

    # Midpoints of top facets (for robust rectangle classification)
    top_mid = mesh.compute_midpoints(domain, fdim, top_facets)

    # Build a mapping facet -> tag (with gate tags overriding top-rest)
    facet_to_tag = {}

    # Bottom = 10
    for f in bot_facets:
        facet_to_tag[int(f)] = 10

    # Gate patches (tags 101+)
    gate_facets_union = np.array([], dtype=np.int32)
    for i, g in enumerate(gates):
        tag = 101 + i
        x0 = float(g["x0"]); y0 = float(g["y0"])
        x1 = x0 + float(g["dx"]); y1 = y0 + float(g["dy"])

        in_gate = (
            (top_mid[:, 0] >= x0 - eps) & (top_mid[:, 0] <= x1 + eps) &
            (top_mid[:, 1] >= y0 - eps) & (top_mid[:, 1] <= y1 + eps)
        )
        gf = top_facets[in_gate]
        if gf.size == 0 and domain.comm.rank == 0:
            print(f"[WARN] Gate {i} rectangle hit 0 top facets (check nx/ny or gate coords).")
        gate_facets_union = np.unique(np.concatenate([gate_facets_union, gf.astype(np.int32)]))
        for f in gf:
            facet_to_tag[int(f)] = tag

    # Top-rest = 11 (only top facets not in any gate)
    gate_set = set(int(f) for f in gate_facets_union)
    for f in top_facets:
        if int(f) not in gate_set:
            facet_to_tag[int(f)] = 11

    # Convert mapping to meshtags arrays
    facets = np.array(sorted(facet_to_tag.keys()), dtype=np.int32)
    values = np.array([facet_to_tag[int(f)] for f in facets], dtype=np.int32)
    facet_tags = mesh.meshtags(domain, fdim, facets, values)

    return facet_tags


def _dirichlet_on_tag(V, facet_tags, tag, value):
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(int(tag))
    if facets.size == 0:
        if V.mesh.comm.rank == 0:
            print(f"[WARN] No facets for tag={tag} (BC {value} not applied).")
        return None
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(PETSc.ScalarType(value), dofs, V)


def _build_rho_dg0(domain, charges):
    """
    DG0 rho filled by cell-midpoints. Charges are axis-aligned boxes with constant rho.
    If boxes overlap, rho adds.
    """
    Q = fem.functionspace(domain, ("DG", 0))
    rho = fem.Function(Q, name="rho")
    rho.x.array[:] = PETSc.ScalarType(0.0)

    if not charges:
        return rho

    X = Q.tabulate_dof_coordinates().reshape((-1, domain.geometry.dim))

    for c in charges:
        x0 = float(c["x0"]); y0 = float(c["y0"]); z0 = float(c["z0"])
        x1 = x0 + float(c["dx"]); y1 = y0 + float(c["dy"]); z1 = z0 + float(c["dz"])
        val = PETSc.ScalarType(float(c["rho"]))

        mask = (
            (X[:, 0] >= x0) & (X[:, 0] <= x1) &
            (X[:, 1] >= y0) & (X[:, 1] <= y1) &
            (X[:, 2] >= z0) & (X[:, 2] <= z1)
        )
        rho.x.array[mask] += val

    return rho


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=float, required=True)
    p.add_argument("--Ly", type=float, required=True)
    p.add_argument("--Hz", type=float, required=True)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--nz", type=int, default=20)
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--gates", type=str, required=True, help="JSON list of top-gate rectangles with V")
    p.add_argument("--charges", type=str, default="[]", help="JSON list of box charges with rho")
    p.add_argument("--outdir", type=str, default="results/rect_gate_3d")
    p.add_argument("--basename", type=str, default="device3d_boxrho")
    args = p.parse_args()

    gates = json.loads(args.gates)
    charges = json.loads(args.charges)

    comm = MPI.COMM_WORLD

    # Structured hexahedral box mesh
    p0 = np.array([-args.Lx/2, -args.Ly/2, -args.Hz/2], dtype=np.float64)
    p1 = np.array([+args.Lx/2, +args.Ly/2, +args.Hz/2], dtype=np.float64)
    domain = mesh.create_box(
        comm, [p0, p1], [args.nx, args.ny, args.nz],
        cell_type=mesh.CellType.hexahedron
    )

    facet_tags = _mk_facet_tags(domain, args.Lx, args.Ly, args.Hz, gates)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", args.deg))

    # BCs: bottom=0, top-rest=0, gates=V_i
    bcs = []
    bc_bot = _dirichlet_on_tag(V, facet_tags, 10, 0.0)
    if bc_bot is not None:
        bcs.append(bc_bot)

    bc_top_rest = _dirichlet_on_tag(V, facet_tags, 11, 0.0)
    if bc_top_rest is not None:
        bcs.append(bc_top_rest)

    for i, g in enumerate(gates):
        tag = 101 + i
        Vg = float(g["V"])
        bc = _dirichlet_on_tag(V, facet_tags, tag, Vg)
        if bc is not None:
            bcs.append(bc)

    if len(bcs) == 0:
        raise RuntimeError("No Dirichlet BCs were applied; problem is not well-posed.")

    # rho (DG0)
    rho = _build_rho_dg0(domain, charges)

    # Variational problem: -Δphi = rho
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    if charges:
        L = rho * v * ufl.dx
    else:
        L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    problem = LinearProblem(
        a, L,
        bcs=bcs,
        petsc_options_prefix="rg3d_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 2000
        }
    )

    phi = problem.solve()
    phi.name = "phi"

    # Output
    outprefix = os.path.join(args.outdir, args.basename)
    if comm.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)

    with io.XDMFFile(comm, f"{outprefix}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(phi)
        xf.write_function(rho)

    with io.XDMFFile(comm, f"{outprefix}_facets.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_meshtags(facet_tags, domain.geometry)

    if comm.rank == 0:
        print(f"[OK] Wrote {outprefix}.xdmf and {outprefix}_facets.xdmf")


if __name__ == "__main__":
    main()
