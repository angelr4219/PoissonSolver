#!/usr/bin/env python3
"""
3D Rectangular-gate Laplace benchmark (pure DOLFINx; no gmsh)

Domain:
  x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2], z in [-Hz, +Hz]
  (so total height is 2*Hz, with an interface plane at z=0)

PDE:
  ∇ · ( ε(x) ∇φ ) = 0
  ε(x) = ε0 * εr(z), two-layer:
     z >= 0 -> eps_up
     z <  0 -> eps_dn

BCs:
  - Bottom (z=-Hz): φ = 0
  - Top (z=+Hz): φ = 0 except on gate rectangles (Dirichlet patches) with specified voltages
  - Sides: natural (Neumann 0 flux), unless you add Dirichlet yourself

Facet tags (Physical-style):
  10 = bottom (z=-Hz)
  11 = top_ground (z=+Hz excluding gates)
  12 = sides (x=±Lx/2, y=±Ly/2)
  101,102,... = gate patches on top surface

Outputs:
  <basename>.xdmf         mesh + phi
  <basename>_facets.xdmf  mesh + facet tags
  <basename>_cells.xdmf   mesh + cell tags (layer ids: 1=lower z<0, 2=upper z>=0)
"""

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem


def near(a, b, eps=1e-12):
    return np.isclose(a, b, atol=eps, rtol=0.0)


def build_mesh(Lx, Ly, Hz, nx, ny, nz, hexa=True):
    p0 = np.array([-Lx / 2, -Ly / 2, -Hz], dtype=np.float64)
    p1 = np.array([+Lx / 2, +Ly / 2, +Hz], dtype=np.float64)
    cell_type = mesh.CellType.hexahedron if hexa else mesh.CellType.tetrahedron
    return mesh.create_box(MPI.COMM_WORLD, [p0, p1], [nx, ny, nz], cell_type=cell_type)


def make_facet_tags(domain, Lx, Ly, Hz, gates, eps=1e-12):
    tdim = domain.topology.dim
    fdim = tdim - 1

    # --- base facets
    facets_top_all = mesh.locate_entities_boundary(domain, fdim, lambda x: near(x[2], +Hz, eps))
    facets_bot = mesh.locate_entities_boundary(domain, fdim, lambda x: near(x[2], -Hz, eps))
    facets_side = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: (near(x[0], -Lx/2, eps) | near(x[0], +Lx/2, eps) |
                   near(x[1], -Ly/2, eps) | near(x[1], +Ly/2, eps))
    )

    # --- gate facets (subset of top)
    gate_facet_union = np.array([], dtype=np.int32)
    gate_facets_by_tag = {}

    for g in gates:
        tag = int(g["tag"])
        x0, y0 = float(g["x0"]), float(g["y0"])
        dx, dy = float(g["dx"]), float(g["dy"])

        fg = mesh.locate_entities_boundary(
            domain, fdim,
            lambda x, x0=x0, y0=y0, dx=dx, dy=dy: (
                near(x[2], +Hz, eps) &
                (x0 <= x[0]) & (x[0] <= x0 + dx) &
                (y0 <= x[1]) & (x[1] <= y0 + dy)
            )
        )
        gate_facets_by_tag[tag] = fg
        gate_facet_union = np.union1d(gate_facet_union, fg).astype(np.int32)

    facets_top_ground = np.setdiff1d(facets_top_all, gate_facet_union).astype(np.int32)

    # Build meshtags arrays; if duplicates exist, later entries should “win”.
    # We ensure gates get written after top_ground, so their tag dominates.
    facet_indices = []
    facet_values = []

    def add_block(facets, value):
        if facets is None or len(facets) == 0:
            return
        facet_indices.append(np.array(facets, dtype=np.int32))
        facet_values.append(np.full(len(facets), int(value), dtype=np.int32))

    add_block(facets_bot, 10)
    add_block(facets_top_ground, 11)
    add_block(facets_side, 12)

    for tag, fg in gate_facets_by_tag.items():
        add_block(fg, tag)

    if len(facet_indices) == 0:
        raise RuntimeError("No boundary facets were tagged (unexpected).")

    idx = np.concatenate(facet_indices)
    val = np.concatenate(facet_values)

    # Resolve duplicates by keeping the last assignment:
    # sort by index, then stable-keep last by scanning from end.
    order = np.argsort(idx, kind="mergesort")
    idx_s = idx[order]
    val_s = val[order]
    # keep last occurrence
    keep = np.ones_like(idx_s, dtype=bool)
    keep[:-1] = idx_s[:-1] != idx_s[1:]
    facet_tags = mesh.meshtags(domain, fdim, idx_s[keep], val_s[keep])
    return facet_tags


def make_cell_tags_layers(domain):
    """
    Cell tags: 1 for lower layer (z<0), 2 for upper layer (z>=0).
    """
    tdim = domain.topology.dim
    cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    mids = mesh.compute_midpoints(domain, tdim, cells)
    lower = cells[mids[:, 2] < 0.0]
    upper = np.setdiff1d(cells, lower)

    idx = np.concatenate([lower, upper]).astype(np.int32)
    val = np.concatenate([
        np.full(len(lower), 1, dtype=np.int32),
        np.full(len(upper), 2, dtype=np.int32),
    ])

    order = np.argsort(idx)
    return mesh.meshtags(domain, tdim, idx[order], val[order])


def solve(domain, facet_tags, gates, eps_up, eps_dn, deg):
    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("CG", deg))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    eps_r = ufl.conditional(ufl.ge(x[2], 0.0), float(eps_up), float(eps_dn))
    a = ufl.inner(eps_r * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.Constant(domain, 0.0) * v * ufl.dx

    bcs = []

    # bottom z=-Hz => tag 10 => 0V
    dofs_bot = fem.locate_dofs_topological(V, fdim, facet_tags.find(10))
    bcs.append(fem.dirichletbc(fem.Constant(domain, 0.0), dofs_bot, V))

    # top ground => tag 11 => 0V
    dofs_topg = fem.locate_dofs_topological(V, fdim, facet_tags.find(11))
    bcs.append(fem.dirichletbc(fem.Constant(domain, 0.0), dofs_topg, V))

    # gates => tags 101.. with specified voltages
    for g in gates:
        tag = int(g["tag"])
        Vg = float(g.get("V", 0.0))
        facets = facet_tags.find(tag)
        if facets is None or len(facets) == 0:
            continue
        dofs_g = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(fem.Constant(domain, Vg), dofs_g, V))

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10
    })
    phi = problem.solve()
    phi.name = "phi"
    return phi


def main():
    import argparse, json, os
    p = argparse.ArgumentParser()

    p.add_argument("--Lx", type=float, required=True)
    p.add_argument("--Ly", type=float, required=True)
    p.add_argument("--Hz", type=float, required=True, help="Half-height: domain is z in [-Hz, +Hz]")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--nz", type=int, default=20)
    p.add_argument("--hexa", type=int, default=1, help="1=hex mesh, 0=tet mesh")

    p.add_argument("--deg", type=int, default=1)

    p.add_argument("--eps_up", type=float, default=11.7, help="epsilon_r for z>=0")
    p.add_argument("--eps_dn", type=float, default=3.9,  help="epsilon_r for z<0")

    p.add_argument("--gates", type=str, required=True,
                   help='JSON list: [{"x0":..,"y0":..,"dx":..,"dy":..,"tag":101,"V":0.2}, ...]')

    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--basename", type=str, default="rect_gate_3d")

    args = p.parse_args()
    gates = json.loads(args.gates)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, args.basename)

    domain = build_mesh(args.Lx, args.Ly, args.Hz, args.nx, args.ny, args.nz, hexa=bool(args.hexa))
    facet_tags = make_facet_tags(domain, args.Lx, args.Ly, args.Hz, gates)
    cell_tags = make_cell_tags_layers(domain)

    phi = solve(domain, facet_tags, gates, args.eps_up, args.eps_dn, args.deg)

    # Write mesh + solution
    with io.XDMFFile(domain.comm, f"{base}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(phi)

    # Write facet tags
    with io.XDMFFile(domain.comm, f"{base}_facets.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_meshtags(facet_tags, domain.geometry)

    # Write cell tags (layers)
    with io.XDMFFile(domain.comm, f"{base}_cells.xdmf", "w") as xf:
        xf.write_mesh(domain)
        xf.write_meshtags(cell_tags, domain.geometry)

    if domain.comm.rank == 0:
        arr = phi.x.array
        print(f"RESULT phi_min={arr.min():.6e} phi_max={arr.max():.6e} wrote={base}*.xdmf")


if __name__ == "__main__":
    main()
