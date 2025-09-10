# Extended-PS.py
from mpi4py import MPI
import argparse
import numpy as np
import ufl, gmsh
from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem.petsc import LinearProblem


def solve_poisson_gates_var_eps(
    Lx=1.0, Ly=1.0,
    oxide_ymin=0.80,
    gateA=(0.15, 0.75, 0.20, 0.20),   # (x0, y0, w, h)
    gateB=(0.55, 0.75, 0.30, 0.20),
    eps_r_semic=11.7, eps_r_oxide=3.9,
    VgateA=0.20, VgateB=0.00,
    h=0.03,
    outfile="phi_with_gates_var_eps.xdmf"
):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    gmsh.initialize()
    gmsh.model.add("rect-with-holes-materials")

    # --- Outer domain
    outer = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)

    # --- Rectangular holes (gates)
    xA, yA, wA, hA = gateA
    xB, yB, wB, hB = gateB
    holeA = gmsh.model.occ.addRectangle(xA, yA, 0.0, wA, hA)
    holeB = gmsh.model.occ.addRectangle(xB, yB, 0.0, wB, hB)
    gmsh.model.occ.cut([(2, outer)], [(2, holeA), (2, holeB)],
                       removeObject=True, removeTool=True)

    # --- Top oxide strip (NOT a hole); fragment to split materials
    oxide_strip = gmsh.model.occ.addRectangle(0.0, oxide_ymin, 0.0, Lx, Ly - oxide_ymin)
    gmsh.model.occ.fragment(gmsh.model.occ.getEntities(2), [(2, oxide_strip)])
    gmsh.model.occ.synchronize()

    # --- Classify surfaces by centroid y (top = oxide, bottom = semiconductor)
    def cy(tag):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, tag)
        return 0.5 * (ymin + ymax)

    surfs = [s for (dim, s) in gmsh.model.getEntities(2)]
    top_surf_tags = [s for s in surfs if cy(s) > (oxide_ymin - 0.05)]
    bot_surf_tags = [s for s in surfs if s not in top_surf_tags]

    # --- All boundary curves of the material surfaces
    all_curves = gmsh.model.getBoundary([(2, t) for t in top_surf_tags + bot_surf_tags],
                                        oriented=False, recursive=True)
    all_curves = [c[1] for c in all_curves if c[0] == 1]  # keep only dim=1 tags

    # --- Robustly pick gate perimeters via bounding boxes around gate rectangles
    pad = 1e-8
    gateA_bbox = (xA - pad, yA - pad, -pad, xA + wA + pad, yA + hA + pad, +pad)
    gateB_bbox = (xB - pad, yB - pad, -pad, xB + wB + pad, yB + hB + pad, +pad)

    gateA_curves_raw = [tag for (_, tag) in gmsh.model.getEntitiesInBoundingBox(*gateA_bbox, 1)]
    gateB_curves_raw = [tag for (_, tag) in gmsh.model.getEntitiesInBoundingBox(*gateB_bbox, 1)]

    # Intersect with actual boundary curves to avoid picking unrelated lines
    all_set = set(all_curves)
    gateA_curves = sorted(all_set.intersection(gateA_curves_raw))
    gateB_curves = sorted(all_set.intersection(gateB_curves_raw))
    outer_curves = sorted(all_set.difference(gateA_curves).difference(gateB_curves))

    # --- Physical groups (boundaries)
    pg_outer = gmsh.model.addPhysicalGroup(1, outer_curves); gmsh.model.setPhysicalName(1, pg_outer, "outer")
    pg_gateA = gmsh.model.addPhysicalGroup(1, gateA_curves); gmsh.model.setPhysicalName(1, pg_gateA, "gateA")
    pg_gateB = gmsh.model.addPhysicalGroup(1, gateB_curves); gmsh.model.setPhysicalName(1, pg_gateB, "gateB")

    # --- Physical groups (materials)
    pg_semic = gmsh.model.addPhysicalGroup(2, bot_surf_tags); gmsh.model.setPhysicalName(2, pg_semic, "semiconductor")
    pg_oxide = gmsh.model.addPhysicalGroup(2, top_surf_tags); gmsh.model.setPhysicalName(2, pg_oxide, "oxide")

    # --- Mesh controls
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)

    # --- To DOLFINx
    domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
    gmsh.finalize()

    # >>> Ensure facet<->cell connectivity for BC lookup <<<
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)

    # --- FE spaces
    V = fem.functionspace(domain, ("Lagrange", 1))
    W = fem.functionspace(domain, ("DG", 0))  # ε(x) cell-wise constants

    # --- ε(x) on DG0 (default: semiconductor; overwrite oxide cells)
    eps0 = 8.8541878128e-12
    eps = fem.Function(W)
    eps.x.array[:] = eps0 * eps_r_semic
    oxide_cells = cell_tags.find(pg_oxide)            # local cell indices tagged as oxide
    eps.x.array[oxide_cells] = eps0 * eps_r_oxide
    eps.name = "epsilon"

    # --- Source (Laplace)
    rho = fem.Constant(domain, 0.0)

    # --- Dirichlet BCs on gate perimeters
    def dirichlet_on(tag, value):
        facets = facet_tags.find(tag)
        if rank == 0:
            print(f"[debug] tag {tag}: {len(facets)} facets")
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        return fem.dirichletbc(fem.Constant(domain, value), dofs, V)

    bcs = [dirichlet_on(pg_gateA, VgateA),
           dirichlet_on(pg_gateB, VgateB)]

    # --- Weak form: -∇·(ε ∇φ) = ρ
    phi = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx

    uh = LinearProblem(a, L, bcs=bcs).solve()
    uh.name = "phi"

    if rank == 0:
        print(f"[gates] phi: min={uh.x.array.min():.4g} V, max={uh.x.array.max():.4g} V")

    # --- Write results
    with XDMFFile(comm, outfile, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
        xdmf.write_function(eps)

    return uh


def _cli():
    p = argparse.ArgumentParser(description="Poisson with rectangular gates + variable permittivity")
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--oxide_ymin", type=float, default=0.80)
    p.add_argument("--gateA", type=float, nargs=4, default=[0.15, 0.75, 0.20, 0.20], metavar=("x0","y0","w","h"))
    p.add_argument("--gateB", type=float, nargs=4, default=[0.55, 0.75, 0.30, 0.20], metavar=("x0","y0","w","h"))
    p.add_argument("--eps_r_semic", type=float, default=11.7)
    p.add_argument("--eps_r_oxide", type=float, default=3.9)
    p.add_argument("--VgateA", type=float, default=0.20)
    p.add_argument("--VgateB", type=float, default=0.00)
    p.add_argument("--h", type=float, default=0.03)
    p.add_argument("--outfile", type=str, default="phi_with_gates_var_eps.xdmf")
    args = p.parse_args()

    solve_poisson_gates_var_eps(
        Lx=args.Lx, Ly=args.Ly, oxide_ymin=args.oxide_ymin,
        gateA=tuple(args.gateA), gateB=tuple(args.gateB),
        eps_r_semic=args.eps_r_semic, eps_r_oxide=args.eps_r_oxide,
        VgateA=args.VgateA, VgateB=args.VgateB,
        h=args.h, outfile=args.outfile
    )


if __name__ == "__main__":
    _cli()
