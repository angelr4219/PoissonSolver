#!/usr/bin/env python3
import argparse
import math
import os
import json

import numpy as np
from mpi4py import MPI

import gmsh
from dolfinx.io import gmsh as gmshio
from dolfinx import fem, io
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc


def regular_polygon_vertices(n, cx, cy, r, z, angle0=0.0):
    return [(cx + r * math.cos(angle0 + 2.0 * math.pi * k / n),
             cy + r * math.sin(angle0 + 2.0 * math.pi * k / n),
             z) for k in range(n)]


def point_in_convex_polygon_2d(pt, poly_xy):
    x, y = pt
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        px, py = x - x1, y - y1
        cross = ex * py - ey * px
        if cross < -1e-12:
            return False
    return True


def regular_polygon_area(n, r):
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n)


def _petsc_vec_from_function(u: fem.Function):
    if hasattr(u, "vector"):
        return u.vector
    return u.x.petsc_vec


def _local_array_from_function(u: fem.Function):
    if hasattr(u, "x") and hasattr(u.x, "array"):
        return u.x.array
    with u.vector.localForm() as loc:
        return loc.array.copy()


def build_gmsh_box_with_polygon_gate(args, comm: MPI.Comm):
    """
    Returns:
      msh, ct, ft, tags, geom_info
    where geom_info contains polygon vertices, bbox, chosen gate surface COM and area.
    """
    geom_info = {}
    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("polygon_pad_gate")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.lc)

        Lx, Ly, H = args.Lx, args.Ly, args.H
        vol = gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, H)

        if args.shape == "triangle":
            n = 3
            angle0 = math.pi / 2.0
        elif args.shape == "hexagon":
            n = 6
            angle0 = 0.0
        else:
            raise ValueError(f"Unknown shape: {args.shape}")

        cx, cy, r = args.cx, args.cy, args.r
        verts = regular_polygon_vertices(n, cx, cy, r, z=H, angle0=angle0)
        poly_xy = [(x, y) for (x, y, _) in verts]

        xs = [x for (x, _, _) in verts]
        ys = [y for (_, y, _) in verts]
        bbox = {"xmin": float(min(xs)), "xmax": float(max(xs)),
                "ymin": float(min(ys)), "ymax": float(max(ys)),
                "z": float(H)}

        # Store basic polygon definition
        geom_info.update({
            "shape": args.shape,
            "n_vertices": n,
            "center": [float(cx), float(cy), float(H)],
            "r": float(r),
            "vertices": [[float(x), float(y), float(z)] for (x, y, z) in verts],
            "bbox": bbox,
            "target_area": float(regular_polygon_area(n, r)),
        })

        # Build polygon surface
        p_tags = [gmsh.model.occ.addPoint(x, y, H, args.lc) for (x, y, _) in verts]
        l_tags = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i + 1) % n]) for i in range(n)]
        cloop = gmsh.model.occ.addCurveLoop(l_tags)
        gate_surf = gmsh.model.occ.addPlaneSurface([cloop])

        gmsh.model.occ.fragment([(3, vol)], [(2, gate_surf)])
        gmsh.model.occ.synchronize()

        surfs = gmsh.model.occ.getEntities(dim=2)

        def com(dimtag):
            return gmsh.model.occ.getCenterOfMass(dimtag[0], dimtag[1])

        def area(dimtag):
            return gmsh.model.occ.getMass(dimtag[0], dimtag[1])  # area for dim=2

        top_surfs, bot_surfs, side_surfs = [], [], []
        for s in surfs:
            xcm, ycm, zcm = com(s)
            if abs(zcm - H) < 1e-9:
                top_surfs.append(s)
            elif abs(zcm - 0.0) < 1e-9:
                bot_surfs.append(s)
            else:
                side_surfs.append(s)

        gate_candidates = []
        for s in top_surfs:
            xcm, ycm, _ = com(s)
            if point_in_convex_polygon_2d((xcm, ycm), poly_xy):
                gate_candidates.append(s)

        if len(gate_candidates) == 0:
            raise RuntimeError("No gate surface candidate found. Try reducing r or moving (cx,cy).")

        A_target = geom_info["target_area"]
        cand_areas = [(s, area(s)) for s in gate_candidates]
        gate = min(cand_areas, key=lambda t: abs(t[1] - A_target))[0]
        top_rest = [s for s in top_surfs if s != gate]

        # Record chosen gate surface info
        gx, gy, gz = com(gate)
        gA = area(gate)
        geom_info["chosen_gate_surface"] = {
            "gmsh_dimtag": [int(gate[0]), int(gate[1])],
            "com": [float(gx), float(gy), float(gz)],
            "area": float(gA),
            "area_error": float(abs(gA - A_target)),
        }

        if args.debug_geom:
            print("Top surfaces:", len(top_surfs))
            print("Gate candidates (COM-in-poly):", len(gate_candidates))
            for s, A in cand_areas:
                xcm, ycm, _ = com(s)
                print(f"  cand tag={s[1]}  area={A:.6e}  |A-A_target|={abs(A-A_target):.6e}  com=({xcm:.3f},{ycm:.3f})")
            print(f"Chosen gate tag={gate[1]}  target area={A_target:.6e}")

        TAG_VOL = 11
        TAG_GATE = 101
        TAG_TOP_REST = 102
        TAG_BOTTOM = 103
        TAG_SIDES = 104

        gmsh.model.addPhysicalGroup(3, [vol], TAG_VOL)
        gmsh.model.setPhysicalName(3, TAG_VOL, "VOLUME")

        gmsh.model.addPhysicalGroup(2, [gate[1]], TAG_GATE)
        gmsh.model.setPhysicalName(2, TAG_GATE, "GATE")

        gmsh.model.addPhysicalGroup(2, [s[1] for s in top_rest], TAG_TOP_REST)
        gmsh.model.setPhysicalName(2, TAG_TOP_REST, "TOP_REST")

        gmsh.model.addPhysicalGroup(2, [s[1] for s in bot_surfs], TAG_BOTTOM)
        gmsh.model.setPhysicalName(2, TAG_BOTTOM, "BOTTOM")

        gmsh.model.addPhysicalGroup(2, [s[1] for s in side_surfs], TAG_SIDES)
        gmsh.model.setPhysicalName(2, TAG_SIDES, "SIDES")

        gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]

    if comm.rank == 0:
        gmsh.finalize()

    tags = {"VOL": 11, "GATE": 101, "TOP_REST": 102, "BOTTOM": 103, "SIDES": 104}
    return msh, ct, ft, tags, geom_info


def solve_laplace(args, msh, ft, tags):
    V = fem.functionspace(msh, ("CG", args.p))

    u_gate = fem.Constant(msh, PETSc.ScalarType(args.Vg))
    u_gnd = fem.Constant(msh, PETSc.ScalarType(0.0))

    fdim = msh.topology.dim - 1
    gate_facets = np.where(ft.values == tags["GATE"])[0]
    bottom_facets = np.where(ft.values == tags["BOTTOM"])[0]

    gate_dofs = fem.locate_dofs_topological(V, fdim, ft.indices[gate_facets])
    bottom_dofs = fem.locate_dofs_topological(V, fdim, ft.indices[bottom_facets])

    bcs = [
        fem.dirichletbc(u_gate, gate_dofs, V),
        fem.dirichletbc(u_gnd, bottom_dofs, V),
    ]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    zero = fem.Constant(msh, PETSc.ScalarType(0.0))
    L = zero * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fem_petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = fem_petsc.assemble_vector(L_form)

    fem_petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    uh = fem.Function(V)
    x = _petsc_vec_from_function(uh)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=args.rtol, max_it=args.max_it)
    ksp.setFromOptions()
    ksp.solve(b, x)

    u_local = _local_array_from_function(uh)
    umin_local = np.min(u_local) if u_local.size else np.inf
    umax_local = np.max(u_local) if u_local.size else -np.inf
    umin = msh.comm.allreduce(umin_local, op=MPI.MIN)
    umax = msh.comm.allreduce(umax_local, op=MPI.MAX)

    ngate_local = int(np.sum(ft.values == tags["GATE"]))
    ngate = msh.comm.allreduce(ngate_local, op=MPI.SUM)

    its = ksp.getIterationNumber()
    rnorm = ksp.getResidualNorm()

    if msh.comm.rank == 0:
        print("\n--- Polygon Pad Gate Laplace Test ---")
        print(f"Box: Lx={args.Lx}, Ly={args.Ly}, H={args.H}")
        print(f"Shape: {args.shape}, center=({args.cx},{args.cy}), r={args.r}")
        print(f"p={args.p}, lc={args.lc}, Vg={args.Vg}")
        print(f"Gate facets (count): {ngate}")
        print(f"KSP: its={its}, resid_norm={rnorm:.3e}")
        print(f"Potential min/max: {umin:.6e} / {umax:.6e}")
        print("------------------------------------\n")

    return uh


def write_outputs(args, msh, ct, ft, uh, geom_info):
    if not args.write_xdmf:
        return
    os.makedirs(args.outdir, exist_ok=True)

    mesh_path = os.path.join(args.outdir, "mesh_tags.xdmf")
    sol_path = os.path.join(args.outdir, "solution.xdmf")
    info_path = os.path.join(args.outdir, "geometry_info.json")

    with io.XDMFFile(msh.comm, mesh_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(ct, msh.geometry)
        xdmf.write_meshtags(ft, msh.geometry)

    with io.XDMFFile(msh.comm, sol_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(uh)

    if msh.comm.rank == 0:
        with open(info_path, "w") as f:
            json.dump(geom_info, f, indent=2)

        # Help the human: print exact locations and what to open
        host_hint = os.path.join(os.getcwd(), args.outdir)
        print("Wrote outputs:")
        print(f"  Container path: /app/{args.outdir}")
        print(f"  Host path:      {host_hint}")
        print("Files:")
        print(f"  {mesh_path}")
        print(f"  {sol_path}")
        print(f"  {info_path}")
        print("ParaView: open mesh_tags.xdmf, then color by facet tags (ft).")


def main():
    parser = argparse.ArgumentParser(description="Single polygon pad gate on top face: Laplace + facet tag verification.")
    parser.add_argument("--shape", choices=["triangle", "hexagon"], default="hexagon")
    parser.add_argument("--Lx", type=float, default=2.0)
    parser.add_argument("--Ly", type=float, default=2.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--cx", type=float, default=1.0)
    parser.add_argument("--cy", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.4)
    parser.add_argument("--lc", type=float, default=0.15)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--Vg", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--max-it", type=int, default=500)
    parser.add_argument("--write-xdmf", action="store_true")
    parser.add_argument("--outdir", type=str, default="Results/polygon_pad_gate")
    parser.add_argument("--debug-geom", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    msh, ct, ft, tags, geom_info = build_gmsh_box_with_polygon_gate(args, comm)
    uh = solve_laplace(args, msh, ft, tags)
    write_outputs(args, msh, ct, ft, uh, geom_info)

    if comm.rank == 0:
        # Also print the polygon vertices plainly
        print("\nGate polygon vertices (x,y,z):")
        for v in geom_info.get("vertices", []):
            print(f"  ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")
        if "chosen_gate_surface" in geom_info:
            com = geom_info["chosen_gate_surface"]["com"]
            area = geom_info["chosen_gate_surface"]["area"]
            print(f"\nChosen gate surface COM: ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})")
            print(f"Chosen gate surface area: {area:.6e}")


if __name__ == "__main__":
    main()
