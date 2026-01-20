#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from dolfinx import fem, io
from dolfinx.io import gmshio
import ufl

from petsc4py import PETSc

# ----------------------------
# Gmsh geometry + tagging
# ----------------------------

@dataclass
class Tags:
    # Volumes
    OX: int = 11
    SI: int = 12
    # Facets (boundaries)
    TOP_GROUND: int = 21
    BOTTOM: int = 22
    SIDES: int = 23
    # Gates: gate i uses tag GATE0 + i
    GATE0: int = 100


def build_gmsh_model(args, tags: Tags, comm: MPI.Comm):
    """
    Build a stacked box:
      z in [0, tox]      : oxide
      z in [tox, H]      : semiconductor
    Gates are square Dirichlet patches on the top plane z=0.

    We "imprint" the gate rectangles onto the top boundary by boolean fragmenting.
    """
    import gmsh  # imported here so the script still imports without gmsh

    rank = comm.rank
    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("gate_stack")

        Lx, Ly, H, tox = args.Lx, args.Ly, args.H, args.tox
        x0, y0 = -Lx / 2.0, -Ly / 2.0

        occ = gmsh.model.occ

        # Volumes
        if tox > 0:
            v_ox = occ.addBox(x0, y0, 0.0, Lx, Ly, tox)
            v_si = occ.addBox(x0, y0, tox, Lx, Ly, H - tox)
            obj = [(3, v_ox)]
            tool = [(3, v_si)]
            out = occ.fragment(obj, tool)
            occ.synchronize()
        else:
            # Single material: just "si" volume
            v_si = occ.addBox(x0, y0, 0.0, Lx, Ly, H)
            occ.synchronize()

        # Gate rectangles (2D entities) on z=0
        a = args.a
        xs = args.gate_xs
        ys = args.gate_ys
        assert len(xs) == len(ys) == len(args.gate_Vs)

        gate_surfs = []
        for (xc, yc) in zip(xs, ys):
            r = occ.addRectangle(xc - a, yc - a, 0.0, 2.0 * a, 2.0 * a)
            gate_surfs.append((2, r))
        occ.synchronize()

        # Imprint the rectangles onto the volume boundary by fragmenting with tools=gate_surfs
        if tox > 0:
            vols = gmsh.model.getEntities(dim=3)  # oxide + si after fragment
            occ.fragment(vols, gate_surfs)
        else:
            occ.fragment([(3, v_si)], gate_surfs)
        occ.synchronize()

        # Mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.h)

        # Identify volumes by center-of-mass z
        vols = gmsh.model.getEntities(dim=3)
        ox_vols, si_vols = [], []
        for (dim, tag) in vols:
            cx, cy, cz = occ.getCenterOfMass(dim, tag)
            if tox > 0 and cz < tox * 0.5:
                ox_vols.append(tag)
            else:
                si_vols.append(tag)

        if tox > 0 and len(ox_vols) == 0:
            raise RuntimeError("Failed to identify oxide volume(s).")
        if len(si_vols) == 0:
            raise RuntimeError("Failed to identify semiconductor volume(s).")

        # Physical groups for volumes
        if tox > 0:
            gmsh.model.addPhysicalGroup(3, ox_vols, tags.OX)
            gmsh.model.setPhysicalName(3, tags.OX, "oxide")
        gmsh.model.addPhysicalGroup(3, si_vols, tags.SI)
        gmsh.model.setPhysicalName(3, tags.SI, "semiconductor")

        # Classify boundary surfaces
        surfs = gmsh.model.getEntities(dim=2)

        gate_phys = {i: [] for i in range(len(xs))}
        top_ground = []
        bottom = []
        sides = []

        tol = 1e-12

        def in_gate(i, x, y):
            return (abs(x - xs[i]) <= a + 1e-12) and (abs(y - ys[i]) <= a + 1e-12)

        for (dim, s) in surfs:
            cx, cy, cz = occ.getCenterOfMass(dim, s)

            if abs(cz - 0.0) < 1e-10:
                # Top plane: either a gate patch or ground
                assigned = False
                for i in range(len(xs)):
                    if in_gate(i, cx, cy):
                        gate_phys[i].append(s)
                        assigned = True
                        break
                if not assigned:
                    top_ground.append(s)

            elif abs(cz - H) < 1e-10:
                bottom.append(s)
            else:
                # likely a side or internal interface
                # internal interface at z=tox will also appear; we do NOT tag it as a boundary
                if tox > 0 and abs(cz - tox) < 1e-10:
                    continue
                sides.append(s)

        # Physical groups for facets
        if len(top_ground) == 0:
            raise RuntimeError("No top-ground surface found (outside gates).")
        gmsh.model.addPhysicalGroup(2, top_ground, tags.TOP_GROUND)
        gmsh.model.setPhysicalName(2, tags.TOP_GROUND, "top_ground")

        gmsh.model.addPhysicalGroup(2, bottom, tags.BOTTOM)
        gmsh.model.setPhysicalName(2, tags.BOTTOM, "bottom")

        gmsh.model.addPhysicalGroup(2, sides, tags.SIDES)
        gmsh.model.setPhysicalName(2, tags.SIDES, "sides")

        for i, ss in gate_phys.items():
            if len(ss) == 0:
                raise RuntimeError(f"Gate {i} got zero surface entities. Check geometry.")
            phys = tags.GATE0 + i
            gmsh.model.addPhysicalGroup(2, ss, phys)
            gmsh.model.setPhysicalName(2, phys, f"gate_{i}")

        gmsh.model.mesh.generate(3)

        # Optional: write msh
        if args.write_msh:
            gmsh.write(os.path.join(args.outdir, f"{args.basename}.msh"))

    return gmsh


# ----------------------------
# Analytic square-gate potential (half-space) for tox=0 benchmark
# from Anderson et al. Eq. (10)-(11)
# ----------------------------
def g_uvz(u, v, z):
    # g(u,v,z) = (1/(2π)) arctan( u v / ( z sqrt(u^2+v^2+z^2) ) )
    denom = z * np.sqrt(u*u + v*v + z*z)
    return (1.0 / (2.0 * np.pi)) * np.arctan2(u * v, denom)


def phi_square_gates_row(x, y, z, a, xs, Vs):
    # phi0(x,y,z) = - Σ_i V_i [ g(a-x_i + x, a+y, z) + g(a-x_i + x, a-y, z)
    #                          + g(a+x_i - x, a+y, z) + g(a+x_i - x, a-y, z) ]
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    out = np.zeros_like(x, dtype=float)
    for xi, Vi in zip(xs, Vs):
        out -= Vi * (
            g_uvz(a - xi + x, a + y, z) +
            g_uvz(a - xi + x, a - y, z) +
            g_uvz(a + xi - x, a + y, z) +
            g_uvz(a + xi - x, a - y, z)
        )
    return out


# ----------------------------
# Point evaluation helper
# ----------------------------
def eval_function_on_points(u: fem.Function, points: np.ndarray, comm: MPI.Comm):
    """
    Evaluate scalar Function u at points (N,3).
    Returns values on rank 0 (others return None).
    Points outside local mesh get NaN, then we reduce by taking the first finite value.
    """
    from dolfinx import geometry

    mesh = u.function_space.mesh
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    # collisions: candidates per point
    cand = geometry.compute_collisions_points(tree, points)
    cells = geometry.compute_colliding_cells(mesh, cand, points)

    # local values default NaN
    vals_local = np.full((points.shape[0],), np.nan, dtype=np.float64)

    for i in range(points.shape[0]):
        cell_candidates = cells.links(i)
        if len(cell_candidates) > 0:
            # pick the first colliding cell
            cell = cell_candidates[0]
            v = np.zeros((1,), dtype=np.float64)
            u.eval(v, points[i:i+1], np.array([cell], dtype=np.int32))
            vals_local[i] = v[0]

    # Reduce: choose finite value if any rank has it.
    # We'll do a gather and resolve on rank 0.
    all_vals = comm.gather(vals_local, root=0)
    if comm.rank != 0:
        return None

    all_vals = np.stack(all_vals, axis=0)  # (nranks, N)
    out = np.full((points.shape[0],), np.nan, dtype=np.float64)
    for j in range(points.shape[0]):
        col = all_vals[:, j]
        finite = col[np.isfinite(col)]
        out[j] = finite[0] if finite.size > 0 else np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="3D realistic gate stack benchmark (oxide+semiconductor) with tagged gate facets.")
    parser.add_argument("--outdir", type=str, default="results/realistic_gates")
    parser.add_argument("--basename", type=str, default="gate_stack")

    # Geometry
    parser.add_argument("--Lx", type=float, default=300e-9)
    parser.add_argument("--Ly", type=float, default=300e-9)
    parser.add_argument("--H",  type=float, default=200e-9)
    parser.add_argument("--tox", type=float, default=0.0, help="oxide thickness. Use 0 for single-medium analytic check.")
    parser.add_argument("--a", type=float, default=35e-9, help="half-size of square gate. gate size = 2a.")
    parser.add_argument("--gate_xs", type=float, nargs="+", default=[-70e-9, 0.0, 70e-9])
    parser.add_argument("--gate_ys", type=float, nargs="+", default=[0.0, 0.0, 0.0])
    parser.add_argument("--gate_Vs", type=float, nargs="+", default=[0.25, 0.10, 0.25])

    # Mesh/FEM
    parser.add_argument("--h", type=float, default=10e-9)
    parser.add_argument("--deg", type=int, default=1)

    # Materials
    parser.add_argument("--eps_ox", type=float, default=3.9)
    parser.add_argument("--eps_si", type=float, default=11.7)

    # BC switches
    parser.add_argument("--bc_sides", choices=["dirichlet0", "none"], default="dirichlet0")
    parser.add_argument("--bc_bottom", choices=["dirichlet0", "none"], default="dirichlet0")

    # Output
    parser.add_argument("--write_msh", type=int, default=0)
    parser.add_argument("--probe_line", type=int, default=1)
    parser.add_argument("--probe_n", type=int, default=401)
    parser.add_argument("--probe_y", type=float, default=0.0)
    parser.add_argument("--probe_z", type=float, default=35e-9, help="probe depth below top plane (for tox=0 analytic check).")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    os.makedirs(args.outdir, exist_ok=True)

    tags = Tags()
    gmsh = build_gmsh_model(args, tags, comm)

    # Convert gmsh -> dolfinx mesh/tags
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
    if comm.rank == 0:
        gmsh.finalize()

    # Function space
    V = fem.functionspace(mesh, ("CG", args.deg))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Permittivity on cells (DG0)
    eps0 = 8.8541878128e-12  # vacuum permittivity
    Q = fem.functionspace(mesh, ("DG", 0))
    eps = fem.Function(Q)
    eps.x.array[:] = eps0 * args.eps_si  # default

    # Fill oxide cells if present
    if args.tox > 0:
        ox_cells = cell_tags.find(tags.OX)
        if ox_cells.size > 0:
            eps.x.array[ox_cells] = eps0 * args.eps_ox

    # Variational form: ∫ eps ∇u·∇v dx = 0
    a_form = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

    # Dirichlet BCs
    bcs = []
    # Gates
    for i, Vg in enumerate(args.gate_Vs):
        tag = tags.GATE0 + i
        facets = facet_tags.find(tag)
        dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vg), dofs, V))

    # Top ground always 0
    facets = facet_tags.find(tags.TOP_GROUND)
    dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    # Bottom, sides optional
    if args.bc_bottom == "dirichlet0":
        facets = facet_tags.find(tags.BOTTOM)
        dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    if args.bc_sides == "dirichlet0":
        facets = facet_tags.find(tags.SIDES)
        dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))

    # Solve
    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": "1.0e-10",
        "ksp_max_it": "20000",
    }
    problem = fem.petsc.LinearProblem(a_form, L_form, bcs=bcs, petsc_options=petsc_opts)
    uh = problem.solve()
    uh.name = "phi"

    # Print basic stats
    local_min = np.min(uh.x.array) if uh.x.array.size else np.inf
    local_max = np.max(uh.x.array) if uh.x.array.size else -np.inf
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    if comm.rank == 0:
        print("=== realistic_gate_stack_benchmark ===")
        print(f"Lx={args.Lx:.3e} Ly={args.Ly:.3e} H={args.H:.3e} tox={args.tox:.3e} h~{args.h:.3e} deg={args.deg}")
        print(f"eps_ox={args.eps_ox} eps_si={args.eps_si}")
        print(f"Gate count={len(args.gate_Vs)}; a={args.a:.3e}")
        print(f"BC: top_ground=0, bottom={args.bc_bottom}, sides={args.bc_sides}")
        print(f"phi min={gmin:.6e}  max={gmax:.6e}")

    # Write XDMF
    with io.XDMFFile(comm, os.path.join(args.outdir, f"{args.basename}.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(uh)

        # also write eps (DG0) for sanity
        eps_out = fem.Function(Q)
        eps_out.x.array[:] = eps.x.array[:]
        eps_out.name = "eps_abs"
        xdmf.write_function(eps_out)

    # Probe line and (optionally) analytic comparison for tox=0
    if args.probe_line:
        # probe at fixed y, z
        y = args.probe_y
        if args.tox > 0:
            z = 0.0 + args.probe_z  # measured from top plane, still ok for probing
        else:
            z = args.probe_z

        xs = np.linspace(-args.Lx/2, args.Lx/2, args.probe_n)
        pts = np.column_stack([xs, y*np.ones_like(xs), z*np.ones_like(xs)]).astype(np.float64)

        vals = eval_function_on_points(uh, pts, comm)
        if comm.rank == 0:
            ok = np.isfinite(vals)
            if not np.all(ok):
                print(f"WARNING: {np.sum(~ok)} probe points returned NaN (outside mesh).")

            # Save CSV
            csv_path = os.path.join(args.outdir, f"{args.basename}_probe.csv")
            data = np.column_stack([xs, vals])
            np.savetxt(csv_path, data, delimiter=",", header="x,phi_num", comments="")
            print(f"Wrote probe CSV: {csv_path}")

            # Analytic check only meaningful for tox=0 and single medium
            if args.tox == 0.0:
                phi_an = phi_square_gates_row(xs, y, z, args.a, args.gate_xs, args.gate_Vs)
                # error metrics on interior region (avoid near Dirichlet side walls)
                pad = int(0.10 * args.probe_n)
                sl = slice(pad, -pad if pad > 0 else None)
                err = vals - phi_an
                maxerr = np.nanmax(np.abs(err[sl]))
                l2 = math.sqrt(np.nanmean(err[sl]**2))
                print(f"Analytic (half-space) comparison at y={y:.3e}, z={z:.3e}:")
                print(f"  max|phi_num-phi_analytic| (middle 80%) = {maxerr:.6e}")
                print(f"  RMS error (middle 80%)                 = {l2:.6e}")

                # write analytic CSV too
                csv_path2 = os.path.join(args.outdir, f"{args.basename}_probe_analytic.csv")
                data2 = np.column_stack([xs, vals, phi_an, err])
                np.savetxt(csv_path2, data2, delimiter=",", header="x,phi_num,phi_analytic,err", comments="")
                print(f"Wrote probe+analytic CSV: {csv_path2}")

    if comm.rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
