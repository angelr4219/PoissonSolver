#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from mpi4py import MPI

import gmsh
import ufl
from petsc4py import PETSc

from dolfinx import fem, io, mesh as dmesh

# geometry API moved across dolfinx versions
try:
    from dolfinx import geometry as dgeom
except Exception:
    import dolfinx.geometry as dgeom

# DOLFINx compatibility: gmsh import helper moved across versions
try:
    from dolfinx.io import gmshio as gmshio
except Exception:
    from dolfinx.io import gmsh as gmshio

from dolfinx.fem.petsc import LinearProblem as _LinearProblem

COMM = MPI.COMM_WORLD
RANK = COMM.rank


@dataclass(frozen=True)
class Gate:
    name: str
    xc: float
    yc: float
    a: float
    V: float


@dataclass(frozen=True)
class Tags:
    TOP_GROUND: int = 10
    BOTTOM: int = 11
    SIDES: int = 12
    GATE0: int = 100


def normalize_outdir(outdir: str) -> str:
    if outdir == "results":
        return "Results"
    if outdir.startswith("results/"):
        return "Results/" + outdir[len("results/") :]
    return outdir


def parse_float_list(s: str, expected: int, name: str) -> List[float]:
    if s is None:
        raise ValueError(f"{name} was None")
    s = s.strip()
    if not s:
        raise ValueError(f"{name} is empty")
    toks = s.replace(",", " ").split()
    try:
        vals = [float(t) for t in toks]
    except Exception as e:
        raise ValueError(f"Could not parse {name}='{s}' as floats. Error: {e}") from e
    if len(vals) != expected:
        raise ValueError(f"{name} must contain exactly {expected} numbers, got {len(vals)} from '{s}'")
    return vals


def _write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


def _degree_compat(deg_or_callable) -> int:
    try:
        return int(deg_or_callable())
    except TypeError:
        return int(deg_or_callable)


def function_for_xdmf(phi: fem.Function, msh: dmesh.Mesh) -> fem.Function:
    mesh_deg = _degree_compat(msh.geometry.cmap.degree)
    sol_deg = _degree_compat(phi.function_space.ufl_element().degree)
    if sol_deg == mesh_deg:
        return phi
    if RANK == 0:
        print(f"[INFO] XDMF requires degree match: mesh degree={mesh_deg}, phi degree={sol_deg}. Writing interpolated CG{mesh_deg}.")
    Vout = fem.functionspace(msh, ("CG", mesh_deg))
    phi_out = fem.Function(Vout, name=phi.name)
    try:
        phi_out.interpolate(phi)
    except Exception:
        expr = fem.Expression(phi, Vout.element.interpolation_points())
        phi_out.interpolate(expr)
    return phi_out


# -----------------------
# Analytic pads reference
# -----------------------
def g_rect(u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (1.0 / (2.0 * math.pi)) * np.arctan2(u * v, z * np.sqrt(u * u + v * v + z * z))


def phi0_three_squares(x: np.ndarray, y: np.ndarray, z: np.ndarray, gates: List[Gate]) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for gate in gates:
        a = gate.a
        xc, yc = gate.xc, gate.yc
        u1 = a + (x - xc)
        u2 = a - (x - xc)
        v1 = a + (y - yc)
        v2 = a - (y - yc)
        s = g_rect(u1, v1, z) + g_rect(u1, v2, z) + g_rect(u2, v1, z) + g_rect(u2, v2, z)
        out += -gate.V * s
    return out


# -----------------------
# Gmsh geometry
# -----------------------
def build_gmsh_model(
    mode: str,
    Lx: float,
    Ly: float,
    H: float,
    h: float,
    gates: List[Gate],
    gate_thickness: float,
    cavity_depth: float,
    occ_scale: float,
    model_name: str = "anderson_3gates",
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags, Dict[str, int]]:
    if mode not in ("pads", "extruded", "cavity"):
        raise ValueError("--mode must be pads|extruded|cavity")

    s = occ_scale
    Lx_s, Ly_s, H_s, h_s = Lx * s, Ly * s, H * s, h * s
    tg_s = gate_thickness * s
    dc_s = cavity_depth * s

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)
    gmsh.model.add(model_name)
    occ = gmsh.model.occ

    xmin, ymin, zmin = -0.5 * Lx_s, -0.5 * Ly_s, 0.0
    diel = occ.addBox(xmin, ymin, zmin, Lx_s, Ly_s, H_s)

    region_map: Dict[str, int] = {"DIELECTRIC": 1}

    if mode == "extruded":
        gate_dimtags = []
        gate_ids = []
        for i, g in enumerate(gates):
            a_s = g.a * s
            xc_s, yc_s = g.xc * s, g.yc * s
            gx0 = xc_s - a_s
            gy0 = yc_s - a_s
            gvol = occ.addBox(gx0, gy0, -tg_s, 2 * a_s, 2 * a_s, tg_s)
            gate_dimtags.append((3, gvol))
            gid = Tags.GATE0 + i
            gate_ids.append(gid)
            region_map[g.name] = gid

        occ.synchronize()

        in_dimtags = [(3, diel)] + gate_dimtags
        _, out_map = occ.fragment([in_dimtags[0]], in_dimtags[1:])
        occ.synchronize()

        def dim3_tags(dimtags: List[Tuple[int, int]]) -> List[int]:
            tags = [t for (d, t) in dimtags if d == 3]
            seen = set()
            out = []
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out

        diel_out = dim3_tags(out_map[0])
        gmsh.model.addPhysicalGroup(3, diel_out, region_map["DIELECTRIC"])
        gmsh.model.setPhysicalName(3, region_map["DIELECTRIC"], "DIELECTRIC")

        for i, g in enumerate(gates):
            gid = gate_ids[i]
            gout = dim3_tags(out_map[1 + i])
            if gout:
                gmsh.model.addPhysicalGroup(3, gout, gid)
                gmsh.model.setPhysicalName(3, gid, f"GATEVOL_{g.name}")

    elif mode == "cavity":
        tools = []
        for g in gates:
            a_s = g.a * s
            xc_s, yc_s = g.xc * s, g.yc * s
            gx0 = xc_s - a_s
            gy0 = yc_s - a_s
            cav = occ.addBox(gx0, gy0, 0.0, 2 * a_s, 2 * a_s, dc_s)
            tools.append((3, cav))

        occ.synchronize()
        out_dimtags, _ = occ.cut([(3, diel)], tools, removeObject=True, removeTool=True)
        occ.synchronize()

        diel_out = [t for (d, t) in out_dimtags if d == 3]
        if not diel_out:
            raise RuntimeError("Cut produced no dielectric volumes.")
        gmsh.model.addPhysicalGroup(3, diel_out, region_map["DIELECTRIC"])
        gmsh.model.setPhysicalName(3, region_map["DIELECTRIC"], "DIELECTRIC")

    else:
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [diel], region_map["DIELECTRIC"])
        gmsh.model.setPhysicalName(3, region_map["DIELECTRIC"], "DIELECTRIC")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_s)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_s)
    gmsh.model.mesh.generate(3)

    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh, ct, ft = out[0], out[1], out[2]

    gmsh.finalize()
    return msh, ct, ft, region_map


# -----------------------
# Facet tagging by centroid
# -----------------------
def facet_centroids(msh: dmesh.Mesh, fdim: int, facets: np.ndarray) -> np.ndarray:
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    X = msh.geometry.x
    C = np.zeros((facets.size, 3), dtype=np.float64)
    for i, fid in enumerate(facets.tolist()):
        vs = f2v.links(int(fid))
        C[i, :] = X[vs].mean(axis=0)
    return C


def make_facet_tags(
    msh: dmesh.Mesh,
    mode: str,
    H: float,
    gates: List[Gate],
    cavity_depth: float,
    tol: float,
) -> Tuple[dmesh.MeshTags, Dict[str, int], Dict[int, str], Dict[int, float]]:
    fdim = msh.topology.dim - 1
    tags = Tags()

    def is_bdry(x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[1], True, dtype=bool)

    bfacets = dmesh.locate_entities_boundary(msh, fdim, is_bdry).astype(np.int32)
    C = facet_centroids(msh, fdim, bfacets)
    x, y, z = C[:, 0], C[:, 1], C[:, 2]

    values = np.full(bfacets.size, tags.SIDES, dtype=np.int32)

    on_top = np.isclose(z, 0.0, atol=tol)
    values[on_top] = tags.TOP_GROUND

    on_bottom = np.isclose(z, H, atol=tol)
    values[on_bottom] = tags.BOTTOM

    name_to_tag: Dict[str, int] = {
        "TOP_GROUND": tags.TOP_GROUND,
        "BOTTOM": tags.BOTTOM,
        "SIDES": tags.SIDES,
    }
    tag_to_name: Dict[int, str] = {
        tags.TOP_GROUND: "TOP_GROUND",
        tags.BOTTOM: "BOTTOM",
        tags.SIDES: "SIDES",
    }
    tag_to_V: Dict[int, float] = {
        tags.TOP_GROUND: 0.0,
        tags.BOTTOM: 0.0,
        tags.SIDES: 0.0,
    }

    def in_square(g: Gate, xx: np.ndarray, yy: np.ndarray, pad: float) -> np.ndarray:
        return (np.abs(xx - g.xc) <= (g.a + pad)) & (np.abs(yy - g.yc) <= (g.a + pad))

    for i, g in enumerate(gates):
        gtag = tags.GATE0 + i
        name_to_tag[g.name] = gtag
        tag_to_name[gtag] = g.name
        tag_to_V[gtag] = float(g.V)

        inside_xy = in_square(g, x, y, pad=tol)

        if mode == "pads":
            mask = inside_xy & on_top
        elif mode == "cavity":
            # cavity walls + floor region (z in [0, cavity_depth])
            mask = inside_xy & (z >= -tol) & (z <= cavity_depth + tol)
        elif mode == "extruded":
            # extruded gate outer surfaces live at z<0
            mask = inside_xy & (z < -0.5 * tol)
        else:
            raise ValueError("bad mode")

        values[mask] = gtag

    sort = np.argsort(bfacets)
    mt = dmesh.meshtags(msh, fdim, bfacets[sort], values[sort])
    return mt, name_to_tag, tag_to_name, tag_to_V


# -----------------------
# Solve
# -----------------------
def solve_with_dirichlet_tags(
    msh: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
    tag_to_V: Dict[int, float],
    deg: int,
    eps_r: float,
    rho: float,
) -> fem.Function:
    V = fem.functionspace(msh, ("CG", deg))
    phi = fem.Function(V, name="phi")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps0 = 8.8541878128e-12
    eps = fem.Constant(msh, PETSc.ScalarType(eps0 * eps_r))
    rhs = fem.Constant(msh, PETSc.ScalarType(rho))

    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rhs * v * ufl.dx

    fdim = msh.topology.dim - 1
    bcs = []

    unique = np.unique(facet_tags.values)
    for t in unique.tolist():
        t = int(t)
        facets_t = facet_tags.indices[facet_tags.values == t]
        if facets_t.size == 0:
            continue
        dofs_t = fem.locate_dofs_topological(V, fdim, facets_t)
        Vt = float(tag_to_V.get(t, 0.0))
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vt), dofs_t, V))

    problem = _LinearProblem(
        a, L, bcs=bcs, u=phi,
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-12,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 20000,
        },
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


# -----------------------
# Top view helper
# -----------------------
def write_top_view_gate_id(
    msh: dmesh.Mesh,
    gates: List[Gate],
    name_to_tag: Dict[str, int],
    tol: float,
    outpath: str,
) -> None:
    fdim = msh.topology.dim - 1

    def is_top(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[2], 0.0, atol=tol)

    top_facets = dmesh.locate_entities_boundary(msh, fdim, is_top)
    sub = dmesh.create_submesh(msh, fdim, top_facets)
    smesh = sub[0]

    tdim = smesh.topology.dim
    smesh.topology.create_connectivity(tdim, 0)
    c2v = smesh.topology.connectivity(tdim, 0)

    V0 = fem.functionspace(smesh, ("DG", 0))
    gate_id = fem.Function(V0, name="gate_id")

    coords = smesh.geometry.x
    im = smesh.topology.index_map(tdim)
    ncells = im.size_local + im.num_ghosts
    ncells = min(ncells, gate_id.x.array.size)

    bg = name_to_tag.get("TOP_GROUND", 0)
    vals = np.full(ncells, bg, dtype=gate_id.x.array.dtype)

    for c in range(ncells):
        vs = c2v.links(c)
        xc = coords[vs].mean(axis=0)
        xx, yy = float(xc[0]), float(xc[1])
        tag_here = bg
        for g in gates:
            if (abs(xx - g.xc) <= g.a + tol) and (abs(yy - g.yc) <= g.a + tol):
                tag_here = name_to_tag[g.name]
        vals[c] = tag_here

    gate_id.x.array[:ncells] = vals

    with io.XDMFFile(COMM, outpath, "w") as xdmf:
        xdmf.write_mesh(smesh)
        xdmf.write_function(gate_id)


# -----------------------
# Point eval compatibility
# -----------------------
def _bb_tree(mesh: dmesh.Mesh, dim: int):
    try:
        return dmesh.create_bounding_box_tree(mesh, dim)  # type: ignore[attr-defined]
    except Exception:
        return dgeom.bb_tree(mesh, dim)


def _collisions_points(tree, points: np.ndarray):
    try:
        return dmesh.compute_collisions_points(tree, points)  # type: ignore[attr-defined]
    except Exception:
        return dgeom.compute_collisions_points(tree, points)


def _colliding_cells(mesh: dmesh.Mesh, candidates, points: np.ndarray):
    try:
        return dmesh.compute_colliding_cells(mesh, candidates, points)  # type: ignore[attr-defined]
    except Exception:
        return dgeom.compute_colliding_cells(mesh, candidates, points)


def eval_function_at_points(u: fem.Function, pts: np.ndarray) -> np.ndarray:
    msh = u.function_space.mesh
    tree = _bb_tree(msh, msh.topology.dim)
    cands = _collisions_points(tree, pts)
    cells_links = _colliding_cells(msh, cands, pts)

    cells = np.full(pts.shape[0], -1, dtype=np.int32)
    for i in range(pts.shape[0]):
        links = cells_links.links(i)
        if len(links) > 0:
            cells[i] = int(links[0])

    vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    ok = cells >= 0
    if np.any(ok):
        v = u.eval(pts[ok], cells[ok])
        vals[ok] = np.asarray(v).reshape(-1)
    return vals


def sample_line_phi(phi: fem.Function, xs: np.ndarray, y0: float, z0: float) -> np.ndarray:
    pts = np.column_stack([xs, np.full_like(xs, y0), np.full_like(xs, z0)]).astype(np.float64)
    return eval_function_at_points(phi, pts)


def interpolate_values_onto_space(V: fem.FunctionSpace, values: np.ndarray) -> fem.Function:
    f = fem.Function(V)
    # For CG spaces, tabulate_dof_coordinates gives one point per dof (for Lagrange)
    f.x.array[:] = values.astype(f.x.array.dtype, copy=False)
    return f


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Anderson-style 3-gate benchmark: pads vs extruded gates vs embedded cavities."
    )
    ap.add_argument("--mode", choices=["pads", "extruded", "cavity"], default="pads")
    ap.add_argument("--outdir", type=str, default="Results/anderson_3gates")
    ap.add_argument("--basename", type=str, default="anderson_3gates")

    ap.add_argument("--Lx", type=float, default=3.0e-7)
    ap.add_argument("--Ly", type=float, default=3.0e-7)
    ap.add_argument("--H", type=float, default=2.0e-7)

    ap.add_argument("--a", type=float, default=3.5e-8)

    # IMPORTANT: pass these with quotes; parser expects a single string
    ap.add_argument("--gate_xs", type=str, default="-7.0e-8,0,7.0e-8")
    ap.add_argument("--gate_ys", type=str, default="0,0,0")
    ap.add_argument("--gate_Vs", type=str, default="0.25,0.10,0.25")

    ap.add_argument("--gate_thickness", type=float, default=2.0e-8)
    ap.add_argument("--cavity_depth", type=float, default=2.0e-8)

    ap.add_argument("--h", type=float, default=1.0e-8)
    ap.add_argument("--deg", type=int, default=2)
    ap.add_argument("--eps_r", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.0)

    # internal: scale geometry so OCC gets ~O(1e2..1e3) sizes
    ap.add_argument("--occ_scale", type=float, default=1e9)

    # sampling
    ap.add_argument("--sample_y", type=float, default=0.0)
    ap.add_argument("--sample_z", type=float, default=-1.0)
    ap.add_argument("--sample_xmax", type=float, default=-1.0)
    ap.add_argument("--nsample", type=int, default=401)
    ap.add_argument("--tau", type=float, default=1e-9)

    # for extruded/cavity, also solve pads and write delta_phi on THIS mesh
    ap.add_argument("--also_solve_pads", action="store_true")

    args = ap.parse_args()
    args.outdir = normalize_outdir(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    gate_xs = parse_float_list(args.gate_xs, expected=3, name="--gate_xs")
    gate_ys = parse_float_list(args.gate_ys, expected=3, name="--gate_ys")
    gate_Vs = parse_float_list(args.gate_Vs, expected=3, name="--gate_Vs")

    gates: List[Gate] = []
    for i in range(3):
        gates.append(Gate(name=f"G{i}", xc=float(gate_xs[i]), yc=float(gate_ys[i]), a=float(args.a), V=float(gate_Vs[i])))

    sample_z = float(args.a) if args.sample_z < 0 else float(args.sample_z)
    sample_xmax = 2.0 * float(args.a) if args.sample_xmax < 0 else float(args.sample_xmax)

    if RANK == 0:
        print("\n=== Benchmark config ===")
        print(f"mode={args.mode}")
        print(f"Box: Lx={args.Lx:.3e}, Ly={args.Ly:.3e}, H={args.H:.3e}")
        print(f"a={args.a:.3e}")
        print(f"gate_xs={gate_xs}")
        print(f"gate_ys={gate_ys}")
        print(f"gate_Vs={gate_Vs}")
        print(f"h={args.h:.3e}, deg={args.deg}, eps_r={args.eps_r:g}, rho={args.rho:g}")
        print(f"sample line: y={args.sample_y:g}, z={sample_z:.3e}, x in [-{sample_xmax:.3e}, +{sample_xmax:.3e}]")
        print(f"occ_scale={args.occ_scale:g}")

    # Build primary mesh for selected mode
    msh, ct, ft, region_map = build_gmsh_model(
        mode=args.mode,
        Lx=args.Lx, Ly=args.Ly, H=args.H,
        h=args.h,
        gates=gates,
        gate_thickness=args.gate_thickness,
        cavity_depth=args.cavity_depth,
        occ_scale=args.occ_scale,
        model_name=f"anderson_3gates_{args.mode}",
    )
    msh.geometry.x[:] *= (1.0 / args.occ_scale)

    tol = 1e-10 * max(args.Lx, args.Ly, args.H, 1.0)

    facet_tags, name_to_tag, tag_to_name, tag_to_V = make_facet_tags(
        msh=msh,
        mode=args.mode,
        H=args.H,
        gates=gates,
        cavity_depth=args.cavity_depth,
        tol=tol,
    )

    if RANK == 0:
        print("\n=== Facet tag summary (centroid classification) ===")
        for t in sorted(np.unique(facet_tags.values).tolist()):
            t = int(t)
            n = int(np.sum(facet_tags.values == t))
            nm = tag_to_name.get(t, f"tag{t}")
            vv = tag_to_V.get(t, 0.0)
            print(f"  {nm:8s} tag={t:4d} nfacets={n:6d} V={vv:g}")

    phi = solve_with_dirichlet_tags(
        msh=msh,
        facet_tags=facet_tags,
        tag_to_V=tag_to_V,
        deg=args.deg,
        eps_r=args.eps_r,
        rho=args.rho,
    )

    local_min = float(np.min(phi.x.array)) if phi.x.array.size else 0.0
    local_max = float(np.max(phi.x.array)) if phi.x.array.size else 0.0
    gmin = COMM.allreduce(local_min, op=MPI.MIN)
    gmax = COMM.allreduce(local_max, op=MPI.MAX)

    if RANK == 0:
        print("\n=== Solve summary ===")
        print(f"phi min/max = [{gmin:.6e}, {gmax:.6e}]")

    base = os.path.join(args.outdir, args.basename + f"_{args.mode}")
    phi_path = base + "_phi.xdmf"
    tags_path = base + "_facet_tags.xdmf"
    top_path = base + "_top_view_gate_id.xdmf"
    map_path = base + "_tag_map.txt"
    line_csv = base + "_line_sample.csv"

    phi_out = function_for_xdmf(phi, msh)

    with io.XDMFFile(COMM, phi_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_out)

    with io.XDMFFile(COMM, tags_path, "w") as xdmf:
        xdmf.write_mesh(msh)
        _write_meshtags_compat(xdmf, facet_tags, msh)

    write_top_view_gate_id(msh, gates, name_to_tag, tol=tol, outpath=top_path)

    if RANK == 0:
        with open(map_path, "w", encoding="utf-8") as f:
            f.write("Facet tag map (centroid classification)\n")
            f.write(f"mode = {args.mode}\n\n")
            for t in sorted(tag_to_name.keys()):
                f.write(f"tag {t:4d}  name={tag_to_name[t]:8s}  V={tag_to_V.get(t, 0.0): .6g}\n")
            f.write("\nGates:\n")
            for g in gates:
                f.write(f"  {g.name}: center=({g.xc:.6e},{g.yc:.6e}), a={g.a:.6e}, V={g.V:.6g}\n")

    # Line sample on this mode mesh
    xs = np.linspace(-sample_xmax, sample_xmax, args.nsample)
    yh = float(args.sample_y)
    zh = float(sample_z)
    phi_line = sample_line_phi(phi, xs, yh, zh)

    # Pads-only analytic reference (only meaningful for pads mode)
    phi0_line = None
    relerr_line = None
    if args.mode == "pads":
        phi0_line = phi0_three_squares(xs, np.full_like(xs, yh), np.full_like(xs, zh), gates)
        tau = float(args.tau)
        relerr_line = np.abs(phi_line - phi0_line) / np.maximum(np.abs(phi0_line), tau)

    if RANK == 0:
        with open(line_csv, "w", encoding="utf-8") as f:
            if phi0_line is None:
                f.write("x,y,z,phi\n")
                for i in range(xs.size):
                    f.write(f"{xs[i]:.9e},{yh:.9e},{zh:.9e},{phi_line[i]:.9e}\n")
            else:
                f.write("x,y,z,phi,phi0,relerr\n")
                for i in range(xs.size):
                    f.write(f"{xs[i]:.9e},{yh:.9e},{zh:.9e},{phi_line[i]:.9e},{phi0_line[i]:.9e},{relerr_line[i]:.9e}\n")

    # ---- ALSO SOLVE PADS AND WRITE delta_phi ON CURRENT MESH ----
    # This gives you the “realistic 3D gates change the field” artifact.
    if args.also_solve_pads and args.mode != "pads":
        if RANK == 0:
            print("\n[INFO] also_solve_pads enabled: building pads reference mesh + solving...")

        msh_p, ct_p, ft_p, _ = build_gmsh_model(
            mode="pads",
            Lx=args.Lx, Ly=args.Ly, H=args.H,
            h=args.h,
            gates=gates,
            gate_thickness=args.gate_thickness,
            cavity_depth=args.cavity_depth,
            occ_scale=args.occ_scale,
            model_name="anderson_3gates_pads_ref",
        )
        msh_p.geometry.x[:] *= (1.0 / args.occ_scale)

        facet_tags_p, name_to_tag_p, tag_to_name_p, tag_to_V_p = make_facet_tags(
            msh=msh_p,
            mode="pads",
            H=args.H,
            gates=gates,
            cavity_depth=args.cavity_depth,
            tol=tol,
        )
        phi_pads = solve_with_dirichlet_tags(
            msh=msh_p,
            facet_tags=facet_tags_p,
            tag_to_V=tag_to_V_p,
            deg=args.deg,
            eps_r=args.eps_r,
            rho=args.rho,
        )
        phi_pads.name = "phi_pads"

        # Evaluate pads solution at CURRENT mesh dof coordinates (only where z>=0)
        Vcur = phi.function_space
        dof_xyz = Vcur.tabulate_dof_coordinates()
        pts = np.array(dof_xyz, dtype=np.float64)

        vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
        inside = pts[:, 2] >= -1e-15  # pads ref domain is z>=0
        if np.any(inside):
            vals[inside] = eval_function_at_points(phi_pads, pts[inside])

        # Replace any remaining NaNs with 0 (primarily gate volumes in extruded mode)
        vals = np.where(np.isfinite(vals), vals, 0.0)

        phi_pads_on_cur = fem.Function(Vcur, name="phi_pads_on_this_mesh")
        phi_pads_on_cur.x.array[:] = vals.astype(phi_pads_on_cur.x.array.dtype, copy=False)

        delta = fem.Function(Vcur, name="delta_phi")
        delta.x.array[:] = phi.x.array - phi_pads_on_cur.x.array

        # Write extra fields
        pads_on_cur_path = base + "_phi_pads_on_this_mesh.xdmf"
        delta_path = base + "_delta_phi.xdmf"
        delta_line_csv = base + "_delta_line_sample.csv"

        with io.XDMFFile(COMM, pads_on_cur_path, "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(function_for_xdmf(phi_pads_on_cur, msh))

        with io.XDMFFile(COMM, delta_path, "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(function_for_xdmf(delta, msh))

        # Compare along the same Anderson-style sampling line too
        phi_pads_line = sample_line_phi(phi_pads, xs, yh, zh)
        delta_line = phi_line - phi_pads_line

        if RANK == 0:
            with open(delta_line_csv, "w", encoding="utf-8") as f:
                f.write("x,y,z,phi_mode,phi_pads,delta\n")
                for i in range(xs.size):
                    f.write(f"{xs[i]:.9e},{yh:.9e},{zh:.9e},{phi_line[i]:.9e},{phi_pads_line[i]:.9e},{delta_line[i]:.9e}\n")

            print("[INFO] Wrote pads-on-current + delta_phi:")
            print(f"  {pads_on_cur_path}")
            print(f"  {delta_path}")
            print(f"  {delta_line_csv}")

    if RANK == 0:
        print("\n=== Wrote files ===")
        print(f"  phi:        {phi_path}")
        print(f"  facet tags: {tags_path}")
        print(f"  top view:   {top_path}")
        print(f"  tag map:    {map_path}")
        print(f"  line CSV:   {line_csv}")
        print("\nParaView:")
        print("  - Open *_top_view_gate_id.xdmf for the layout (2D top plane).")
        print("  - Open *_facet_tags.xdmf and color by 'gmsh:physical' (your tag integers).")
        print("  - Open *_phi.xdmf and color by phi.")
        if args.also_solve_pads and args.mode != "pads":
            print("  - Open *_delta_phi.xdmf and color by delta_phi (mode - pads reference on this mesh).")


if __name__ == "__main__":
    main()
