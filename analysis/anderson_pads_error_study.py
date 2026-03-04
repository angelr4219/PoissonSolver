#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from petsc4py import PETSc

from dolfinx import fem, io, mesh as dmesh

# DOLFINx compatibility: gmsh import helper moved across versions
try:
    from dolfinx.io import gmshio as gmshio
except Exception:
    from dolfinx.io import gmsh as gmshio

from dolfinx.fem.petsc import LinearProblem

COMM = MPI.COMM_WORLD
RANK = COMM.rank

# ----------------------------
# Analytic rectangular-pad potential (Anderson-style)
# ----------------------------
def g_rect(u: np.ndarray, v: np.ndarray, z: float) -> np.ndarray:
    """
    g(u,v,z) = (1/2pi) * atan2( u*v, z*sqrt(u^2+v^2+z^2) )
    """
    return (1.0 / (2.0 * math.pi)) * np.arctan2(u * v, z * np.sqrt(u * u + v * v + z * z))


def phi0_three_squares(
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    a: float,
    gate_xs: List[float],
    gate_ys: List[float],
    gate_Vs: List[float],
) -> np.ndarray:
    """
    Anderson-style formula for square pads (half-width a) centered at (xi, yi) on plane z=0.
    Overall minus sign matches the common rectangle formula conventions used in the benchmark.
    """
    out = np.zeros_like(x, dtype=float)
    for (xi, yi, Vi) in zip(gate_xs, gate_ys, gate_Vs):
        # Shift coordinates so pad is centered at (xi, yi)
        X = x - xi
        Y = y - yi

        term = (
            g_rect(a + X, a + Y, z)
            + g_rect(a + X, a - Y, z)
            + g_rect(a - X, a + Y, z)
            + g_rect(a - X, a - Y, z)
        )
        out -= Vi * term
    return out


# ----------------------------
# Gmsh mesh builder (version-safe model_to_mesh unpacking)
# ----------------------------
def build_box_mesh_gmsh(Lx: float, Ly: float, H: float, h: float, model_name: str = "anderson_box"):
    """
    Build a 3D box domain:
      x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2], z in [0, H]
    Returns (mesh, cell_tags, facet_tags). Tags may be None depending on DOLFINx version.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if RANK == 0 else 0)
    gmsh.model.add(model_name)
    occ = gmsh.model.occ

    x0, y0, z0 = -0.5 * Lx, -0.5 * Ly, 0.0
    box = occ.addBox(x0, y0, z0, Lx, Ly, H)
    occ.synchronize()

    VOL = 1
    gmsh.model.addPhysicalGroup(3, [box], VOL)
    gmsh.model.setPhysicalName(3, VOL, "DOMAIN")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(3)

    # DOLFINx version compatibility: model_to_mesh returns 3 or 4 values depending on version
    out = gmshio.model_to_mesh(gmsh.model, COMM, rank=0, gdim=3)
    msh = out[0]
    ct = out[1] if len(out) > 1 else None
    ft = out[2] if len(out) > 2 else None

    gmsh.finalize()
    return msh, ct, ft


# ----------------------------
# Pad BC tagging by centroid classification on top plane
# ----------------------------
@dataclass(frozen=True)
class RectPad:
    xc: float
    yc: float
    a: float  # half-width

    @property
    def xmin(self) -> float:
        return self.xc - self.a

    @property
    def xmax(self) -> float:
        return self.xc + self.a

    @property
    def ymin(self) -> float:
        return self.yc - self.a

    @property
    def ymax(self) -> float:
        return self.yc + self.a


def tag_top_facets_centroid(
    msh: dmesh.Mesh,
    pads: List[RectPad],
    z_top: float,
    tol_z: float,
    tol_xy: float,
    tag_top_ground: int = 10,
    tag_gate0: int = 100,
) -> Tuple[dmesh.MeshTags, List[int]]:
    """
    Create facet MeshTags on boundary facets:
      - Top plane facets z=z_top get either TOP_GROUND or a gate tag (G0,G1,...)
      - Everything else is left untagged here (solver will still apply 0 Dirichlet on sides/bottom separately)

    Returns:
      mt : MeshTags on all top facets
      gate_tags : list of tags used for the gates (same order as pads)
    """
    fdim = msh.topology.dim - 1

    def is_top(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[2], z_top, atol=tol_z)

    top_facets = dmesh.locate_entities_boundary(msh, fdim, is_top)
    if top_facets.size == 0:
        raise RuntimeError("No top facets found (check z_top/tol_z).")

    # Default to top ground
    values = np.full(top_facets.shape, tag_top_ground, dtype=np.int32)
    gate_tags = [tag_gate0 + i for i in range(len(pads))]

    # We'll classify each top facet by its centroid
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    X = msh.geometry.x

    for k, f in enumerate(top_facets):
        vs = f2v.links(int(f))
        xc = X[vs].mean(axis=0)
        x, y = float(xc[0]), float(xc[1])

        t = tag_top_ground
        for i, pad in enumerate(pads):
            if (x >= pad.xmin - tol_xy) and (x <= pad.xmax + tol_xy) and (y >= pad.ymin - tol_xy) and (y <= pad.ymax + tol_xy):
                t = gate_tags[i]
                break
        values[k] = t

    # Sort for stable meshtags creation
    order = np.argsort(top_facets)
    mt = dmesh.meshtags(msh, fdim, top_facets[order], values[order])
    return mt, gate_tags


# ----------------------------
# Solve Laplace/Poisson with Dirichlet on pads + rest 0
# ----------------------------
def solve_with_pad_dirichlet(
    msh: dmesh.Mesh,
    mt_top: dmesh.MeshTags,
    pad_tags: List[int],
    pad_Vs: List[float],
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

    a_form = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = rhs * v * ufl.dx

    fdim = msh.topology.dim - 1

    # Boundary facets for "everywhere" zero Dirichlet:
    # We'll set bc=0 on ALL boundary dofs first, then override pads by stronger per-tag bcs.
    def on_boundary(x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[1], True, dtype=bool)

    bfacets = dmesh.locate_entities_boundary(msh, fdim, on_boundary)
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), bdofs, V)]

    # Now pad-specific Dirichlet on top-plane tag subsets
    # Get facets for each tag from mt_top
    for tag, Vgate in zip(pad_tags, pad_Vs):
        facets = mt_top.indices[mt_top.values == tag]
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(Vgate), dofs, V))

    problem = LinearProblem(
        a_form,
        L_form,
        bcs=bcs,
        u=phi,
        petsc_options_prefix="anderson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-12,
            "ksp_atol": 1.0e-14,
        },
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


# ----------------------------
# Sampling + errors
# ----------------------------
def sample_line_points(xmin: float, xmax: float, n: int, y: float, z: float) -> np.ndarray:
    xs = np.linspace(xmin, xmax, n)
    pts = np.zeros((n, 3), dtype=float)
    pts[:, 0] = xs
    pts[:, 1] = y
    pts[:, 2] = z
    return pts


def eval_phi_on_points(phi: fem.Function, pts: np.ndarray) -> np.ndarray:
    """
    Evaluate phi at points.

    DOLFINx versions differ on Function.eval() point-shape:
      - some expect x.shape == (N, 3)
      - others expect x.shape == (3, N)

    This routine tries (N,3) first, then falls back to (3,N).
    """
    msh = phi.function_space.mesh
    tdim = msh.topology.dim

    # Find containing cells for each point
    cells = np.full(pts.shape[0], -1, dtype=np.int32)

    try:
        from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
        tree = bb_tree(msh, tdim)
        cand = compute_collisions_points(tree, pts)
        coll = compute_colliding_cells(msh, cand, pts)
        for i in range(pts.shape[0]):
            cands = coll.links(i)
            if len(cands) > 0:
                cells[i] = cands[0]
    except Exception:
        # Older fallback API
        from dolfinx import geometry as geom
        tree = geom.BoundingBoxTree(msh, tdim)
        for i in range(pts.shape[0]):
            cands = geom.compute_collisions_point(tree, pts[i])
            if len(cands) > 0:
                cells[i] = cands[0]

    vals = np.full(pts.shape[0], np.nan, dtype=float)
    valid = np.where(cells >= 0)[0]
    if valid.size == 0:
        return vals

    pts_valid = np.asarray(pts[valid], dtype=np.float64)
    cells_valid = np.asarray(cells[valid], dtype=np.int32)

    # Try new-style: (N,3)
    try:
        out = phi.eval(pts_valid, cells_valid)
    except Exception:
        # Fallback old-style: (3,N)
        out = phi.eval(pts_valid.T.copy(), cells_valid)

    out = np.asarray(out).reshape(-1)
    vals[valid] = out
    return vals

    # dolfinx expects points shaped (3, N)
    pT = pts[valid].T.copy()
    c = cells[valid].copy()
    out = phi.eval(pT, c)
    # out may be shape (N, 1) or (N,) depending on version
    out = np.asarray(out).reshape(-1)
    vals[valid] = out
    return vals


def relerr(phi_h: np.ndarray, phi0: np.ndarray, tau: float) -> np.ndarray:
    denom = np.maximum(np.abs(phi0), tau)
    return np.abs(phi_h - phi0) / denom


# ----------------------------
# Main: h/p sweep for pads only (analytic reference)
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Anderson pads: h/p error study against analytic rectangle formula.")
    ap.add_argument("--outdir", type=str, default="Results/anderson_error_study", help="Output root.")
    ap.add_argument("--basename", type=str, default="anderson_pads", help="Basename for outputs.")

    ap.add_argument("--Lx", type=float, default=3.0e-7)
    ap.add_argument("--Ly", type=float, default=3.0e-7)
    ap.add_argument("--H", type=float, default=2.0e-7)

    ap.add_argument("--a", type=float, default=3.5e-8)

    # Provide lists as ONE quoted string for robustness, but allow space-separated too
    ap.add_argument("--gate_xs", type=str, default="-7.0e-8 0.0 7.0e-8")
    ap.add_argument("--gate_ys", type=str, default="0.0 0.0 0.0")
    ap.add_argument("--gate_Vs", type=str, default="0.25 0.10 0.25")

    ap.add_argument("--hs", type=str, default="1.5e-8 1.0e-8 7.5e-9", help="Space-separated h values.")
    ap.add_argument("--degs", type=str, default="1 2 3", help="Space-separated polynomial degrees.")

    ap.add_argument("--eps_r", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.0)

    # Sampling line settings
    ap.add_argument("--sample_y", type=float, default=0.0)
    ap.add_argument("--sample_z", type=float, default=3.5e-8, help="Depth z=a by default.")
    ap.add_argument("--sample_xmax", type=float, default=7.0e-8, help="Sample |x|<=2a by default.")
    ap.add_argument("--nsample", type=int, default=401)

    ap.add_argument("--tau", type=float, default=1.0e-9, help="Regularization for relative error denom.")
    args = ap.parse_args()

    def parse_list(s: str) -> List[float]:
        # allow comma or spaces
        s = s.replace(",", " ")
        return [float(x) for x in s.split() if x.strip()]

    gate_xs = parse_list(args.gate_xs)
    gate_ys = parse_list(args.gate_ys)
    gate_Vs = parse_list(args.gate_Vs)
    if not (len(gate_xs) == len(gate_ys) == len(gate_Vs)):
        raise ValueError("gate_xs, gate_ys, gate_Vs must have same length")

    hs = parse_list(args.hs)
    degs = [int(x) for x in args.degs.replace(",", " ").split() if x.strip()]

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{args.basename}_hp_table.csv")
    line_dir = os.path.join(args.outdir, f"{args.basename}_lines")
    os.makedirs(line_dir, exist_ok=True)

    if RANK == 0:
        print("\n=== Anderson pads error study ===")
        print(f"Domain: x,y in [-L/2,L/2], z in [0,H]")
        print(f"Lx={args.Lx:g}  Ly={args.Ly:g}  H={args.H:g}")
        print(f"a={args.a:g}")
        print(f"gates: xs={gate_xs} ys={gate_ys} Vs={gate_Vs}")
        print(f"hs={hs}")
        print(f"degs={degs}")
        print(f"sample line: y={args.sample_y:g}, z={args.sample_z:g}, x in [-{args.sample_xmax:g},{args.sample_xmax:g}], N={args.nsample}")
        print("")

    # Build sample points (global)
    pts = sample_line_points(-args.sample_xmax, args.sample_xmax, args.nsample, args.sample_y, args.sample_z)
    x = pts[:, 0]
    y = pts[:, 1]
    phi0 = phi0_three_squares(x, y, args.sample_z, args.a, gate_xs, gate_ys, gate_Vs)

    # Run sweep
    rows = []
    for h in hs:
        for deg in degs:
            if RANK == 0:
                print("--------------------------------------------------")
                print(f"[RUN] h={h:g}  deg={deg}")
            msh, ct, ft = build_box_mesh_gmsh(args.Lx, args.Ly, args.H, h, model_name=f"box_h{h:g}_p{deg}")

            # Gate pads on top plane z=0
            pads = [RectPad(xc=xi, yc=yi, a=args.a) for (xi, yi) in zip(gate_xs, gate_ys)]
            Lref = max(args.Lx, args.Ly, args.H, args.a)
            tol_z = 1.0e-12 * Lref + 1.0e-15
            tol_xy = 1.0e-12 * Lref + 1.0e-15

            mt_top, gate_tags = tag_top_facets_centroid(
                msh, pads=pads, z_top=0.0, tol_z=tol_z, tol_xy=tol_xy, tag_top_ground=10, tag_gate0=100
            )

            phi = solve_with_pad_dirichlet(
                msh, mt_top=mt_top, pad_tags=gate_tags, pad_Vs=gate_Vs, deg=deg, eps_r=args.eps_r, rho=args.rho
            )

            # Evaluate along line
            phi_h = eval_phi_on_points(phi, pts)
            r = relerr(phi_h, phi0, args.tau)

            # Metrics on valid points
            valid = np.isfinite(phi_h)
            if valid.any():
                r_max = float(np.max(r[valid]))
                r_mean = float(np.mean(r[valid]))
                r_l2 = float(np.sqrt(np.mean(r[valid] ** 2)))
            else:
                r_max = float("nan")
                r_mean = float("nan")
                r_l2 = float("nan")

            # Print min/max
            local_min = float(np.nanmin(phi_h))
            local_max = float(np.nanmax(phi_h))
            gmin = COMM.allreduce(local_min, op=MPI.MIN)
            gmax = COMM.allreduce(local_max, op=MPI.MAX)
            if RANK == 0:
                print(f"phi_h line min/max = [{gmin:.6e}, {gmax:.6e}]")
                print(f"relerr: mean={r_mean:.3e}  L2={r_l2:.3e}  max={r_max:.3e}")

            # Save line CSV for this run (rank0 writes)
            if RANK == 0:
                line_csv = os.path.join(line_dir, f"line_h{h:g}_p{deg}.csv".replace("+", ""))
                data = np.column_stack([x, phi0, phi_h, r])
                header = "x,phi0,phi_h,relerr"
                np.savetxt(line_csv, data, delimiter=",", header=header, comments="")
                print(f"[WROTE] {line_csv}")

            rows.append((h, deg, r_mean, r_l2, r_max))

    # Write summary CSV (rank0)
    if RANK == 0:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("h,deg,relerr_mean,relerr_L2,relerr_max\n")
            for (h, deg, r_mean, r_l2, r_max) in rows:
                f.write(f"{h:.16e},{deg:d},{r_mean:.16e},{r_l2:.16e},{r_max:.16e}\n")
        print("")
        print(f"[DONE] Summary table: {csv_path}")
        print(f"[DONE] Line CSVs:      {line_dir}")
        print("Tip: smallest errors should appear as h decreases and/or deg increases.")

if __name__ == "__main__":
    main()
