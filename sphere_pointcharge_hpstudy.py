#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from typing import List

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

EPS0 = 8.8541878128e-12  # F/m


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def phi_uniform_sphere(r, q, R, eps):
    r = np.asarray(r)
    out = np.empty_like(r)
    outside = r >= R
    inside = ~outside
    out[outside] = q / (4 * math.pi * eps * r[outside])
    out[inside] = q / (8 * math.pi * eps * R) * (3 - (r[inside] / R) ** 2)
    return out


def build_mesh(comm, L, h_target):
    n = max(2, int(math.ceil(L / h_target)))
    h_eff = L / n
    m = mesh.create_box(
        comm,
        [np.array([-L/2]*3), np.array([L/2]*3)],
        [n, n, n],
        cell_type=mesh.CellType.tetrahedron,
    )
    return m, n, h_eff


def exterior_facets(m):
    tdim = m.topology.dim
    fdim = tdim - 1
    m.topology.create_connectivity(fdim, tdim)
    try:
        return mesh.exterior_facet_indices(m.topology)
    except AttributeError:
        from dolfinx.cpp.mesh import exterior_facet_indices
        return exterior_facet_indices(m.topology)


def make_rho(m, q, R, center):
    W = fem.functionspace(m, ("DG", 0))
    rho = fem.Function(W)
    tdim = m.topology.dim
    nc = m.topology.index_map(tdim).size_local
    cells = np.arange(nc, dtype=np.int32)
    mids = mesh.compute_midpoints(m, tdim, cells)
    r = np.linalg.norm(mids - center[None, :], axis=1)
    rho0 = q / ((4/3) * math.pi * R**3)
    rho.x.array[:] = np.where(r <= R, rho0, 0.0)
    return rho


def make_phi_exact(V, q, R, eps, center):
    phi = fem.Function(V)
    def eval(x):
        r = np.sqrt((x[0]-center[0])**2 +
                    (x[1]-center[1])**2 +
                    (x[2]-center[2])**2)
        return phi_uniform_sphere(r, q, R, eps)
    phi.interpolate(eval)
    return phi


def assemble(comm, expr):
    return comm.allreduce(
        fem.assemble_scalar(fem.form(expr)),
        MPI.SUM
    )


@dataclass
class Result:
    p: int
    n: int
    h: float
    l2: float
    h1: float
    centroid: np.ndarray


def solve_one(comm, L, h_target, p, q, R, epsr, center):
    m, n, h_eff = build_mesh(comm, L, h_target)
    V = fem.functionspace(m, ("Lagrange", p))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    eps = EPS0 * epsr
    rho = make_rho(m, q, R, center)
    phi_ex = make_phi_exact(V, q, R, eps, center)

    facets = exterior_facets(m)
    dofs = fem.locate_dofs_topological(V, m.topology.dim-1, facets)
    bc = fem.dirichletbc(phi_ex, dofs)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    Lf = rho * v * ufl.dx

    problem = LinearProblem(
        a, Lf,
        bcs=[bc],
        petsc_options_prefix="sphere_"
    )
    phi = problem.solve()

    e = phi - phi_ex
    l2 = math.sqrt(assemble(comm, ufl.inner(e, e) * ufl.dx))
    h1 = math.sqrt(assemble(comm, ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))

    x = ufl.SpatialCoordinate(m)
    w = ufl.inner(e, e)
    I0 = assemble(comm, w * ufl.dx)
    Ix = assemble(comm, w * x[0] * ufl.dx)
    Iy = assemble(comm, w * x[1] * ufl.dx)
    Iz = assemble(comm, w * x[2] * ufl.dx)
    centroid = np.array([Ix/I0, Iy/I0, Iz/I0]) if I0 > 0 else np.zeros(3)

    return Result(p, n, h_eff, l2, h1, centroid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=float, default=1e-7)
    parser.add_argument("--R", type=float, default=40e-9)
    parser.add_argument("--q", type=float, default=1.602e-19)
    parser.add_argument("--epsr", type=float, default=11.7)
    parser.add_argument("--center", type=str, default="0,0,0")
    parser.add_argument("--hs", required=True)
    parser.add_argument("--degs", required=True)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    center = np.array(parse_list_floats(args.center))
    hs = parse_list_floats(args.hs)
    ps = parse_list_ints(args.degs)

    if comm.rank == 0:
        print("=== Running hp study ===")

    for p in ps:
        for h in hs:
            r = solve_one(comm, args.L, h, p, args.q, args.R, args.epsr, center)
            if comm.rank == 0:
                print(
                    f"p={r.p}  n={r.n:4d}  h={r.h:.3e}  "
                    f"L2={r.l2:.3e}  H1={r.h1:.3e}  "
                    f"err_center=({r.centroid[0]:+.2e},"
                    f"{r.centroid[1]:+.2e},"
                    f"{r.centroid[2]:+.2e})"
                )


if __name__ == "__main__":
    main()
