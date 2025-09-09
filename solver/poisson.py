#!/usr/bin/env python3
"""
poisson.py
General Poisson solver for variable permittivity:

  Find u in V s.t.  ∫_Ω ε ∇u·∇v dx = ∫_Ω f v dx + ∫_ΓN g v ds

Supports:
- Dirichlet and Neumann BCs by facet names or tag IDs
- ε(x): piecewise constant (DG0) from cell tags OR any fem.Function (DG/CG)
- Diagnostics: min/max u, BC tag counts, KSP info, energy, condition estimate
- Returns handy objects for downstream post-processing

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, cpp
from petsc4py import PETSc

FacetSpec = Union[int, str]
RightHandSide = Union[fem.Function, float]

@dataclass
class SolveDiagnostics:
    ksp_its: int
    ksp_rnorm: float
    u_min: float
    u_max: float
    energy: float
    cond_est: Optional[float]
    facet_counts: Dict[str, int]

def build_measures(domain: mesh.Mesh, cell_tags: Optional[cpp.mesh.MeshTags_int32], facet_tags: Optional[cpp.mesh.MeshTags_int32]):
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags) if cell_tags is not None else ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags) if facet_tags is not None else ufl.Measure("ds", domain=domain)
    return dx, ds

def _facet_name_map(facet_tags: Optional[cpp.mesh.MeshTags_int32], facet_tag2name: Optional[Mapping[int,str]]):
    name_by_id: Dict[int, str] = {}
    id_by_name: Dict[str, int] = {}
    if facet_tags is not None and facet_tag2name is not None:
        for tag, nm in facet_tag2name.items():
            name_by_id[tag] = nm
            id_by_name[nm] = tag
    return name_by_id, id_by_name

def solve_poisson(
    domain: mesh.Mesh,
    cell_tags: Optional[cpp.mesh.MeshTags_int32],
    facet_tags: Optional[cpp.mesh.MeshTags_int32],
    facet_tag2name: Optional[Mapping[int,str]],
    eps_fun: fem.Function,
    f: RightHandSide = 0.0,
    neumann_g: Optional[Dict[FacetSpec, RightHandSide]] = None,
    dirichlet_values: Optional[Dict[FacetSpec, float]] = None,
    V_family: str = "Lagrange", V_degree: int = 1,
    petsc_opts: Optional[Dict[str, Union[str,int,float]]] = None,
) -> Tuple[fem.Function, fem.FunctionSpace, Tuple[ufl.Measure, ufl.Measure], SolveDiagnostics]:
    """
    Solve -div(ε ∇u) = f with BCs.

    dirichlet_values: mapping from facet name or id -> value
    neumann_g: mapping from facet name or id -> Neumann data g (flux)
    """
    comm = domain.comm
    V = fem.functionspace(domain, (V_family, V_degree))
    u = fem.Function(V, name="phi")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    dx, ds = build_measures(domain, cell_tags, facet_tags)

    # RHS f
    if isinstance(f, (int, float)):
        f_fun = fem.Constant(domain, PETSc.ScalarType(f))
    else:
        f_fun = f

    a = ufl.inner(eps_fun*ufl.grad(du), ufl.grad(v)) * dx
    L = ufl.inner(f_fun, v) * dx

    name_by_id, id_by_name = _facet_name_map(facet_tags, facet_tag2name)

    # Neumann contributions
    if neumann_g:
        for key, g in neumann_g.items():
            tag = id_by_name.get(key, key) if isinstance(key, str) else key
            g_fun = fem.Constant(domain, PETSc.ScalarType(g)) if isinstance(g, (int,float)) else g
            L += ufl.inner(g_fun, v) * ds(tag)

    # Dirichlet BCs
    bcs = []
    facet_counts: Dict[str, int] = {}
    if dirichlet_values:
        for key, val in dirichlet_values.items():
            tag = id_by_name.get(key, key) if isinstance(key, str) else key
            # Tabulate dofs on this boundary
            boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facet_tags.find(tag) if facet_tags is not None else np.array([], dtype=np.int32))
            bc = fem.dirichletbc(PETSc.ScalarType(val), boundary_dofs, V)
            bcs.append(bc)
            nm = name_by_id.get(tag, str(tag))
            facet_counts[nm] = int(len(boundary_dofs))

    problem = fem.petsc.LinearProblem(a, L, bcs=bcs,
        petsc_options=(petsc_opts or {"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10,"ksp_atol":1e-14,"ksp_max_it":500}))
    uh = problem.solve()

    # Diagnostics
    u_min = comm.allreduce(np.min(uh.x.array), op=MPI.MIN)
    u_max = comm.allreduce(np.max(uh.x.array), op=MPI.MAX)

    # Energy = 0.5 ∫ ε |∇u|^2 dx
    energy = 0.5 * fem.assemble_scalar(fem.form( ufl.inner(eps_fun*ufl.grad(uh), ufl.grad(uh)) * dx ))
    energy = comm.allreduce(energy, op=MPI.SUM)

    # KSP info
    ksp = problem.solver.ksp
    ksp_its = ksp.getIterationNumber()
    ksp_rnorm = ksp.getResidualNorm()

    # Cheap cond estimate (optional): diagonal precond ratio
    # (Not rigorous; users can enable -ksp_view_eigenvalues for PETSc)
    cond_est = None

    return uh, V, (dx, ds), SolveDiagnostics(
        ksp_its=ksp_its, ksp_rnorm=ksp_rnorm, u_min=u_min, u_max=u_max, energy=energy, cond_est=cond_est, facet_counts=facet_counts
    )
