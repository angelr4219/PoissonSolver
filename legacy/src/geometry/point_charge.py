#!/usr/bin/env python3
from __future__ import annotations
import numpy as np, ufl
from typing import Iterable, Tuple
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem import petsc as fem_petsc

def gaussian_rho(domain, Q: float, pos: Tuple[float, ...], sigma: float):
    """
    UFL expression for a normalized Gaussian with total charge Q.
    2D if len(pos)==2; 3D if len(pos)==3. Keep sigma >= ~2h for stability.
    """
    x = ufl.SpatialCoordinate(domain)
    if len(pos) == 2:
        qx, qy = pos
        term = (x[0]-qx)**2 + (x[1]-qy)**2
        norm = Q / (2*np.pi*sigma**2) if Q != 0.0 else 0.0
    else:
        qx, qy, qz = pos
        term = (x[0]-qx)**2 + (x[1]-qy)**2 + (x[2]-qz)**2
        norm = Q / ((np.sqrt(2*np.pi)*sigma)**3) if Q != 0.0 else 0.0
    return norm * ufl.exp(-term/(2*sigma**2))

def _solve_petsc(a_form: fem.Form, L_form: fem.Form, V: fem.FunctionSpace, bcs):
    """Assemble A,b; apply BCs; solve with PETSc KSP; return fem.Function in V."""
    # Assemble
    A = fem_petsc.assemble_matrix(a_form, bcs=bcs); A.assemble()
    b = fem_petsc.assemble_vector(L_form)
    fem_petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    # KSP (CG+Hypre if available)
    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    try:
        pc.setType("hypre")
    except Exception:
        pc.setType("gamg")
    ksp.setTolerances(rtol=1e-12, atol=1e-16, max_it=1000)

    uh = fem.Function(V)
    x = A.createVecRight()
    ksp.solve(b, x)
    uh.x.array[:] = x.getArray()
    uh.x.scatter_forward()
    return uh, ksp

def _project_vector(domain, vexpr):
    """Project a vector expression to CG1 via manual assembly (portable across dolfinx versions)."""
    Vv = fem.functionspace(domain, ("Lagrange", 1, (domain.topology.dim,)))
    w = ufl.TrialFunction(Vv); v = ufl.TestFunction(Vv)
    aP = ufl.inner(w, v) * ufl.dx
    LP = ufl.inner(vexpr, v) * ufl.dx
    a_form = fem.form(aP); L_form = fem.form(LP)
    Eh, _ = _solve_petsc(a_form, L_form, Vv, bcs=[])
    Eh.name = "E"
    return Eh

def solve_with_point_charge(domain, facet_tags, gate_tags_V: Iterable[Tuple[int, float]],
                            eps_dg, rho_expr, outfile: str, write_eps_viz: bool = True):
    """
    Solve -div(eps grad phi) = rho with Dirichlet on gate facet tags, Neumann elsewhere.
    Writes XDMF (phi, eps, rho, E).
    """
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = ufl.inner(eps_dg * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho_expr * v * ufl.dx

    # Dirichlet BCs on gate surfaces
    bcs = []
    for tag, Vgate in gate_tags_V:
        fids = facet_tags.find(tag)
        if fids.size:
            dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, fids)
            gfun = fem.Function(V); gfun.x.array[:] = Vgate
            bcs.append(fem.dirichletbc(gfun, dofs))

    # Assemble & solve
    a_form = fem.form(a); L_form = fem.form(L)
    uh, ksp = _solve_petsc(a_form, L_form, V, bcs)
    uh.name = "phi"

    # Vector field E = -grad(phi), projected for viz
    Eh = _project_vector(domain, -ufl.grad(uh))

    # For viz: rho and eps (CG1 copies)
    Vr = fem.functionspace(domain, ("Lagrange", 1))
    rhoh = fem.Function(Vr); rhoh.name = "rho"
    rhoh.interpolate(fem.Expression(rho_expr, Vr.element.interpolation_points()))

    with io.XDMFFile(domain.comm, f"{outfile}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        if facet_tags is not None:
            xf.write_meshtags(facet_tags, domain.geometry)
        xf.write_function(uh)
        xf.write_function(rhoh)
        if write_eps_viz:
            from materials import eps_to_cg1_for_viz
            eps_cg = eps_to_cg1_for_viz(domain, eps_dg)
            xf.write_function(eps_cg)
        xf.write_function(Eh)

    if domain.comm.rank == 0:
        print(f"[out] {outfile}.xdmf  (iters={ksp.getIterationNumber()}, rnorm={ksp.getResidualNorm():.3e})")
    return uh
