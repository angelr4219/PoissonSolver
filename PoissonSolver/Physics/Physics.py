"""
Physics utilities and solver routines for Poisson problems.

This module provides helper functions to assign material properties,
define charge distributions and assemble and solve Poisson's equation
on FEniCSx meshes.  Boundary conditions can be specified via
``dirichlet_bc_on_tag`` or ``gate_bcs`` for gate benchmarks.  A
generic solver routine, ``solve_poisson``, takes a permittivity
function, a charge density expression and a list of Dirichlet
conditions and returns the computed potential.

The intention is to separate physical definitions (materials and
charges) from geometry (see ``geometry.py``) and from the command‑line
interface (see ``main.py``).  These utilities are flexible enough to
be reused across different problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem


def assign_permittivity(domain: fem.mesh.Mesh,
                        cell_tags: fem.MeshTags,
                        tag_to_epsr: Dict[int, float],
                        eps0: float = 8.854187817e-12
                        ) -> fem.Function:
    """Assign permittivity values to cells according to tags.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh on which to define the permittivity.
    cell_tags : dolfinx.mesh.MeshTags
        Tags associated with each cell of the mesh.
    tag_to_epsr : dict
        Mapping from integer tag to relative permittivity εr.
    eps0 : float, optional
        Vacuum permittivity in SI units.  Default is 8.854187817e-12 F/m.

    Returns
    -------
    eps : dolfinx.fem.Function
        A DG0 function representing the permittivity ε = ε0 εr on each cell.
    """
    DG0 = fem.FunctionSpace(domain, ("DG", 0))
    epsr = fem.Function(DG0)
    cell_values = epsr.x.array
    # Assign the relative permittivity based on cell_tags
    for tag, epsr_val in tag_to_epsr.items():
        mask = cell_tags.values == tag
        cell_values[mask] = epsr_val
    epsr.x.array[:] = cell_values
    # Multiply by vacuum permittivity to obtain absolute permittivity
    eps = fem.Function(DG0)
    eps.x.array[:] = eps0 * epsr.x.array
    return eps


def gaussian_charge(domain: fem.mesh.Mesh,
                    q: float,
                    center: Sequence[float],
                    sigma: float
                    ) -> ufl.Form:
    """Return a UFL expression for a normalized 3‑D Gaussian charge density.

    The returned expression can be used directly in a weak form.  The
    normalization is such that the total integrated charge equals ``q``.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh on which the charge density is defined.
    q : float
        Total charge (Coulombs).
    center : tuple of floats
        Coordinates (x0, y0, z0) of the Gaussian centre.
    sigma : float
        Standard deviation of the Gaussian.
    """
    x0, y0, z0 = center
    x = ufl.SpatialCoordinate(domain)
    dx_vec = x - ufl.as_vector((x0, y0, z0))
    r2 = ufl.inner(dx_vec, dx_vec)
    two_pi = 2.0 * np.pi
    norm = q * (two_pi * sigma ** 2) ** (-1.5)
    return norm * ufl.exp(-r2 / (2.0 * sigma ** 2))


def constant_charge(domain: fem.mesh.Mesh,
                    rho0: float
                    ) -> ufl.Form:
    """Return a constant charge density expression.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh on which the charge density is defined.
    rho0 : float
        Constant charge density (C/m³).
    """
    return ufl.Constant(domain, PETSc.ScalarType(rho0))


def dirichlet_bc_on_tag(V: fem.FunctionSpace,
                        facet_tags: fem.MeshTags,
                        tag: int,
                        value: float
                        ) -> fem.DirichletBC:
    """Create a DirichletBC that applies a constant value on facets with a given tag.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Finite‑element function space.
    facet_tags : dolfinx.mesh.MeshTags
        Tags on facets (dimension = mesh.topology.dim - 1).
    tag : int
        The tag value identifying the facets to constrain.
    value : float
        The constant potential value to set on those facets.
    """
    facets = facet_tags.find(tag)
    dofs = fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, facets)
    return fem.dirichletbc(PETSc.ScalarType(value), dofs, V)


def gate_bcs(V: fem.FunctionSpace,
             a: float,
             xs: Sequence[float],
             Vs: Sequence[float],
             tol: float = 1e-10
             ) -> List[fem.DirichletBC]:
    """Create Dirichlet boundary conditions on a planar gate array.

    Given a list of gate centre positions ``xs`` and corresponding voltages
    ``Vs`` (all in SI units), this helper finds degrees of freedom on the
    top surface (z ≈ 0) whose x‑ and y‑coordinates lie within each gate
    square of half‑width ``a``.  Those DOFs receive the specified voltages.
    DOFs on the top surface that do not lie in any gate receive zero
    potential.  All other facets remain unconstrained.
    """
    def on_top(x):
        return np.isclose(x[2], 0.0, atol=tol)

    dofs_top = fem.locate_dofs_geometrical(V, on_top)
    x_top = V.tabulate_dof_coordinates()[dofs_top].reshape((-1, V.mesh.geometry.dim))
    used = np.zeros(len(dofs_top), dtype=bool)
    bcs: List[fem.DirichletBC] = []
    # Assign gate voltages
    for xi, Vi in zip(xs, Vs):
        in_rect = (
            (x_top[:, 0] >= (xi - a) - tol) & (x_top[:, 0] <= (xi + a) + tol) &
            (x_top[:, 1] >= -a - tol)      & (x_top[:, 1] <=  a + tol)
        )
        idx = np.where(in_rect)[0]
        if idx.size > 0:
            dofs_gate = dofs_top[idx]
            bc_fun = fem.Function(V)
            bc_fun.x.array[:] = 0.0
            bc_fun.x.array[dofs_gate] = Vi
            bcs.append(fem.dirichletbc(bc_fun, dofs_gate))
            used[idx] = True
    # Ground the remaining top DOFs
    idx0 = np.where(~used)[0]
    if idx0.size > 0:
        dofs0 = dofs_top[idx0]
        bc_fun0 = fem.Function(V)
        bc_fun0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(bc_fun0, dofs0))
    return bcs


def solve_poisson(domain: fem.mesh.Mesh,
                  eps: fem.Function,
                  rho: ufl.Form,
                  bcs: Sequence[fem.DirichletBC],
                  degree: int = 1,
                  solver_options: Dict[str, str] | None = None
                  ) -> Tuple[fem.FunctionSpace, fem.Function]:
    """Solve the Poisson equation ``-div(ε ∇u) = ρ`` with Dirichlet BCs.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh on which to solve the equation.
    eps : dolfinx.fem.Function
        Absolute permittivity field ε defined on DG0.
    rho : UFL expression
        Charge density expressed as a UFL expression.  For Laplace’s
        equation (ρ = 0), pass ``constant_charge(domain, 0.0)``.
    bcs : sequence of fem.DirichletBC
        Dirichlet boundary conditions to apply.
    degree : int, optional
        Polynomial degree of the continuous Galerkin space.  Default is 1.
    solver_options : dict, optional
        PETSc solver options.  If None, sensible defaults are used.

    Returns
    -------
    (V, phi) : tuple
        ``V`` is the function space and ``phi`` is the finite‑element
        solution.
    """
    V = fem.FunctionSpace(domain, ("CG", degree))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(eps * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho * v * ufl.dx
    # Default PETSc options if none provided
    opts = solver_options or {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
    }
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=opts)
    phi = problem.solve()
    phi.name = "phi"
    return V, phi