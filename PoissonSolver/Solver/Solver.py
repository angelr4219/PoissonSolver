"""
High‑level solver routines combining geometry, materials and charges.

This module offers convenience functions that orchestrate the workflow
of building a geometry, assigning material properties, defining
charges, setting boundary conditions and solving Poisson’s equation.
It uses the lower‑level utilities defined in ``geometry.py`` and
``physics.py`` to assemble complete simulations for the examples
present in the repository.

The functions here return both the function space and the solution,
allowing callers to postprocess or save the solution as needed.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from mpi4py import MPI
from dolfinx import fem, io

from geometry import (RodBoreParams, DiskParams, GateBoxParams,
                      build_rod_with_bore, build_disk_with_hole,
                      build_gate_box)
from physics import (assign_permittivity, gaussian_charge,
                     constant_charge, dirichlet_bc_on_tag,
                     gate_bcs, solve_poisson)


def solve_rod(params: RodBoreParams,
              eps_map: Dict[int, float],
              total_charge: float,
              charge_center: Sequence[float],
              charge_sigma: float,
              outer_potential: float = 0.0,
              degree: int = 1,
              solver_options: Optional[Dict[str, str]] = None
              ) -> Tuple[fem.FunctionSpace, fem.Function]:
    """Solve the Poisson problem on a rod with a cylindrical bore.

    Parameters
    ----------
    params : RodBoreParams
        Geometry parameters for the rod.
    eps_map : dict
        Mapping from cell tag to relative permittivity εr.
        Tags 101, 201, 202, 203 correspond to air and the three
        solid layers, respectively.
    total_charge : float
        Total charge of the Gaussian (Coulombs).
    charge_center : sequence of floats
        Coordinates of the centre of the Gaussian charge distribution.
    charge_sigma : float
        Standard deviation of the Gaussian.
    outer_potential : float, optional
        Dirichlet potential on the outer boundary facets.  Default is 0 V.
    degree : int, optional
        Polynomial degree of the finite‑element space.  Default is 1.
    solver_options : dict, optional
        PETSc solver options.

    Returns
    -------
    (V, phi) : tuple
        The function space and computed potential.
    """
    domain, cell_tags, facet_tags = build_rod_with_bore(params, comm=MPI.COMM_WORLD)
    eps = assign_permittivity(domain, cell_tags, eps_map)
    rho = gaussian_charge(domain, total_charge, charge_center, charge_sigma)
    # Create a function space temporarily to construct BCs
    V_tmp = fem.FunctionSpace(domain, ("CG", degree))
    bc = dirichlet_bc_on_tag(V_tmp, facet_tags, tag=301, value=outer_potential)
    # Solve
    V, phi = solve_poisson(domain, eps, rho, [bc], degree=degree,
                           solver_options=solver_options)
    return V, phi


def solve_disk(params: DiskParams,
               outer_potential: float,
               hole_potential: float,
               degree: int = 1,
               solver_options: Optional[Dict[str, str]] = None
               ) -> Tuple[fem.FunctionSpace, fem.Function]:
    """Solve a 2‑D Laplace problem on a rectangle with a circular hole.

    All cells are assigned a uniform permittivity of εr = 1 (vacuum).
    Dirichlet conditions are applied on the outer boundary (tag 1)
    and on the hole boundary (tag 2).

    Parameters
    ----------
    params : DiskParams
        Geometry parameters for the disk domain.
    outer_potential : float
        Potential on the outer boundary.
    hole_potential : float
        Potential on the hole boundary.
    degree : int, optional
        Polynomial degree of the finite‑element space.  Default is 1.
    solver_options : dict, optional
        PETSc solver options.

    Returns
    -------
    (V, phi) : tuple
        The function space and computed potential.
    """
    domain, cell_tags, facet_tags = build_disk_with_hole(params, comm=MPI.COMM_WORLD)
    # Assign εr = 1 everywhere
    eps = assign_permittivity(domain, cell_tags, {1: 1.0})
    rho = constant_charge(domain, 0.0)
    # Build BCs
    V_tmp = fem.FunctionSpace(domain, ("CG", degree))
    bc_outer = dirichlet_bc_on_tag(V_tmp, facet_tags, tag=1, value=outer_potential)
    bc_hole  = dirichlet_bc_on_tag(V_tmp, facet_tags, tag=2, value=hole_potential)
    V, phi = solve_poisson(domain, eps, rho, [bc_outer, bc_hole], degree=degree,
                           solver_options=solver_options)
    return V, phi


def solve_gate(params: GateBoxParams,
               a: float,
               xs_gates: Sequence[float],
               Vs_gates: Sequence[float],
               eps_r: float = 11.7,
               degree: int = 1,
               solver_options: Optional[Dict[str, str]] = None
               ) -> Tuple[fem.FunctionSpace, fem.Function]:
    """Solve a Laplace problem for a gate array on a box domain.

    The domain is a box extending from z=0 down to z=-H (or z=H if using
    the gmsh definition).  A uniform permittivity of εr is assumed.
    Voltages are applied on square gate regions on the top surface;
    remaining top DOFs are grounded and all other boundaries are left
    natural (Neumann) so that the normal component of ε∇u vanishes.

    Parameters
    ----------
    params : GateBoxParams
        Parameters describing the box geometry.  If ``use_gmsh`` is
        True, gmsh is used to generate and tag the top surface.
    a : float
        Half‑width of each gate square.
    xs_gates : sequence of floats
        x‑coordinates of the gate centres.  The gates are centred at
        (xi, 0) on the top surface.
    Vs_gates : sequence of floats
        Voltages to apply on each gate.  Must have the same length as
        ``xs_gates``.
    eps_r : float, optional
        Relative permittivity of the entire domain.  Default is 11.7
        (silicon).
    degree : int, optional
        Polynomial degree of the finite‑element space.  Default is 1.
    solver_options : dict, optional
        PETSc solver options.

    Returns
    -------
    (V, phi) : tuple
        The function space and computed potential.
    """
    domain, cell_tags, facet_tags = build_gate_box(params, comm=MPI.COMM_WORLD)
    # Assign uniform permittivity
    # In this simple example, we assume a single tag on all cells (0)
    eps = assign_permittivity(domain, cell_tags, {0: eps_r})
    # No free charge
    rho = constant_charge(domain, 0.0)
    # Build a temporary function space to locate gate DOFs
    V_tmp = fem.FunctionSpace(domain, ("CG", degree))
    bcs = gate_bcs(V_tmp, a, xs_gates, Vs_gates)
    V, phi = solve_poisson(domain, eps, rho, bcs, degree=degree,
                           solver_options=solver_options)
    return V, phi


def write_solution(filename_prefix: str,
                   domain: fem.mesh.Mesh,
                   phi: fem.Function,
                   extra_functions: Optional[Sequence[fem.Function]] = None
                   ) -> None:
    """Write a mesh and solution to XDMF format.

    Parameters
    ----------
    filename_prefix : str
        Prefix for the output file.  ``{prefix}.xdmf`` will be created.
    domain : dolfinx.mesh.Mesh
        Mesh to write.
    phi : dolfinx.fem.Function
        Solution function to write.
    extra_functions : sequence of dolfinx.fem.Function, optional
        Additional functions to write (e.g., permittivity).  They should
        live on the same mesh.
    """
    comm = domain.comm
    domain.name = "mesh"
    with io.XDMFFile(comm, f"{filename_prefix}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi)
        if extra_functions:
            for f in extra_functions:
                xdmf.write_function(f)