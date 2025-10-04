from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc


# ----------------- helpers -----------------
def _coerce_tag_to_int(tag, tags_obj) -> int:
    """Resolve a str/int tag to an int ID in MeshTags."""
    if isinstance(tag, (int, np.integer)):
        return int(tag)
    name_map = {
        # 2D facet defaults from our builder
        "outer_boundary": 101,
        "inclusion_boundary": 102,
        # 3D facet defaults from our builder
        "bore_boundary": 202,
        # cell names that sometimes leak through
        "outer": 10, "inclusion": 20, "solid": 110, "bore": 120,
    }
    if isinstance(tag, str):
        key = tag.strip().lower()
        if key in name_map:
            return int(name_map[key])
    try:
        vals = np.unique(tags_obj.values)
        if vals.size == 1:
            return int(vals[0])
        if vals.size > 1:
            return int(sorted(vals)[0])
    except Exception:
        pass
    raise TypeError(f"Cannot resolve tag {tag!r} to an integer id.")


def _build_measures(domain: mesh.Mesh, cell_tags=None, facet_tags=None):
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags) if cell_tags is not None else ufl.dx(domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags) if facet_tags is not None else ufl.ds(domain=domain)
    return dx, ds


def _dirichlet_bc(V, value, facet_tags, tag) -> fem.DirichletBC:
    """Create a Dirichlet BC on 'tag' (str or int)."""
    tag_id = _coerce_tag_to_int(tag, facet_tags)
    dofs = fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, facet_tags.find(tag_id))
    if isinstance(value, (float, int, np.floating)):
        g = fem.Function(V)
        g.x.array[:] = float(value)
        return fem.dirichletbc(g, dofs)
    elif isinstance(value, fem.Function):
        return fem.dirichletbc(value, dofs)
    else:
        raise TypeError("Dirichlet value must be a float or fem.Function")


# ----------------- main -----------------
def solve_poisson(
    domain,                      # dolfinx.mesh.Mesh
    cell_tags,                   # dolfinx.mesh.meshtags or None
    facet_tags,                  # dolfinx.mesh.meshtags or None
    eps_fun,                     # fem.Function OR dict[tag->epsr/name]
    f=None,                      # volumetric source (UFL/fem.Function) - CLI passes this
    dirichlet_values=None,       # dict[tag->value] (CLI name)
    petsc_opts=None,             # PETSc options dict
    **kwargs,                    # tolerate extra kwargs from older callers (rhs_f, dirichlet, neumann, etc.)
):
    """
    -div(ε ∇u) = f  in Ω
    u = g on Γ_D (dirichlet_values)
    (ε ∇u)·n = t on Γ_N (optional: 'neumann' or 'neumann_values' via kwargs)

    Returns: uh, V, (dx, ds), diag
    """
    # Accept aliases / legacy keys from callers
    rhs_f = kwargs.get("rhs_f", None)
    if f is None:
        f = rhs_f
    # If caller provided 'dirichlet' under a different key, prefer explicit dirichlet_values
    if dirichlet_values is None:
        dirichlet_values = kwargs.get("dirichlet", None)
    # Optional Neumann map
    neumann = kwargs.get("neumann_values", kwargs.get("neumann", None))

    # AUTO_EPS_FROM_DICT: accept dict tag->epsr or tag->name and build DG0 epsilon
    if not isinstance(eps_fun, fem.Function):
        from src.physics.permittivity import eps_from_materials as _eps_from_materials
        eps_fun = _eps_from_materials(domain, cell_tags, eps_fun)

    # FE space
    V = fem.functionspace(domain, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dx, ds = _build_measures(domain, cell_tags, facet_tags)

    a = ufl.inner(eps_fun * ufl.grad(u), ufl.grad(v)) * dx

    # prefer f if provided; else zero
    if f is None:
        L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * dx
    else:
        L = f * v * dx

    # Neumann BCs (natural)
    if neumann:
        for tag, flux in neumann.items():
            tag_id = _coerce_tag_to_int(tag, facet_tags)
            if isinstance(flux, (float, int, np.floating)):
                t_val = fem.Constant(domain, PETSc.ScalarType(float(flux)))
                L += t_val * v * ds(tag_id)
            elif isinstance(flux, fem.Function) or isinstance(flux, ufl.core.expr.Expr):
                L += flux * v * ds(tag_id)
            else:
                raise TypeError("Neumann value must be float, fem.Function, or UFL expression")

    # Dirichlet BCs
    bcs = []
    if dirichlet_values:
        for tag, val in dirichlet_values.items():
            bcs.append(_dirichlet_bc(V, val, facet_tags, tag))

    uh = fem.Function(V, name="phi")

    # Wrap forms for PETSc to avoid rank issues in nightly
    a_form = fem.form(a)
    L_form = fem.form(L)

    problem = LinearProblem(
        a_form, L_form,
        bcs=bcs,
        u=uh,
        petsc_options_prefix="poisson_",
        petsc_options=petsc_opts or {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 500
        }
    )
    uh = problem.solve()

    diag = {}
    try:
        ksp = problem.solver.ksp  # type: ignore[attr-defined]
        diag["ksp_its"] = ksp.getIterationNumber()
        diag["ksp_rnorm"] = ksp.getResidualNorm()
    except Exception:
        pass
    try:
        energy = fem.assemble_scalar(fem.form(ufl.inner(eps_fun * ufl.grad(uh), ufl.grad(uh)) * dx))
        diag["energy"] = energy
    except Exception:
        pass

    return uh, V, (dx, ds), diag
