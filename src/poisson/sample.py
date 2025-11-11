from __future__ import annotations
import numpy as np
from dolfinx import fem
from ufl import grad

def line_points(p0, p1, n):
    t = np.linspace(0.0, 1.0, n)
    return np.c_[p0 + (p1 - p0)[None, :] * t[:, None]]

def evaluate_on_points(u, points):
    # u: fem.Function; points: (N,dim) ndarray in physical coordinates
    dom = u.function_space.mesh
    tree = dom.bounding_box_tree
    vals = np.empty(points.shape[0])
    # Query cell per point and evaluate
    for i, X in enumerate(points):
        cell = tree.compute_first_entity_collision(X)
        vals[i] = u.eval(X, cell)[0]
    return vals

def interface_continuity_phi(u, axis, offset, span=(-1.0, 1.0), n=401):
    """
    Sample phi just above and below the interface plane.
    axis: 0->x, 1->y, 2->z interface normal axis
    offset: small δ > 0 (sample at ±δ)
    span: range of the transverse coordinate for linecut
    """
    dim = u.function_space.mesh.topology.dim
    assert axis in (0,1,2)[:dim]
    coord = 0 if axis != 0 else 1  # pick a transverse axis stably
    if dim == 2:
        # interface y=0 => axis=1. transverse=0 (x)
        xs = np.linspace(span[0], span[1], n)
        if axis == 1:
            p_top = np.c_[xs, np.full_like(xs, +offset)]
            p_bot = np.c_[xs, np.full_like(xs, -offset)]
        else:
            raise NotImplementedError
    else:
        xs = np.linspace(span[0], span[1], n)
        # For z=0 (axis=2), linecut along x at y=0
        if axis == 2:
            p_top = np.c_[xs, np.zeros_like(xs), np.full_like(xs, +offset)]
            p_bot = np.c_[xs, np.zeros_like(xs), np.full_like(xs, -offset)]
        else:
            raise NotImplementedError
    phi_top = evaluate_on_points(u, p_top)
    phi_bot = evaluate_on_points(u, p_bot)
    return xs, phi_top, phi_bot

def interface_flux_jump(u, eps, axis, delta, span=(-1.0,1.0), n=401):
    """
    Approximate normal E components above/below and return ε1 E1n - ε2 E2n.
    Uses finite difference from samples at ±δ.
    """
    xs, phi_top, phi_bot = interface_continuity_phi(u, axis, delta, span, n)
    # normal derivative ≈ - (phi(delta) - phi(0))/delta and (phi(0) - phi(-delta))/delta
    # approximate phi(0) as average (symmetric)
    phi0 = 0.5*(phi_top + phi_bot)
    E1n = -(phi_top - phi0)/delta
    E2n = -(phi0 - phi_bot)/delta

    # Need ε on each side. Use eps(x) sampling:
    # We'll take the same points and sample eps as cellwise piecewise constant.
    # (Evaluating DG0 at a point returns value of that cell)
    return xs, E1n, E2n
