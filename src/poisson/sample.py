from __future__ import annotations
import numpy as np

def line_points(p0, p1, n):
    t = np.linspace(0.0, 1.0, n)
    return np.c_[p0 + (p1 - p0)[None, :] * t[:, None]]

def evaluate_on_points(u, points):
    dom = u.function_space.mesh
    tree = dom.bounding_box_tree
    vals = np.empty(points.shape[0])
    for i, X in enumerate(points):
        cell = tree.compute_first_entity_collision(X)
        vals[i] = u.eval(X, cell)[0]
    return vals
