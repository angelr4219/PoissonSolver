from __future__ import annotations
import numpy as np
import ufl

def gaussian_rho(domain, x0, q, sigma):
    """2D/3D isotropic Gaussian with total charge q."""
    x = ufl.SpatialCoordinate(domain)
    dim = domain.topology.dim
    dx2 = sum((x[i]-x0[i])**2 for i in range(dim))
    if dim == 2:
        norm = q / (2.0*np.pi*sigma**2)
    elif dim == 3:
        norm = q / ((2.0*np.pi)**1.5 * sigma**3)
    else:
        raise ValueError("Only 2D/3D supported")
    return norm * ufl.exp(-dx2/(2.0*sigma**2))
