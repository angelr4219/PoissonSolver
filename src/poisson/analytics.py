from __future__ import annotations
import numpy as np

EPS0 = 8.8541878128e-12  # F/m


def phi_point_charge(points: np.ndarray, Q: float, eps: float = EPS0) -> np.ndarray:
    """Coulomb potential [V] of a point charge Q at the origin."""
    r = np.linalg.norm(points, axis=1)
    return Q / (4.0 * np.pi * eps * r)


def phi_conducting_sphere(
    points: np.ndarray, Q: float, R: float, eps: float = EPS0
) -> np.ndarray:
    """
    Analytic potential [V] of a grounded conducting sphere of radius R
    carrying total charge Q (constant inside, Coulomb outside).
    """
    r = np.linalg.norm(points, axis=1)
    phi = np.where(r >= R,
                   Q / (4.0 * np.pi * eps * r),
                   Q / (4.0 * np.pi * eps * R))
    return phi


def image_charge_coeffs(eps1, eps2):
    def qp(q):  return (eps1 - eps2)/(eps1 + eps2) * q
    def qpp(q): return (2.0*eps2)/(eps1 + eps2) * q
    return qp, qpp

def phi_image_3d(points, q, d, eps1, eps2):
    qp, qpp = image_charge_coeffs(eps1, eps2)
    x = points[:,0]; y = points[:,1]; z = points[:,2]
    R1 = np.sqrt(x**2 + y**2 + (z - d)**2)
    R2 = np.sqrt(x**2 + y**2 + (z + d)**2)
    phi = np.empty_like(R1)
    idx1 = z > 0.0
    phi[idx1] = (1.0/(4.0*np.pi*eps1)) * ( q / R1[idx1] + qp(q) / R2[idx1] )
    idx2 = ~idx1
    phi[idx2] = (1.0/(4.0*np.pi*eps2)) * ( qpp(q) / R1[idx2] )
    return phi
