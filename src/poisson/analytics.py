from __future__ import annotations
import numpy as np

EPS0: float = 8.8541878128e-12   # F/m


def phi_conducting_sphere(r_m: np.ndarray, V0: float, R_m: float) -> np.ndarray:
    """Analytic potential for an isolated conducting sphere of radius R at V₀.

    Valid in free space (ε = ε₀) for r ≥ R; returns V₀ for r < R.
    φ(r) = V₀ · R / r  (solution of Laplace's equation in infinite medium).
    """
    r = np.asarray(r_m, dtype=float)
    phi = np.where(r > 0, V0 * R_m / np.maximum(r, R_m), V0)
    return phi


def phi_point_charge(r_m: np.ndarray, q: float, eps_r: float = 1.0) -> np.ndarray:
    """Coulomb potential of a point charge q in a homogeneous medium.

    φ(r) = q / (4πε₀εᵣ r)
    """
    r = np.asarray(r_m, dtype=float)
    return q / (4.0 * np.pi * EPS0 * eps_r * np.maximum(r, 1e-30))


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
