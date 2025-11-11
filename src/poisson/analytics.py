from __future__ import annotations
import numpy as np

def image_charge_coeffs(eps1, eps2):
    """
    For point charge q at (0,0,d) in z>0, planar interface z=0.
    Region z>0: phi1 = (1/(4π ε1)) ( q/R1 + q' / R2 )
    Region z<0: phi2 = (1/(4π ε2)) ( q'' / R1 )
    where q'  = (eps1 - eps2)/(eps1 + eps2) * q
          q'' = (2 eps2)/(eps1 + eps2) * q
    """
    def qp(q):  return (eps1 - eps2)/(eps1 + eps2) * q
    def qpp(q): return (2.0*eps2)/(eps1 + eps2) * q
    return qp, qpp

def phi_image_3d(points, q, d, eps1, eps2):
    """
    Evaluate analytic image-charge potential at array of points (N,3).
    points[:,2]>0 -> use phi1, else phi2.
    """
    qp, qpp = image_charge_coeffs(eps1, eps2)
    x = points[:,0]; y = points[:,1]; z = points[:,2]
    R1 = np.sqrt(x**2 + y**2 + (z - d)**2)
    R2 = np.sqrt(x**2 + y**2 + (z + d)**2)
    phi = np.empty_like(R1)

    # z>0
    idx1 = z > 0.0
    phi[idx1] = (1.0/(4.0*np.pi*eps1)) * ( q / R1[idx1] + qp(q) / R2[idx1] )

    # z<0
    idx2 = ~idx1
    # Note: R1 in phi2 uses distance to +d source, even below the plane.
    phi[idx2] = (1.0/(4.0*np.pi*eps2)) * ( qpp(q) / R1[idx2] )
    return phi
