import numpy as np

def g_uvz(u: np.ndarray | float, v: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
    """
    g(u,v,z) = (1/(2π)) * atan2(u*v, z*sqrt(u^2 + v^2 + z^2))
    Vectorized over numpy arrays.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(z, dtype=float)
    denom = z * np.sqrt(u*u + v*v + z*z)
    return (1.0 / (2.0 * np.pi)) * np.arctan2(u * v, denom)

def phi0_rect(x, y, z, a: float, xs, Vs):
    """
    Analytic potential φ0 for a sum of identical square gates of half-size 'a'.

    Parameters
    ----------
    x, y, z : float or array-like
        Coordinates (same shape).
    a : float
        Gate half side-length (meters).
    xs : sequence of float
        X–positions of gate centres (meters).
    Vs : sequence of float
        Gate voltages (volts); same length as xs.

    Returns
    -------
    float or np.ndarray
        Analytic potential evaluated at the supplied points.

    Notes
    -----
    Uses the standard rectangle potential primitive g(u,v,z) with the **required minus sign**
    (consistent with Eq. (10)-style superposition logic for the potential).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    xs = np.asarray(xs, dtype=float)
    Vs = np.asarray(Vs, dtype=float)
    if xs.shape != Vs.shape:
        raise ValueError("xs and Vs must have the same length")

    s = np.zeros_like(np.broadcast_to(x, np.broadcast(x, y, z).shape), dtype=float)
    for xi, Vi in zip(xs, Vs):
        s += Vi * (
            g_uvz(a - xi + x, a + y, z)
            + g_uvz(a - xi + x, a - y, z)
            + g_uvz(a + xi - x, a + y, z)
            + g_uvz(a + xi - x, a - y, z)
        )
    return -s
