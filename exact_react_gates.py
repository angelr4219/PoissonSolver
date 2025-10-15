# exact_rect_gates.py
import numpy as np

def g_uvz(u, v, z):
    # g(u,v,z) = (1/(2π)) * atan( uv / ( z * sqrt(u^2 + v^2 + z^2) ) )
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))

def phi0_rect_three_gates_factory(a, zbar, x_positions, Vgates):
    """
    Returns a callable f(x) for dolfinx.interpolate.
    - a: half side-length of each square gate (meters)
    - zbar: depth where you're sampling analytic potential
    - x_positions: list/array of gate centers in x, e.g. [-2*a, 0.0, 2*a]
    - Vgates: list/array of gate voltages same length as x_positions
    """
    x_positions = np.asarray(x_positions, dtype=float)
    Vgates = np.asarray(Vgates, dtype=float)

    def exact_fun(X):
        # X shape: (gdim, npoints)
        x = X[0]; y = X[1]
        # If 3D mesh, ignore X[2] and use fixed zbar for analytic slice:
        # If your problem is 3D with φ(x,y,z), set z = X[2] instead.
        z = zbar if X.shape[0] == 2 else X[2]

        val = np.zeros_like(x)
        for xi, Vi in zip(x_positions, Vgates):
            # Four-corner sum per gate square centered at (xi, 0)
            u1 =  a - (xi - x); v1 =  a + y
            u2 =  a - (xi - x); v2 =  a - y
            u3 =  a + (xi - x); v3 =  a + y
            u4 =  a + (xi - x); v4 =  a - y
            val -= Vi * (g_uvz(u1, v1, z) + g_uvz(u2, v2, z) + g_uvz(u3, v3, z) + g_uvz(u4, v4, z))
        return val[np.newaxis, :]  # shape (1, npoints) for scalar field
    return exact_fun

