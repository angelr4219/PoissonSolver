import numpy as np
def g_uvz(u, v, z):
    return (1.0/(2.0*np.pi)) * np.arctan2(u*v, z*np.sqrt(u*u + v*v + z*z))
def phi0_rect_three_gates_factory(a, zbar, x_positions, Vgates):
    x_positions = np.asarray(x_positions, dtype=float)
    Vgates = np.asarray(Vgates, dtype=float)
    def exact_fun(X):
        x = X[0]; y = X[1]
        z = zbar if X.shape[0] == 2 else X[2]
        val = np.zeros_like(x)
        for xi, Vi in zip(x_positions, Vgates):
            u1 =  a - (xi - x); v1 =  a + y
            u2 =  a - (xi - x); v2 =  a - y
            u3 =  a + (xi - x); v3 =  a + y
            u4 =  a + (xi - x); v4 =  a - y
            val -= Vi * (g_uvz(u1, v1, z) + g_uvz(u2, v2, z) + g_uvz(u3, v3, z) + g_uvz(u4, v4, z))
        return val[np.newaxis, :]
    return exact_fun
