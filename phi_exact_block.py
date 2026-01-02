# ---- Exact Jackson interface solution and error computation ----

def phi_exact_eval(x):
    """
    Analytic Jackson point-charge + dielectric interface solution.

    x: (3, N) array of coordinates [X; Y; Z]
    returns: (N,) array of phi_exact values.
    Assumes:
      - point charge q in medium eps1 at (0,0,z0), z0 > 0
      - planar interface at z=0 with medium eps2 below
    """
    X, Y, Z = x

    # Distances to real charge and its image
    r1 = np.sqrt(X**2 + Y**2 + (Z - z0)**2)
    r2 = np.sqrt(X**2 + Y**2 + (Z + z0)**2)

    # Avoid division by zero at the charge location (won't be sampled often,
    # but just in case; this is only for error evaluation anyway).
    r1 = np.where(r1 < 1e-30, 1e-30, r1)
    r2 = np.where(r2 < 1e-30, 1e-30, r2)

    # Image-charge coefficients from Jackson
    beta = (eps1 - eps2) / (eps1 + eps2)         # strength of image in region 1
    gamma = 2.0 * eps2 / (eps1 + eps2)          # effective charge seen in region 2

    phi = np.zeros_like(Z)

    # Region 1: z >= 0
    mask_up = Z >= 0.0
    if np.any(mask_up):
        phi[mask_up] = (q / (4.0 * np.pi * eps0 * eps1)) * (
            1.0 / r1[mask_up] + beta / r2[mask_up]
        )

    # Region 2: z < 0
    mask_down = ~mask_up
    if np.any(mask_down):
        phi[mask_down] = (q * gamma / (4.0 * np.pi * eps0 * eps2)) * (
            1.0 / r1[mask_down]
        )

    return phi  # shape (N,)


# Interpolate exact solution into same FE space as phi
phi_exact = fem.Function(V_phi, name="phi_exact")
phi_exact.interpolate(phi_exact_eval)

# Compute L2 and relative L2 errors
dx = ufl.dx(domain)

err_L2 = np.sqrt(
    fem.assemble_scalar(
        fem.form((phi - phi_exact) ** 2 * dx)
    )
)

norm_exact = np.sqrt(
    fem.assemble_scalar(
        fem.form(phi_exact ** 2 * dx)
    )
)

rel_L2 = err_L2 / norm_exact

if MPI.COMM_WORLD.rank == 0:
    print(f"L2 error (phi)      = {err_L2:.6e}")
    print(f"Relative L2 error   = {rel_L2:.6e}")
