    # ------------------------------------------------------------
    # Analytic Jackson solution (delta point charge in half-spaces)
    # ------------------------------------------------------------
    def phi_exact_eval(x):
        """
        Analytic Jackson point charge at planar dielectric interface.
        x: (3, N) coordinate array -> returns (N,) values.
        """

        X, Y, Z = x  # shapes (N,)

        # Physical parameters
        q    = args.q
        z0   = args.z0
        eps1 = eps1_abs
        eps2 = eps2_abs
        eps0 = EPS0

        # Distances to real and image charges
        r1 = np.sqrt(X**2 + Y**2 + (Z - z0)**2)
        r2 = np.sqrt(X**2 + Y**2 + (Z + z0)**2)

        # Prevent division by zero
        r1 = np.where(r1 < 1e-30, 1e-30, r1)
        r2 = np.where(r2 < 1e-30, 1e-30, r2)

        # Image-charge coefficients (Jackson)
        beta  = (eps1 - eps2) / (eps1 + eps2)      # q' / q
        gamma = 2.0 * eps2 / (eps1 + eps2)         # q'' / q

        phi = np.zeros_like(Z)

        # Region 1: z >= 0  (upper medium)
        mask_up = (Z >= 0.0)
        if np.any(mask_up):
            phi[mask_up] = (q / (4 * np.pi * eps0 * eps1)) * (
                1.0 / r1[mask_up] + beta / r2[mask_up]
            )

        # Region 2: z < 0  (lower medium)
        mask_lo = ~mask_up
        if np.any(mask_lo):
            phi[mask_lo] = (q * gamma / (4 * np.pi * eps0 * eps2)) * (
                1.0 / r1[mask_lo]
            )

        return phi

    # Interpolate analytic solution into FE space V
    phi_exact = fem.Function(V, name="phi_exact")
    phi_exact.interpolate(phi_exact_eval)

    if RANK == 0:
        print(f"phi_exact: min={phi_exact.x.array.min():.3e}, "
              f"max={phi_exact.x.array.max():.3e}")

