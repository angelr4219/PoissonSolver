"""
Spectral (FFT) Poisson solver on a uniform 3-D Cartesian grid.

Solves  -eps0 * nabla^2 phi = rho  via the discrete Fourier transform.

Limitations
-----------
* Periodic boundary conditions only (all 3 directions).
* Constant permittivity (eps0); variable eps is not supported.
* The k=0 (DC) mode is forced to zero (charge-neutral assumption).
* Memory: O(N^3) complex128 array.  Call max_n_for_memory() first.
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12  # F/m


def max_n_for_memory(fraction: float = 0.25) -> int:
    """
    Return the largest N such that an N^3 complex128 array fits in
    `fraction` of available system RAM.  Falls back to 256 on error.
    """
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        avail = 2 * 1024 ** 3  # assume 2 GB
    bytes_per_cell = 16  # complex128
    n_max = int((fraction * avail / bytes_per_cell) ** (1.0 / 3.0))
    return max(8, n_max)


def fft_poisson_3d(
    rho: np.ndarray,
    dx: float,
    dy: float | None = None,
    dz: float | None = None,
) -> np.ndarray:
    """
    Solve  -eps0 * nabla^2 phi = rho  on a uniform periodic grid.

    Parameters
    ----------
    rho : (Nx, Ny, Nz) float array   [C/m^3]
    dx  : grid spacing in x          [m]
    dy  : grid spacing in y (default = dx)
    dz  : grid spacing in z (default = dx)

    Returns
    -------
    phi : (Nx, Ny, Nz) float array   [V]
    """
    dy = dy if dy is not None else dx
    dz = dz if dz is not None else dx

    Nx, Ny, Nz = rho.shape

    # Wave-numbers via rfft convention for full complex transform
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2

    rho_hat = np.fft.fftn(rho)

    # Avoid division by zero at DC mode; force phi_DC = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_hat = np.where(K2 == 0.0, 0.0, rho_hat / (EPS0 * K2))

    phi = np.real(np.fft.ifftn(phi_hat))
    return phi


def sphere_charge_shell_grid(
    N: int,
    L: float,
    R: float,
    Q: float,
    sigma_nm: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Represent a conducting sphere of radius R and total charge Q as a
    thin Gaussian shell smeared over a uniform NxNxN grid.

    Parameters
    ----------
    N       : grid points per side
    L       : half-side of cubic box [m]  (box is [-L, L]^3)
    R       : sphere radius [m]
    Q       : total charge [C]
    sigma_nm: Gaussian smearing width in nm

    Returns
    -------
    rho : (N, N, N) charge density [C/m^3]
    xyz : (N,) coordinate array [m] (same for all 3 axes)
    """
    sigma = sigma_nm * 1e-9
    xyz = np.linspace(-L, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(xyz, xyz, xyz, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    shell = np.exp(-0.5 * ((r - R) / sigma) ** 2)
    shell /= shell.sum()  # normalise to unit charge
    dx = xyz[1] - xyz[0]
    rho = shell * Q / dx ** 3  # convert to density

    return rho, xyz


def phi_analytic_sphere(xyz: np.ndarray, R: float, Q: float) -> np.ndarray:
    """
    Analytic potential of a uniformly charged sphere (outside: Coulomb,
    inside: constant = surface value).

    Parameters
    ----------
    xyz : (N,) 1-D coordinate array [m]
    R   : sphere radius [m]
    Q   : total charge [C]

    Returns phi on the NxNxN grid derived from xyz [V].
    """
    X, Y, Z = np.meshgrid(xyz, xyz, xyz, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(r >= R, Q / (4.0 * np.pi * EPS0 * np.where(r > 0, r, 1.0)),
                       Q / (4.0 * np.pi * EPS0 * R))
    return phi
