"""FFT-based Poisson solver for uniform periodic 3-D grids.

Solves  -ε₀ εᵣ ∇²φ = ρ  on a periodic cubic domain [-L/2, L/2]³
via spectral decomposition:

    φ̂(k) = ρ̂(k) / (ε₀ εᵣ |k|²)   (k ≠ 0)
    φ̂(0) = 0                        (zero-mean gauge)

Setting k = 0 to zero is equivalent to subtracting a uniform neutralising
background charge density of  -∫ρ dV / L³.  This is the standard Ewald
convention for periodic charged systems.

Limitations
-----------
* Uniform Cartesian grid only (no adaptive refinement).
* Constant (scalar) permittivity only; variable ε breaks the spectral
  decomposition.
* Periodic BCs on all faces.  For isolated-charge problems the periodic
  images are present; domain must be large enough to suppress them.
* Memory: N³ complex128 arrays.  Keep N ≤ ~300 per dimension for <2 GB
  working set.  The helper `max_n_for_memory` can guide parameter choice.
"""
from __future__ import annotations
import numpy as np

EPS0: float = 8.8541878128e-12   # F/m
_MAX_N_DEFAULT: int = 512        # per-dimension guard


def fft_poisson_3d(
    rho_grid: np.ndarray,
    L_m: float,
    eps_r: float = 1.0,
) -> np.ndarray:
    """Solve -ε₀εᵣ ∇²φ = ρ on a uniform periodic cubic grid.

    Parameters
    ----------
    rho_grid : (N, N, N) float array, charge density [C/m³]
    L_m      : domain side length [m]
    eps_r    : relative permittivity (uniform, dimensionless)

    Returns
    -------
    phi_grid : (N, N, N) float array, electrostatic potential [V]
    """
    Nx, Ny, Nz = rho_grid.shape
    if not (Nx == Ny == Nz):
        raise ValueError("Only cubic grids (Nx == Ny == Nz) are supported.")
    N = Nx
    h = L_m / N

    # Physical wave-vector components  k_m = 2π m / L
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=h)    # shape (N,)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2                     # |k|² grid (N,N,N)
    k2[0, 0, 0] = 1.0   # placeholder; zeroed below

    rho_hat = np.fft.fftn(rho_grid)
    phi_hat = rho_hat / (eps_r * EPS0 * k2)
    phi_hat[0, 0, 0] = 0.0     # zero-mean (neutralising background)

    return np.real(np.fft.ifftn(phi_hat))


def gaussian_rho_grid(
    N: int,
    L_m: float,
    q: float,
    sigma_m: float,
) -> np.ndarray:
    """Gaussian charge distribution on an N³ uniform periodic grid.

    The Gaussian is centred at the grid centre and normalised so that
    the discrete sum equals q (total charge in Coulombs).

    Parameters
    ----------
    N       : grid points per dimension
    L_m     : domain side length [m]
    q       : total charge [C]
    sigma_m : Gaussian half-width (std-dev) [m]

    Returns
    -------
    rho : (N, N, N) float array [C/m³]
    """
    h = L_m / N
    xi = (np.arange(N) - N // 2) * h   # centred coordinates
    X, Y, Z = np.meshgrid(xi, xi, xi, indexing="ij")
    r2 = X**2 + Y**2 + Z**2
    norm = q / ((2.0 * np.pi) ** 1.5 * sigma_m**3)
    rho = norm * np.exp(-r2 / (2.0 * sigma_m**2))
    # Re-normalise to correct for discrete sampling drift
    rho *= q / (rho.sum() * h**3)
    return rho


def sphere_charge_shell_grid(
    N: int,
    L_m: float,
    q: float,
    R_m: float,
    sigma_shell_m: float | None = None,
) -> np.ndarray:
    """Approximate a conducting sphere as a thin Gaussian charge shell.

    The charge q is distributed as a Gaussian ring at radius R_m with
    width sigma_shell (default R_m/8).  This reproduces the conducting-
    sphere far-field  φ ≈ q/(4πε₀r)  for r > R.

    Parameters
    ----------
    N            : grid points per dimension
    L_m          : domain side [m]
    q            : total charge [C]
    R_m          : sphere radius [m]
    sigma_shell_m: Gaussian half-width of the shell [m]
    """
    if sigma_shell_m is None:
        sigma_shell_m = R_m / 8.0
    h = L_m / N
    xi = (np.arange(N) - N // 2) * h
    X, Y, Z = np.meshgrid(xi, xi, xi, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    shell = np.exp(-((r - R_m) ** 2) / (2.0 * sigma_shell_m**2))
    shell /= shell.sum() * h**3   # normalise volume integral to 1
    return q * shell


def evaluate_fft_phi_at_points(
    phi_grid: np.ndarray,
    L_m: float,
    points_m: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation of FFT potential at arbitrary points.

    Parameters
    ----------
    phi_grid : (N,N,N) potential on grid spanning [-L/2, L/2]³ [V]
    L_m      : domain side [m]
    points_m : (M, 3) query coordinates [m]

    Returns
    -------
    phi_vals : (M,) potential values [V]
    """
    N = phi_grid.shape[0]
    h = L_m / N
    # Map from physical coords to grid index; grid spans [-L/2, L/2]
    idx = (points_m / h) + N / 2   # (M, 3)

    i0 = np.floor(idx[:, 0]).astype(int) % N
    j0 = np.floor(idx[:, 1]).astype(int) % N
    k0 = np.floor(idx[:, 2]).astype(int) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    k1 = (k0 + 1) % N

    fx = idx[:, 0] - np.floor(idx[:, 0])
    fy = idx[:, 1] - np.floor(idx[:, 1])
    fz = idx[:, 2] - np.floor(idx[:, 2])

    return (
        phi_grid[i0, j0, k0] * (1-fx)*(1-fy)*(1-fz)
      + phi_grid[i1, j0, k0] *    fx *(1-fy)*(1-fz)
      + phi_grid[i0, j1, k0] * (1-fx)*   fy *(1-fz)
      + phi_grid[i0, j0, k1] * (1-fx)*(1-fy)*   fz
      + phi_grid[i1, j1, k0] *    fx *   fy *(1-fz)
      + phi_grid[i1, j0, k1] *    fx *(1-fy)*   fz
      + phi_grid[i0, j1, k1] * (1-fx)*   fy *   fz
      + phi_grid[i1, j1, k1] *    fx *   fy *   fz
    )


def max_n_for_memory(mem_gb: float = 2.0) -> int:
    """Return the largest N such that three N³ float64 arrays fit in mem_gb GB."""
    bytes_per_array = mem_gb * 1e9 / 3
    return int((bytes_per_array / 8) ** (1/3))
