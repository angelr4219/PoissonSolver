# main.py
import argparse
import numpy as np

# ---- Your modules ----
from poisson_linear import poisson_linear  # FEniCSx linear solver (returns dolfinx Function)

# The following modules are your finite-difference versions you shared earlier.
# They may expose functions with these names; we try them in order.
from poisson_intro_solution import *   # expected: assemble_poisson_mms(Nx, Ny) -> (phi, err_l2, err_max)
from poissonLinear2 import *           # expected: solve_poisson_linear(Nx, Ny, eps_r, rho_val, phi_left, phi_right) -> phi
from DoS_2D import *                   # expected: solve_poisson_2d_dos(...) -> phi

def _as_numpy(result):
    """
    Normalize solver output to a NumPy array for reporting.
    - If result is a dolfinx Function, return its vector array.
    - If result is a tuple like (phi, err1, err2), return phi.
    - If result already looks like a NumPy array, return it.
    """
    # dolfinx Function
    try:
        from dolfinx.fem import Function as _FenicsFunction  # type: ignore
        if isinstance(result, _FenicsFunction):
            return result.x.array
    except Exception:
        pass

    # Tuple return (e.g., MMS FD: (phi, err_l2, err_max))
    if isinstance(result, tuple) and len(result) >= 1:
        first = result[0]
        if isinstance(first, np.ndarray):
            return first

    # Already a NumPy array
    if isinstance(result, np.ndarray):
        return result

    # Unknown → return None
    return None

def run_mms(args):
    """
    Try FD MMS first (assemble_poisson_mms), else fallback to FEniCS MMS if you add it later.
    """
    # Try FD method-of-manufactured-solutions
    for candidate in ("assemble_poisson_mms", "poisson_mms"):
        fn = globals().get(candidate, None)
        if callable(fn):
            res = fn(Nx=args.Nx, Ny=args.Ny) if "assemble" in candidate else fn(args.Nx, args.Ny)
            arr = _as_numpy(res)
            if arr is not None and arr.size:
                print(f"[MMS] phi: min={arr.min():.3e} V, max={arr.max():.3e} V")
            else:
                print("[MMS] Finished (no array to report).")
            # If FD returned errors, print them
            if isinstance(res, tuple) and len(res) >= 3:
                _, err_l2, err_max = res[:3]
                print(f"[MMS] L2 error={err_l2:.6e}, max error={err_max:.6e}")
            return
    print("[MMS] No MMS function found. Implement assemble_poisson_mms(...) or poisson_mms(...).")

def run_linear(args):
    """
    Prefer FEniCSx linear solver (poisson_linear).
    If you want FD variant instead, call run_linear_fd.
    """
    uh = poisson_linear(Nx=args.Nx, Ny=args.Ny,
                        eps_r=args.eps_r, rho_val=args.rho,
                        phi_left=args.phi_left, phi_right=args.phi_right)
    arr = _as_numpy(uh)
    if arr is not None and arr.size:
        print(f"[Linear] phi: min={arr.min():.3e} V, max={arr.max():.3e} V")

def run_linear_fd(args):
    """
    Finite-difference linear solver from poissonLinear2 (solve_poisson_linear).
    """
    fn = globals().get("solve_poisson_linear", None)
    if not callable(fn):
        print("[Linear-FD] solve_poisson_linear(...) not found in poissonLinear2.")
        return
    phi = fn(args.Nx, args.Ny, args.eps_r, args.rho, args.phi_left, args.phi_right)
    arr = _as_numpy(phi)
    if arr is not None and arr.size:
        print(f"[Linear-FD] phi: min={arr.min():.3e} V, max={arr.max():.3e} V")

def run_dos(args):
    """
    Nonlinear DOS reservoir (prefer FD function name you shared: solve_poisson_2d_dos).
    If you later add a FEniCSx version, you can wire it similarly.
    """
    fn = globals().get("solve_poisson_2d_dos", None)
    if not callable(fn):
        print("[DOS] solve_poisson_2d_dos(...) not found in DoS_2D.")
        return

    # Map CLI eV inputs to the FD signature if it expects Joules or eV.
    # Your original FD used Joules; here we pass eV-like params and let the function handle conversion if it does.
    phi = fn(Nx=args.Nx, Ny=args.Ny, eps_r=args.eps_r, rho_vol=args.rho,
             phi_left=args.phi_left, phi_right=args.phi_right,
             mu=args.mu_eV * 1.602176634e-19,  # if your function expects Joules
             Ec0=args.Ec0_eV * 1.602176634e-19,
             g_s=args.g_s, g_v=args.g_v, m_eff=args.m_eff, T=args.T,
             max_iter=args.max_iter, rtol=args.rtol, damp=args.damp)
    arr = _as_numpy(phi)
    if arr is not None and arr.size:
        print(f"[DOS] phi: min={arr.min():.3e} V, max={arr.max():.3e} V")

def main():
    p = argparse.ArgumentParser(description="Run Poisson solvers (FEniCSx or FD).")
    p.add_argument("--mode", choices=["mms", "linear", "linear-fd", "dos"], default="linear",
                   help="Which solver to run.")
    p.add_argument("--Nx", type=int, default=96)
    p.add_argument("--Ny", type=int, default=96)

    # Physics / materials
    p.add_argument("--eps_r", type=float, default=3.9)
    p.add_argument("--rho", type=float, default=0.0, help="Volume charge density (C/m^3)")
    p.add_argument("--phi_left", type=float, default=0.0)
    p.add_argument("--phi_right", type=float, default=0.2)

    # DOS boundary params
    p.add_argument("--mu_eV", type=float, default=0.10, help="Chemical potential in eV")
    p.add_argument("--Ec0_eV", type=float, default=0.0, help="Conduction band edge offset in eV")
    p.add_argument("--g_s", type=float, default=2.0)
    p.add_argument("--g_v", type=float, default=1.0)
    p.add_argument("--m_eff", type=float, default=0.19*9.1093837015e-31, help="Effective mass in kg (default 0.19 m0)")
    p.add_argument("--T", type=float, default=300.0)

    # Nonlinear solver controls (FD DOS version)
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--damp", type=float, default=1.0)

    args = p.parse_args()

    if args.mode == "mms":
        run_mms(args)
    elif args.mode == "linear":
        run_linear(args)
    elif args.mode == "linear-fd":
        run_linear_fd(args)
    elif args.mode == "dos":
        run_dos(args)

if __name__ == "__main__":
    main()
