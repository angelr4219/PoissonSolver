Sphere benchmark -- conducting sphere in a dielectric box
============================================================
Geometry: conducting sphere, radius R=50nm, held at V0=1V, centred in a
cubic box of side L=500nm.  Outer box surface: phi=0 (FEM-Dirichlet) or
periodic (FFT).  Reference: poisson.analytics.phi_conducting_sphere
(isolated-sphere analytic solution).

WHAT WAS ACTUALLY RUN -- 2 of the intended 3 methods
------------------------------------------------------
  FEM-Dirichlet : ran for real (Docker/dolfinx), h = 20, 10, 5 nm
                  (h = 1, 0.1 nm skipped -- mesh too large/slow for this leg)
  FEM-Periodic  : NEVER RAN. Needs dolfinx_mpc, which has no PyPI wheel and
                  can't be installed in the dolfinx/dolfinx:nightly image
                  used here (no conda/mamba present to build it from
                  conda-forge either). The script detects this
                  ("MPC available: False") and skips the leg automatically.
  FFT           : ran for real (pure NumPy, no Docker), h = 20, 10, 5, 1 nm
                  (h = 0.1nm skipped -- N=5000/dim needs ~1TB RAM)

So "periodic" in this benchmark is the FFT leg standing in for the
unavailable FEM-Periodic leg, per CLAUDE.md's documented limitation.

RAW RESULTS (max relative error vs. analytic isolated-sphere solution)
------------------------------------------------------------------------
  method          h[nm]      dofs/N        t_solve[s]   max_rel_err
  FEM-Dirichlet    20.0        1,845             7.21       57.8%
  FEM-Dirichlet    10.0       12,911             0.34       58.2%
  FEM-Dirichlet     5.0       96,346             3.71       60.8%
  FFT              20.0       15,625             0.01      106.0%
  FFT              10.0      125,000             0.01      100.8%
  FFT               5.0    1,000,000             0.08      101.1%
  FFT               1.0  125,000,000            49.22      101.1%

Full per-run numbers: comparison_results.csv (FEM-Dirichlet + FFT h<=5nm)
and fft_only_results.csv (FFT, full h sweep including h=1nm).

WHY THE ERRORS LOOK SO LARGE -- this is a real, understood effect, not a bug
------------------------------------------------------------------------
Both methods plateau (error barely changes across an 11x mesh refinement,
20nm -> ~2nm-equivalent for FFT) because the dominant error source is NOT
mesh density here -- it's domain truncation:
  - FEM-Dirichlet: phi=0 enforced at the box face only L=500nm away (R/L=0.1)
    is itself a 50-60%-level approximation to the true open-boundary decay
    phi ~ V0*R/r: refining the mesh just gets you a more *precise* solution
    of the *wrong* (truncated) boundary value problem.
  - FFT: solves an infinite *periodic* array of spheres, not one isolated
    sphere -- the periodic images contaminate the field at this R/L=0.1
    ratio regardless of grid resolution, plateauing near 100%.
This is documented in detail, with the supporting L=2000nm control run, in
ANALYSIS.txt item 0 (root-caused and confirmed empirically in an earlier
session). Bottom line: to get these errors down, increase L/R (box size
relative to sphere), not mesh density -- see ANALYSIS.txt for the concrete
options (larger domain, far-field/Robin BC, or an explicit periodic-image
correction).

FILES IN THIS FOLDER
------------------------------------------------------------------------
  sphere_triple_comparison.py   the script (FEM-Dirichlet + FEM-Periodic[skipped] + FFT)
  fft_only_sweep.py             standalone FFT-only script (no Docker needed)
  RUN_COMMANDS.txt              exact commands used to produce every file here
  comparison_results.csv        FEM-Dirichlet + FFT summary (h=20,10,5nm)
  fft_only_results.csv          FFT summary, full sweep incl. h=1nm
  dir_h{20,10,5}nm.xdmf/.h5     FEM-Dirichlet phi field per mesh size (ParaView)
  fft_h{20,10,5}nm.npz          FFT phi grid per mesh size (NumPy, np.load)
  ANALYSIS.txt                  detailed root-cause analysis + suggestions
                                 (originally written as SUGGESTIONS.txt)
