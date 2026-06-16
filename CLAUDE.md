# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Runtime Environment

All code requires **dolfinx** (modern FEniCS), which is not pip-installable in the standard way. The canonical way to run anything is through the Docker wrapper:

```bash
./run_dolfinx.sh <script.py> [args]
```

This mounts `$PWD` as `/app` inside `dolfinx/dolfinx:nightly`, sets `PYTHONPATH=/app/src`, and runs `/dolfinx-env/bin/python3 -u <script.py> [args]` inside the container. The `Makefile` uses this wrapper automatically if `run_dolfinx.sh` is present.

**Do not prefix the script with a literal `python3`** (i.e. do not call `./run_dolfinx.sh python3 script.py`) — `run_dolfinx.sh`'s internal `sh -lc '...' -- "$@"` causes the shell's positional-parameter handling to swallow that token, and the container's python3 ends up trying to open a file literally named `python3`, failing with `can't open file '/app/python3'`. Always call it as `./run_dolfinx.sh script.py [args]`, matching how the `Makefile` already invokes it.

To build the Docker image with the stable tag (includes gmsh, matplotlib, pandas, pyvista, pytest):

```bash
docker build -t poissonsolver .
```

## Commands

```bash
# Run a specific case
make run-2d-dir          # 2D Dirichlet, manufactured solution
make run-2d-mix          # 2D Dirichlet + Neumann
make run-3d-dir          # 3D Dirichlet
make run-3d-mix          # 3D Dirichlet + Neumann
make run-disk2d          # 2D Laplace, disk-in-box

# Generic run
make run CASE=3d_mixed

# Tests (Method of Manufactured Solutions)
make test                # runs both test-2d and test-3d
make test-2d             # ./run_dolfinx.sh tests/test_mms_2d.py
make test-3d             # ./run_dolfinx.sh tests/test_mms_3d.py

# Triple-method sphere benchmark (FEM-Dirichlet / FEM-Periodic / FFT)
# Mesh sweep: h = 20, 10, 5, 1, 0.1 nm. FEM legs cap at h >= 5 nm (1/0.1 nm
# OOM/timeout); FFT runs all five, capped by --fft-max-n on grid points/dim.
./benchmarks/run_sphere_benchmark.sh
./benchmarks/run_sphere_benchmark.sh --skip-per          # skip if dolfinx_mpc absent
./run_dolfinx.sh benchmarks/sphere_triple_comparison.py --help

# FFT-only leg of the above, pure NumPy -- no Docker/dolfinx required
python3 benchmarks/fft_only_sweep.py
# IMPORTANT: at the default L=500nm/R=50nm, FFT/FEM-Periodic error vs. the
# isolated-sphere analytic solution is dominated by periodic-image
# contamination (R/L ratio), not mesh density -- see SUGGESTIONS.txt item 0
# (written by sphere_triple_comparison.py) before chasing "accuracy" by
# refining h further.

# 4-quadrant mesh-density demo: one mesh, 4 zones at 20/10/5/1 nm
./benchmarks/run_quadrant_demo.sh

# Clean outputs
make clean               # remove results/*.xdmf and *.h5
make really-clean        # also remove __pycache__, .pytest_cache, .ruff_cache, build
```

Tests exit with code 0 on pass, 1 on failure. They are standalone scripts (not collected by pytest directly — run them via `make`).

## Repository Layout

```
src/poisson/      Core solver library (installable package)
  fem_solve.py    Low-level FEM assembly (exposes KSP object)
  fft_solve.py    Spectral Poisson solver on uniform periodic grids
  geom_sphere.py  sphere_refined_box / sphere_in_box (gmsh, nm units)
  refinement.py   RefinementBox + build_refined_mesh_3d (gmsh Box field)
  analytics.py    Analytic solutions: image charge, conducting sphere, Coulomb
  charges.py      Gaussian charge density (2D/3D)
  materials.py    piecewise_eps_DG0_2regions
main/             Driver and MMS test suite
  main.py         CLI entry point (-m src.main via Makefile)
  Solver/         solve_dirichlet, solve_mixed, norms
  Geometry/       unit_square, unit_cube, disk_in_box
  physics/        dg0_from_indicator (DG0 permittivity)
  tests/          test_mms_2d.py, test_mms_3d.py
benchmarks/       Multi-method benchmarks
  sphere_triple_comparison.py  FEM-Dir vs FEM-Per vs FFT on sphere geometry
  run_sphere_benchmark.sh      Docker wrapper for the above
  fft_only_sweep.py            FFT leg only, pure NumPy, no Docker/dolfinx needed
  quadrant_mesh_density_demo.py  One mesh, 4 zones at distinct h (20/10/5/1 nm)
  run_quadrant_demo.sh         Docker wrapper for the quadrant demo
src/verify/       Standalone verification benchmarks
PC/               Point-charge problem scripts
scripts/          Utility/sweep shell scripts
legacy/           Inactive legacy code
results/          Generated outputs (gitignored)
```

## Architecture

### PDE Being Solved

The solver targets the electrostatic Poisson equation:

```
-div(ε · grad φ) = ρ
```

where `ε` is a piecewise-constant permittivity (DG0) and `ρ` is a charge density (Gaussian-regularized for point charges).

### Two Solver Paths

There are **two parallel solver implementations** — the library in `src/poisson/` and the driver module in `main/`. They are not fully unified:

- **`src/poisson/fem_solve.py`** — lower-level: assembles PETSc matrices manually, exposes the KSP solver object, reads solver options from `PETSC_OPTIONS` or command line.
- **`main/Solver/poisson.py`** — higher-level: wraps `dolfinx.fem.petsc.LinearProblem`, exposes `solve_dirichlet` / `solve_mixed` / `norms`. This is what `main/main.py` and the MMS tests use.

The `main/` solver path is the active one for running cases and tests. The `src/poisson/` library is used by the `src/verify/` benchmarks and can be used independently.

### Permittivity (ε)

Permittivity is always represented as a **DG0 function** (piecewise constant, one value per cell). The factory function is `dg0_from_indicator(domain, indicator_fn, eps_true, eps_false)` in `main/physics/permittivity.py`. For multi-material problems, the indicator is a lambda over cell coordinates (e.g., `lambda X: X[0] < 0.5`).

### Charge Sources (ρ)

Point charges are regularized as isotropic Gaussians via `gaussian_rho(domain, x0, q, sigma)` in `src/poisson/charges.py`. The normalization is exact in both 2D and 3D.

### Boundary Conditions

BCs are composed manually before calling the solver:
- **Dirichlet**: `fem.dirichletbc` on DOFs located with `fem.locate_dofs_topological`
- **Neumann**: passed as `neumann_terms=[(g_expr, facet_tag)]` to `solve_mixed`; requires a `meshtags` object on facets

### FFT Solver

`src/poisson/fft_solve.py` provides a spectral Poisson solver for uniform periodic grids:
- `fft_poisson_3d(rho_grid, L_m, eps_r)` — solves −ε₀εᵣ∇²φ = ρ via FFT in O(N³ log N).
- `gaussian_rho_grid` / `sphere_charge_shell_grid` — build source arrays on the grid.
- Requires cubic grids; uniform permittivity only; periodic BCs on all faces.
- k=0 mode forced to zero (zero-mean gauge; equivalent to a neutralising background).
- Memory: N³ float64 arrays. Use `max_n_for_memory()` to find safe N for a given RAM budget.

### Mesh Generation

Three approaches coexist:
1. **dolfinx built-in**: `unit_square`, `unit_cube`, `disk_in_box` in `main/Geometry/geometry.py` — used for MMS tests and standard cases.
2. **Gmsh box fields**: `build_refined_mesh_3d` in `src/poisson/refinement.py` — axis-aligned `RefinementBox` regions; multiple boxes combine via Gmsh's `Min` field.
3. **Gmsh sphere geometry**: `sphere_refined_box` and `sphere_in_box` in `src/poisson/geom_sphere.py` — use a Gmsh `Ball` field for spherical refinement zones (nm units); `sphere_in_box` cuts a spherical void for conducting-sphere problems (facet tag 2 = sphere surface, tag 1 = outer box).

### Output Format

All field output is XDMF + HDF5 (`.xdmf` + `.h5` pair), viewable in **ParaView**. The `make view-2d` / `make view-3d` targets open the most recent output. Output files go to `results/` with timestamped names.

## Key Conventions

- **PYTHONPATH**: `src/` must be on the path. The Docker wrapper sets this; locally you need `export PYTHONPATH="$PWD/src:$PYTHONPATH"`.
- **MPI**: All dolfinx operations require an MPI communicator (`MPI.COMM_WORLD`). Scripts are MPI-aware; only rank 0 should print results.
- **PETSc solver defaults**: CG with HYPRE preconditioner, `rtol=1e-9` (low-level path) or `rtol=1e-10` (high-level path). Override via `PETSC_OPTIONS` env var or command-line PETSc flags.
- **Units**: All geometry is in metres (consistent SI units assumed throughout).
- **Verification**: New solver features should be verified against either a manufactured solution (MMS) or an analytic benchmark (image charge theory in `src/poisson/analytics.py`).
