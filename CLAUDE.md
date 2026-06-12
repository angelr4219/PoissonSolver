# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Runtime Environment

All code requires **dolfinx** (modern FEniCS), which is not pip-installable in the standard way. The canonical way to run anything is through the Docker wrapper:

```bash
./run_dolfinx.sh python3 <script.py> [args]
```

This mounts `$PWD` as `/app` inside `dolfinx/dolfinx:nightly`, sets `PYTHONPATH=/app/src`, and runs the given command. The `Makefile` uses this wrapper automatically if `run_dolfinx.sh` is present.

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

# Clean outputs
make clean               # remove results/*.xdmf and *.h5
make really-clean        # also remove __pycache__, .pytest_cache, .ruff_cache, build
```

Tests exit with code 0 on pass, 1 on failure. They are standalone scripts (not collected by pytest directly — run them via `make`).

## Repository Layout

```
src/poisson/      Core solver library (installable package)
main/             Driver and MMS test suite
  main.py         CLI entry point (-m src.main via Makefile)
  Solver/         solve_dirichlet, solve_mixed, norms
  Geometry/       unit_square, unit_cube, disk_in_box
  physics/        dg0_from_indicator (DG0 permittivity)
  tests/          test_mms_2d.py, test_mms_3d.py
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

### Mesh Generation

Two approaches coexist:
1. **dolfinx built-in**: `unit_square`, `unit_cube`, `disk_in_box` in `main/Geometry/geometry.py` — used for MMS tests and standard cases.
2. **Gmsh**: `build_refined_mesh_3d` in `src/poisson/refinement.py` — used when local mesh refinement is needed. `RefinementBox` dataclass defines axis-aligned refinement regions; multiple boxes combine via Gmsh's `Min` field.

### Output Format

All field output is XDMF + HDF5 (`.xdmf` + `.h5` pair), viewable in **ParaView**. The `make view-2d` / `make view-3d` targets open the most recent output. Output files go to `results/` with timestamped names.

## Key Conventions

- **PYTHONPATH**: `src/` must be on the path. The Docker wrapper sets this; locally you need `export PYTHONPATH="$PWD/src:$PYTHONPATH"`.
- **MPI**: All dolfinx operations require an MPI communicator (`MPI.COMM_WORLD`). Scripts are MPI-aware; only rank 0 should print results.
- **PETSc solver defaults**: CG with HYPRE preconditioner, `rtol=1e-9` (low-level path) or `rtol=1e-10` (high-level path). Override via `PETSC_OPTIONS` env var or command-line PETSc flags.
- **Units**: All geometry is in metres (consistent SI units assumed throughout).
- **Verification**: New solver features should be verified against either a manufactured solution (MMS) or an analytic benchmark (image charge theory in `src/poisson/analytics.py`).
