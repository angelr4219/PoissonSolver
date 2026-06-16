Parallelism: MPI ranks per solve, and concurrent independent configs
========================================================================
Two independent ways to use this sandbox's 4 CPU cores to speed up the
existing benchmarks, both implemented and measured for real this session.

WHAT WAS ADDED
------------------------------------------------------------------------
1. run_dolfinx.sh: new DOLFINX_NP env var. When DOLFINX_NP>1, the inner
   python3 invocation is wrapped with `mpirun -n $DOLFINX_NP` (MPICH,
   ships in dolfinx/dolfinx:nightly -- confirmed present via `which
   mpirun mpiexec`). MPICH's mpirun runs fine as root inside the
   container with no extra flag (unlike OpenMPI, which needs
   --allow-run-as-root -- not needed here since this image ships MPICH).
   Usage: `DOLFINX_NP=4 ./run_dolfinx.sh <script.py> [args]`.
   Default (DOLFINX_NP unset/1) is byte-for-byte the previous behaviour.

2. benchmarks/mesh_refinement_tradeoff.py: new `--only {uniform_coarse,
   uniform_fine,quadrant_mixed}` flag, so each of the 3 configs can run as
   a standalone process instead of all 3 sequentially in one process.
   With --only, the config's result (timings, dofs, probe-point phi
   values) is dumped to <outdir>/<name>_result.json instead of the
   merged report/CSV.

3. benchmarks/merge_tradeoff_results.py: reads the 3 *_result.json files
   and produces the exact same tradeoff_results.csv + report.txt format
   that a plain sequential run produces.

4. benchmarks/run_mesh_refinement_parallel.sh: launches the 3 `--only`
   invocations as 3 separate `./run_dolfinx.sh` (i.e. 3 separate `docker
   run`) background processes, waits for all three, then calls the merge
   script. This is "concurrent independent configs": each config is a
   full gmsh + PETSc process with non-thread-safe global C state, so
   process-level (not thread-level) concurrency is the only safe option.
   Can be combined with mechanism #1 (e.g. `DOLFINX_NP=2
   ./benchmarks/run_mesh_refinement_parallel.sh`), though on a 4-core box
   that means 3 configs x 2 ranks = 6-way oversubscription -- see caveat
   below.

A REAL BUG FOUND AND FIXED ALONG THE WAY
------------------------------------------------------------------------
mesh_refinement_tradeoff.py's run_config() previously gated its XDMFFile
write and (implicitly, via eval_at_points) its probe-point evaluation
behind `if IS_ROOT`/rank-0-only logic. That's fine at DOLFINX_NP=1 (the
only mode it had ever been run in before this session) but is WRONG for
DOLFINX_NP>1:
  - XDMFFile read/write is a collective MPI-IO call -- every rank must
    call it. Gating it behind `if IS_ROOT` made non-root ranks skip the
    call entirely, which deadlocked/corrupted the collective and crashed
    with an MPICH "message truncated" error on the very first DOLFINX_NP=4
    test run (see RUN_COMMANDS.txt for the command that triggered it).
  - eval_at_points's compute_colliding_cells only finds points in cells
    LOCAL to each rank's partition, so whichever rank doesn't happen to
    own a probe point's cell would silently get NaN for it.
Fixed: XDMFFile write is now called on all ranks unconditionally,
and eval_at_points now does a NaN-aware MPI Allreduce(MIN) across ranks
so the one rank that actually owns each probe point's cell supplies the
real value regardless of which rank that is.
Verified fixed: DOLFINX_NP=4 now reproduces the exact same max_rel_err
(6.355e-01) as the DOLFINX_NP=1 / original sequential run for the
uniform_fine config -- see mpi_speedup_results.csv.

REAL MEASURED RESULTS
------------------------------------------------------------------------
(1) MPI ranks per solve -- uniform_fine config alone (h=1nm, 531,441 dofs,
    the heaviest of the 3 configs), via mpi_speedup_results.csv:

      ranks   t_mesh[s]   t_solve[s]   t_total[s]   wall[s]   max_rel_err
        1        6.42        11.29        17.71       23.66    6.355e-01
        4        3.30         6.38         9.68        12.50    6.355e-01

    -> 1.83x speedup on compute (t_total), 1.89x on wall time (includes
    fixed container-startup overhead, ~5-6s either way). Sublinear vs the
    ideal 4x for 4 ranks -- expected: mesh partitioning/repartitioning
    and inter-rank ghost communication add overhead, and this problem
    (531k dofs) is on the small side for a 4-rank decomposition to shine.
    Identical max_rel_err confirms correctness is preserved.

(2) Concurrent independent configs -- all 3 mesh_refinement_tradeoff.py
    configs, concurrent_configs_report.txt/.csv (defaults: L=80nm,
    h-coarse=20nm, h-fine=1nm, quadrant Q1-4=20/10/5/1nm):

      config          t_mesh[s]  t_solve[s]  t_total[s]
      uniform_coarse       0.00       0.79        0.79
      uniform_fine         6.25      11.12       17.38
      quadrant_mixed      21.16       4.82       25.97

      sum of t_total (= what one sequential process would take, and
      matches the original mesh_refinement_tradeoff/report.txt numbers
      within run-to-run noise) = 0.79+17.38+25.97 = 44.1s
      measured wall time running all 3 concurrently           = 28.1s

    -> ~1.6x speedup on this metric, but with real caveats (see below).
    All max_rel_err values match the original sequential run exactly
    (uniform_fine 6.355e-01, quadrant_mixed 6.371e-01, uniform_coarse
    9.676e-01) -- confirms the merge step reconstructs the same numbers,
    just computed concurrently instead of sequentially.

CAVEATS -- read before assuming either approach is free 4x/3x speedup
------------------------------------------------------------------------
- This sandbox has exactly 4 CPU cores (`nproc`). Both mechanisms compete
  for the SAME 4 cores, so they don't compose additively: running 3
  configs concurrently with DOLFINX_NP=4 each would be 12-way
  oversubscription on 4 cores and almost certainly slower than either
  alone, not faster. The two mechanisms are best used separately, or
  combined modestly (e.g. DOLFINX_NP=2 with 2 concurrent configs on a
  4-core box), not multiplicatively.
- The concurrent-configs speedup (1.6x) is well below the naive "3
  configs at once -> should approach 3x" expectation, BECAUSE the 3
  configs are wildly unequal in cost (0.8s / 17.4s / 26.0s) and each is
  itself internally CPU-bound (gmsh's serial Delaunay refinement for
  quadrant_mixed, PETSc CG+HYPRE for the solves) -- with only 4 cores for
  3 concurrent processes, each one runs somewhat slower than it would
  alone (e.g. quadrant_mixed's mesh-gen took 21.2s here concurrently vs
  17.3s alone in the original sequential run, ~22% slower from
  contention). The wall-clock win is real but is bounded by
  max(individual times) plus contention slowdown, not by
  sum(individual times)/3.
- MPI's 1.83x at 4 ranks (not 4x) is typical for FEM problems at this
  size -- the fixed overhead (partitioning, ghost exchange) starts to
  amortize away on either much bigger problems or more ranks-per-core-
  count ratios than this 4-core sandbox can demonstrate.

WHEN TO USE WHICH
------------------------------------------------------------------------
- Use DOLFINX_NP>1 (MPI) when you have ONE big solve and want it faster --
  this is the only mechanism that reduces the cost of a SINGLE config.
- Use the concurrent-configs driver when you have SEVERAL independent
  configs/sweep points (e.g. a mesh-size sweep, multiple charge
  positions) that would otherwise run one after another in a single
  script -- this is the only mechanism that overlaps otherwise-sequential
  work that does not depend on each other.
- They compose, but on a small core count the gain from stacking both is
  marginal-to-negative; pick whichever matches the actual bottleneck
  (one slow solve, vs many independent solves) rather than reflexively
  enabling both.

FILES IN THIS FOLDER
------------------------------------------------------------------------
  RUN_COMMANDS.txt                 exact commands used for every result here
  mpi_speedup_results.csv          NP=1 vs NP=4 timing for uniform_fine
  concurrent_configs_report.txt    merged report from the 3-config concurrent run
  concurrent_configs_results.csv   merged CSV from the 3-config concurrent run
  run_mesh_refinement_parallel.sh  copy of the driver script (lives for real
                                    at benchmarks/run_mesh_refinement_parallel.sh)
  merge_tradeoff_results.py        copy of the merge script (lives for real
                                    at benchmarks/merge_tradeoff_results.py)

  (the DOLFINX_NP env var lives in repo-root run_dolfinx.sh; the --only
  flag lives in benchmarks/mesh_refinement_tradeoff.py -- not duplicated
  here since they're small diffs to existing files, not new standalone
  scripts)
