Mesh-refinement tradeoff: 4-quadrant mixed mesh vs. uniform coarse vs. uniform fine
======================================================================================
Same point-charge Poisson problem solved on three different meshes, same
domain (80nm cube), same Dirichlet (phi=0) BC, same probe points -- only
the mesh differs. Reference: analytic isolated Coulomb potential.

  uniform_coarse : h = 20 nm everywhere
  uniform_fine   : h =  1 nm everywhere
  quadrant_mixed : 4 quadrants, each a different h (the "4 quadrants" idea
                   from quadrant_mesh_density_demo.py, but now actually
                   solving a real point-charge problem on it, not just
                   measuring mesh statistics):
                     Q1 (x<0,y<0): h=20nm   Q2 (x>0,y<0): h=10nm
                     Q3 (x<0,y>0): h= 5nm   Q4 (x>0,y>0): h= 1nm <- charge here
                   The charge sits at the centre of Q4 (the finest zone);
                   probe points (r=5,10,15nm) are inside Q4 too, so the
                   comparison is testing exactly where the local refinement
                   is supposed to matter most.

RAW RESULTS
------------------------------------------------------------------------
config          mesh                     n_cells    n_dofs  t_mesh[s]  t_solve[s]  max_relerr  mean_relerr
uniform_coarse  20.0nm uniform               384       125       0.00       0.65    9.676e-01   9.641e-01
uniform_fine     1.0nm uniform         3,072,000   531,441       5.87       9.54    6.355e-01   4.003e-01
quadrant_mixed  Q1=20/Q2=10/Q3=5/Q4=1nm  577,618    98,124      17.27       3.65    6.371e-01   4.079e-01

(full per-probe-point numbers in report.txt / tradeoff_results.csv)

WHAT THIS SHOWS
------------------------------------------------------------------------
1. Accuracy: quadrant_mixed (max_rel_err 0.6371) is essentially IDENTICAL
   to uniform_fine (0.6355) -- within 0.2%. That makes sense: both have the
   same h=1nm resolution right where the charge and probes are (Q4); the
   coarser zones elsewhere don't affect the near-field result. uniform_coarse
   is far worse (0.968) because at h=20nm the mesh can't resolve a
   sigma=2nm Gaussian source at all -- the charge is essentially smeared
   over a single element.

2. DOF count: quadrant_mixed uses 98,124 dofs vs. uniform_fine's 531,441 --
   5.4x fewer -- for the same near-charge accuracy. This is the whole point
   of local refinement: you only pay for resolution where the field actually
   needs it.

3. Time -- the nuance: quadrant_mixed's SOLVE is faster (3.65s vs 9.54s,
   tracking the DOF reduction), but its MESH GENERATION is slower (17.27s
   vs 5.87s), so the *total* wall time is actually higher for quadrant_mixed
   in this particular run (20.92s vs 15.41s). This is not a refinement
   penalty -- it's because uniform_fine uses dolfinx's built-in structured
   box-mesh generator (cheap, almost free, O(N) construction), while
   quadrant_mixed goes through gmsh's unstructured Delaunay tetrahedralizer
   (has real per-run overhead: Delaunay refinement + mesh optimization
   passes, visible in the gmsh log in this run). For a problem this small,
   that fixed gmsh overhead dominates.
   The practical takeaway: if the mesh is solved ONCE, a structured uniform
   mesh can beat a locally-refined gmsh mesh on wall-clock time even with
   far more dofs, simply because mesh generation is "free" for the uniform
   case. The locally-refined mesh wins decisively when:
     (a) the mesh is reused across many solves (mesh-gen cost amortizes,
         solve-time savings compound), or
     (b) the uniform mesh needed to resolve the same feature would have to
         span a domain large enough that O(N^3) uniform scaling becomes
         the bottleneck (e.g. a bigger domain, or a finer feature than 1nm) --
         at that point gmsh's overhead is fixed-cost while uniform-fine's
         cost keeps growing.
   So: "does the 4-quadrant refinement have just as good results and run
   faster" -- yes on accuracy and yes on solve time, but no (this run) on
   total wall time, for the specific reason above, not because local
   refinement is a bad strategy.

FILES IN THIS FOLDER
------------------------------------------------------------------------
  mesh_refinement_tradeoff.py   the script (all 3 configs in one run)
  RUN_COMMANDS.txt              exact command used
  report.txt                    consolidated raw results (all probe points)
  tradeoff_results.csv          summary table (one row per config)
  uniform_coarse.xdmf/.h5       phi field, 20nm uniform mesh (ParaView)
  uniform_fine.xdmf/.h5         phi field, 1nm uniform mesh (ParaView)
  quadrant_mixed.xdmf/.h5       phi field, 4-quadrant mixed mesh (ParaView)
