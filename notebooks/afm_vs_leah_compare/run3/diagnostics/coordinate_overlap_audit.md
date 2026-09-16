# Coordinate / overlap / interpolation-validity diagnostic -- run3 vs Leah VTK

## 1. Load both fields

VTK reference: field='basePotential', N points = 636291
VTK point range -> x: [-150, 150]  y: [-150, 150]  z: [0, 2053]
VTK value range -> [-4.4, 0.159836]

FEM case: field='phi_V', N points = 89527
FEM point range -> x: [-150, 150]  y: [-150, 150]  z: [-320, 2053]
FEM value range -> [-4.4, 1]

run_results.json geometry_nm: {'lx': 300.0, 'ly': 300.0, 'air_height': 320.0, 'tip_gap': 10.0, 'tip_radius': 10.0, 'cone_height': 160.0, 'shank_radius': 120.0, 'cone_half_angle_deg_input': None, 'cone_half_angle_deg_effective': 34.5085229876684, 'shaft_radius': 120.0, 'shaft_height': 100.0, 'probe1_z': 35.0, 'probe2_z': 48.0, 'probe1_radius': 10.0, 'probe2_radius': 10.0, 'bottom_z': 2053.0}
run_results.json voltages_V: {'AFM_tip': 1.0, 'bottom_back_gate': -4.4, 'probe1': None, 'probe2': None}

Units check: calibrate_afm_tip_vs_leah.py builds gmsh geometry directly in the CLI's raw nm values (--lx/--ly default 300.0, no *1e-9/*1e9 conversion anywhere in build_geometry). read_xdmf_case(case, scale=1.0) therefore leaves FEM coordinates un-rescaled, i.e. already in nm, matching the VTK's nm coordinates.  scale=1.0 is the CORRECT choice for this run -- confirmed by inspection of calibrate_afm_tip_vs_leah.py, not assumed.

## 2. Axis / orientation check (z=0 = top surface, z increasing into device)

VTK z=min (0): n=11163, basePotential mean=0.020130, std=0.118484, min=-1.235000, max=0.159836
VTK z=max (2053): n=3721, basePotential mean=-4.400001, std=0.000001, min=-4.400000, max=-4.400000

FEM point z-range -> [-320, 2053] (documented bottom_z=2053.0, air_height=320.0 => expected z-range approx [-320.0, 2053.0])
FEM points at z==2053.0 (tol=0.001): n=511, phi_V mean=-4.400000000, min=-4.400000000, max=-4.400000000  (documented Dirichlet BC bottom_back_gate = -4.4 V)
FEM points at z==0 (nearest_z_mask tol): n=809, phi_V mean=-0.927584, min=-0.982600, max=-0.775635  vs VTK z=0 mean=0.020130

Orientation verdict: PASS -- FEM phi_V at documented bottom_z matches the Dirichlet BC voltage to <1e-6 V, and VTK's own z axis starts at exactly 0.0 (top surface). Both datasets increase z into the device / away from air, consistent orientation.

## 3. x=0 / y=0 alignment

VTK unique x count = 61, range [-150, 150]
VTK unique y count = 61, range [-150, 150]
VTK has an exact x=0.0 grid line: True
VTK has an exact y=0.0 grid line: True

FEM mesh geometric center: build_geometry() in calibrate_afm_tip_vs_leah.py places the AFM tip sphere/cone/shaft, probe1 disk, and probe2 disk all explicitly at (x=0.0, y=0.0, z=...) -- occ.addSphere(0.0, 0.0, ...), occ.addCone(0.0, 0.0, ...), occ.addCylinder(0.0, 0.0, ...), occ.addDisk(0.0, 0.0, z_probe1/2, ...). The lateral domain box is built symmetric: xmin=-lx/2, xmax=+lx/2 (and same for y), so the domain and the tip/probe axis are both exactly centered at x=0, y=0 by construction -- no lateral misalignment was introduced. (No mesh node is guaranteed to sit at exactly x=0,y=0 since the mesh is unstructured, but the geometric symmetry axis is exact.)

FEM point-cloud bbox center (sanity check on symmetry of the actual mesh): x_center=0, y_center=0 (expect ~0 given lx=ly=300.0)

## 4. Bounding-box overlap (via check_bbox_overlap + independent computation)

check_bbox_overlap: no warning raised (bboxes overlap and cover >=50% on every axis)

Independent overlap computation:
  x: ref=[-150,150]  case=[-150,150]  overlap=[-150,150]  overlap_extent=300  coverage_of_ref_extent=100.00%
  y: ref=[-150,150]  case=[-150,150]  overlap=[-150,150]  overlap_extent=300  coverage_of_ref_extent=100.00%
  z: ref=[0,2053]  case=[-320,2053]  overlap=[0,2053]  overlap_extent=2053  coverage_of_ref_extent=100.00%

## 5. Interpolation fallback fractions (global + representative z-slices)

GLOBAL: n_points=636291  fallback_count=0  fallback_frac=0.0000%

  z_target    n_pts  valid(linear)   fallback  fallback_frac    verdict
         0    11163          11163          0        0.0000%       PASS
        35   145119         145119          0        0.0000%       PASS
        48   145119         145119          0        0.0000%       PASS
      1000     3721           3721          0        0.0000%       PASS
      2000     3721           3721          0        0.0000%       PASS

## 6. FEM domain extent: documented config vs actual point cloud

Documented (run_results.json): lx=300.0, ly=300.0  => expected x,y in [-150.0,150.0], [-150.0,150.0]
Documented bottom_z=2053.0, air_height=320.0 => expected z in [-320.0, 2053.0]
Actual FEM point cloud: x in [-150, 150], y in [-150, 150], z in [-320, 2053]
VTK lateral extent actual: x in [-150, 150], y in [-150, 150], z in [0, 2053]

FEM laterally covers full VTK x,y range: True
FEM covers full VTK z range [0, 2053] (ignoring FEM's extra negative-z air region): True

## 7. Air-region (negative z) and z=0 boundary check

FEM points with z < 0 (air region): 43711 of 89527 total (48.82%)
  phi_V range in air region: [-0.836134, 1]
VTK has NO points at z<0 (VTK z-range starts at 0), so any FEM air-region points are only ever used as SOURCE data for griddata (case.points), never as query/target points (ref.points) -- interpolate_case_onto_ref queries only at ref.points, i.e. VTK's own point locations, all of which have z>=0. The negative-z air points correctly never appear as an output row; they can still influence the linear-interpolation weights for VTK z=0 query points if z=0 sits inside the convex hull simplex spanning z<0 and z>0 nodes.

FEM mesh nodes at EXACTLY z=0.0: 809
  unique (x,y) among z=0 nodes: 809, (x,y) locations with >1 coincident z=0 node: 0
  No coincident/duplicate (x,y) nodes found at z=0 -- z=0 appears to be a single well-defined mesh layer, not a duplicated air/Si interface.

## Summary verdicts

Bounding-box overlap: PASS (see section 4 numbers)
Global interpolation fallback fraction: 0.0000%
Worst representative slice: z=0 -- fallback_frac=0.0000% (0 of 11163 points), verdict=PASS

### Slices/regions to EXCLUDE or FLAG for Agent 3's error-metric sweep:

- None of the 5 representative slices (z=0,35,48,1000,2000) showed >0% fallback; all PASS.

General flag regardless of per-slice fallback numbers: any FEM point with z<0 (the explicit air region, negative-z by construction) has NO corresponding VTK data and must never be used as a ground-truth comparison target -- interpolate_case_onto_ref already only queries at VTK point locations (z>=0), so as long as Agent 3 continues to treat VTK as ref and FEM as case (not vice versa), air-region points are structurally excluded from the metric by construction. If Agent 3's sweep ever queries FEM values directly (bypassing interpolate_case_onto_ref) at z<0, those rows must be dropped before any VTK-comparison metric is computed.
