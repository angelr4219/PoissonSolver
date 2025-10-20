# POISSONSOLVER



This repo solves and verifies an electrostatics benchmark with three rectangular metal gates on the top surface of a 3D box. The forward solve uses dolfinx/PETSc (Laplace case: ρ=0; easily extensible to Poisson), and the verification step compares the numerical solution to a closed-form analytic reference along well-defined lines/regions. Outputs (fields and error maps) are written as XDMF/HDF5 for easy inspection in ParaView.

What’s here (high-level)

Forward model (solve): build the box mesh, imprint gate pads, apply BCs, assemble and solve for potential ϕ.

Verification: load the solved ϕ, evaluate the analytic ϕ_ref, compute nodal and cellwise relative errors, and write summaries plus XDMF files that can be visualized in ParaView.

Experiments: scripts to sweep padding (domain size) and mesh density, then gather error summaries.

Repository layout (key files)
verify/
├─ compare_rect_vs_analytic.py   # CLI: load XDMF, build analytic φ, compute errors, write CSV+XDMF
└─ analytic_rect.py              # Analytic potential for three rectangular gates

scripts/
├─ verify_rect_p2a.sh            # Verify a single case (p = 2a)
├─ verify_rect_all.sh            # Verify p = 2a, 3a, 4a, 5a
├─ regen_rect_cases.sh           # Re-generate solves across {padding × mesh}
└─ verify_rect_cases.sh          # Verify everything under Results/cases/**


verify/compare_rect_vs_analytic.py
Loads a numerical solution phi from XDMF, constructs the analytic solution for the three-gate geometry, and computes:

Nodal relative error (POINT data): rel_error_dof

Cellwise relative error (CELL data): rel_error_cell
It writes both error fields to XDMF and a compact CSV with L∞/L² (nodal) and max/mean (cellwise).

verify/analytic_rect.py
Implements the closed-form potential for three rectangular electrodes on the plane 
𝑧
=
0
z=0. Exposes a clean API used by the comparator to sample the reference solution at arbitrary points and depths.

scripts/verify_rect_p2a.sh
One-shot verifier for the p=2a case. Installs minimal Python deps inside the dolfinx/dolfinx:stable container, adds the repo to PYTHONPATH, and runs the comparator on Results/phi_p2a.xdmf. Produces:

Results/verify/phi_p2a_rel_error_nodal.xdmf
Results/verify/phi_p2a_rel_error_cellwise.xdmf
Results/verify/phi_p2a_summary.csv


scripts/verify_rect_all.sh
Convenience wrapper that repeats the same verification for p = 2a, 3a, 4a, 5a (Results/phi_p{2,3,4,5}a.xdmf), producing matched XDMF/CSV outputs for each.

scripts/regen_rect_cases.sh
Regenerates forward solves across a grid of padding and mesh density (controlled via environment variables). Drops results under Results/cases/<tag>/phi_*.xdmf and prints per-run diagnostics.

scripts/verify_rect_cases.sh
Scans Results/cases/** for phi_*.xdmf and runs the comparator on every case found. Good for large sweeps.

rect_gate_benchmark.py
The core driver for the forward solve: constructs the box geometry, tags gate pads, applies BCs (Dirichlet on pads and bottom; natural on sides/top outside pads), assembles -∇·(ε∇φ)=0 (Laplace), solves for ϕ, and writes Results/phi_p?a.xdmf.

exact_rect_gates.py
Utilities for reference potentials; shared logic/backups for the analytic evaluator.

Results/
All outputs: .xdmf/.h5 field files and .csv error summaries. Subfolder Results/verify/ contains verification products; Results/cases/ is used by sweeps.

verification_notes.tex
A LaTeX write-up explaining the benchmark: geometry, BCs, the analytic reference, error metrics, and how padding/mesh affect accuracy.

Requirements

Docker (recommended): uses dolfinx/dolfinx:stable.
No system fenicsx install needed.

ParaView (for visualization).

Quick start

Run (or re-run) the forward solves (optional; you can skip if Results/phi_p?a.xdmf already exist):

bash scripts/regen_rect_cases.sh


You can control the sweep via environment variables:

NX_BASE=120 NY_BASE=120 NZ_BASE=64 \
PADS="2 3 4 5" \
SCALES="0.75 1.0 1.5" \
A=70e-9 VP1=0.25 VX=0.10 VP2=0.25 DEGREE=1 \
bash scripts/regen_rect_cases.sh


Verify a single canonical case (p=2a):

./scripts/verify_rect_p2a.sh


Verify the standard set (p = 2a,3a,4a,5a):

./scripts/verify_rect_all.sh


Verify everything under a sweep (Results/cases/**):

./scripts/verify_rect_cases.sh

Outputs & ParaView

Each verification step writes:

Nodal error (POINT data):

Results/verify/<stem>_rel_error_nodal.xdmf


Array name: rel_error_dof — relative error at degrees of freedom (max/L² shown in CSV).

Cellwise error (CELL data):

Results/verify/<stem>_rel_error_cellwise.xdmf


Array name: rel_error_cell — DG0 projection; summarizes error per element (max/mean shown in CSV).

Summary CSV:

Results/verify/<stem>_summary.csv


With REL_DOF_Linf, REL_DOF_L2, REL_CELL_max, REL_CELL_mean.

Open in ParaView:

File → Open → pick either *_rel_error_nodal.xdmf or *_rel_error_cellwise.xdmf.

In the Coloring dropdown, choose rel_error_dof (nodal) or rel_error_cell (cellwise).

Click Rescale to Data Range. Use Log Color Scale if the range is wide.

Optionally add Slice/Clip filters to inspect interior planes; use Probe Location for point values.

Concepts & expected behavior

Padding 
𝑝
p controls how wide the box is relative to the gate half-width 
𝑎
a. Larger 
𝑝
p pushes the side boundaries farther away, reducing boundary-interaction error. Expect errors to decrease with larger 
𝑝
p (until mesh error dominates).

Mesh density controls discretization error. Finer meshes generally reduce the nodal and cellwise errors.

Nodal vs cellwise:

Nodal (POINT) focuses on pointwise relative error at DOFs — sensitive to local oscillations.

Cellwise (CELL) averages (DG0), giving a smoother, per-element view of local error magnitude.

Troubleshooting tips

If ParaView loads an XDMF but shows nothing, check:

You opened the right file (nodal vs cellwise).

The array is selected in Coloring (should be rel_error_dof or rel_error_cell).

Rescale to Data Range is applied.

If a verification run stops after “loaded mesh+phi_h”, ensure:

meshio and h5py install inside the container (scripts do this automatically).

PYTHONPATH includes the repo (scripts export it).

The field name is phi (default) or pass --field-name accordingly.

Extending to Poisson (ρ ≠ 0)

The forward form is already set up for -∇·(ε∇φ) = ρ. Add a source term and (if needed) modify the analytic reference or switch to MMS (manufactured solution) for verification. The verification pipeline (XDMF load → error fields → CSV) remains the same.
