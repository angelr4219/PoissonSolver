# poisson_solver

A modular DOLFINx / FEniCSx electrostatics solver package for 3D Poisson and Laplace problems.

This package is being refactored toward a single main entry point that can run multiple simulation cases by changing configuration, instead of maintaining many one-off scripts.

At the current stage, the code supports:

- structured box-based Laplace benchmark cases
- gmsh-based 3D top-disk electrode cases
- gmsh-based charged disk-in-box Poisson cases
- optional layered dielectric fields by z-range
- one or more charged disks in the charged-disk workflow
- XDMF output for mesh, potential, dielectric field, charge field, and region IDs
- safe runtime behavior for this Docker/DOLFINx environment using `--hard_exit`

The code is intended to move toward a general workflow where you choose:

- geometry
- materials / dielectric layers
- charges
- boundary conditions
- refinement settings

and then run from one `main.py` interface.

---

# Repository structure

The important files inside `poisson_solver/` are:

- `common.py`
  - shared MPI handles and basic utilities
- `solver.py`
  - shared scalar Poisson / Laplace solve path
- `materials.py`
  - dielectric field builders, layer builders, disk charge field builders
- `mesh_builders.py`
  - gmsh and box mesh construction helpers
- `output.py`
  - XDMF writing helpers
- `analytic.py`
  - probe evaluation and analytic benchmark helpers
- `mpc.py`
  - periodic MPC scaffold for future boundary-condition extensions
- `cases/`
  - case-specific runners

Current active case runners:

- `square_gate_stack.py`
- `topdisk3d.py`
- `charged_disk_box.py`
- `rod_bore.py`

Top-level entry point:

- `main.py`

---

# Main idea

Run everything from:

```bash
python3 main.py <case_name> [options]
n practice, for this project, use Docker:

./run_in_docker.sh python3 main.py <case_name> [options]

For heavier runs, MPI can be used through a wrapper like:

NPROC=4 ./run_in_docker_mpi.sh python3 main.py <case_name> [options]
Current capabilities
1. Square gate stack benchmark

Case name:

square_gate_stack

What it does:

builds a structured 3D box mesh

places square Dirichlet gate patches on the top surface

solves Laplace equation with optional z-dependent dielectric

writes potential and dielectric field to XDMF

can evaluate probe lines

can compare against the Anderson-style analytic square-gate reference when appropriate

supports a scaffold for periodic boundary conditions later

Good for:

benchmark verification

line-probe comparison

h/p refinement checks

controlled gate-layout electrostatics

2. Top disk 3D case

Case name:

topdisk3d

What it does:

builds a gmsh 3D box

places a circular Dirichlet electrode on the top surface

optionally splits the volume into two dielectric regions

tags cell and facet regions

solves Laplace equation

writes mesh, phi, epsilon, and tags

Good for:

verifying gmsh geometry tagging

testing facet BC assignment

two-region dielectric split checks

simple top-surface gate/electrode setups

3. Charged disk box case

Case name:

charged_disk_box

What it does:

builds a 3D gmsh box with mesh refinement around a reference disk region

creates one or more charged disk volumes through DG0 cell marking

builds dielectric field either as uniform bulk or layered by z

solves Poisson equation with volumetric charge

writes pre-solve and solved fields

Good for:

charged disk comparison studies

slice-by-slice field comparison

testing mesh refinement around localized charged regions

layered dielectric visualization

future extension to multiple dots/charges

4. Rod with bore case

Case name:

rod_bore

What it does:

builds a prism with a cylindrical bore

assigns different dielectric regions

solves Poisson with a Gaussian charge source

Good for:

dielectric region tagging tests

nontrivial subdomain validation

future heterogeneous material checks

Runtime notes
Docker

This code is expected to run in Docker with DOLFINx available.

Typical pattern:

./run_in_docker.sh python3 main.py <case> ...

If your host Python says ModuleNotFoundError: dolfinx, that is expected outside the container.

Safe exit

Some DOLFINx / PETSc builds in this environment can segfault during teardown even after a successful solve.

Because of that, most heavy runs should use:

--hard_exit

This forces an MPI barrier and then exits cleanly with os._exit(0).

KSP diagnostics

For this environment, do not rely on PETSc solver-handle introspection after solve unless explicitly re-verified. The current stable pattern avoids touching KSP iteration/residual details because that caused crashes in this build.

How to run the code
A. Square gate benchmark, Dirichlet sides
./run_in_docker.sh python3 main.py square_gate_stack \
  --outdir results/realistic_gates \
  --basename gate_stack_dirichlet \
  --Lx 300e-9 --Ly 300e-9 --H 200e-9 \
  --tox 0.0 \
  --a 35e-9 \
  --gate_xs="-70e-9,0.0,70e-9" \
  --gate_ys="0.0,0.0,0.0" \
  --gate_Vs="0.25,0.10,0.25" \
  --h 10e-9 --deg 1 \
  --eps_ox 3.9 --eps_si 11.7 \
  --boundary_mode dirichlet \
  --bc_sides dirichlet0 --bc_bottom dirichlet0 \
  --probe_line 1 --probe_n 401 --probe_y 0.0 --probe_z 35e-9 \
  --hard_exit
B. Top disk uniform dielectric
./run_in_docker.sh python3 main.py topdisk3d \
  --outdir results/topdisk_uniform \
  --basename topdisk_uniform \
  --Lx 300 --Ly 300 --z_top 0 --Lz 200 \
  --h 10 --deg 1 --scale 1e-9 \
  --disk_xc 0 --disk_yc 0 --disk_R 50 --Vdisk 1.0 \
  --eps_r0 11.7 \
  --bc_top dirichlet0 --bc_bottom dirichlet0 --bc_sides dirichlet0 \
  --hard_exit
C. Top disk with two dielectric regions
./run_in_docker.sh python3 main.py topdisk3d \
  --outdir results/topdisk_twoeps \
  --basename topdisk_twoeps \
  --Lx 300 --Ly 300 --z_top 0 --Lz 200 \
  --h 10 --deg 1 --scale 1e-9 \
  --disk_xc 0 --disk_yc 0 --disk_R 50 --Vdisk 1.0 \
  --eps_r0 3.9 --eps_r1 11.7 --split_z 10 \
  --bc_top dirichlet0 --bc_bottom dirichlet0 --bc_sides dirichlet0 \
  --hard_exit
D. Charged 10 nm disk in explicit box bounds

This is the charged disk workflow with a 10 nm radius disk inside:

x in [-250, 250] nm

y in [-250, 250] nm

z in [0, 260] nm

Disk centered at z = 155 nm:

./run_in_docker.sh python3 main.py charged_disk_box \
  --disk="0.0,0.0,155e-9,10e-9,4e-9,1.602176634e-19" \
  --out out_mesh_fields/disk_R10nm_z155nm_box500x500x260 \
  --p 1 \
  --epsr 12.0 \
  --Lx 500e-9 \
  --Ly 500e-9 \
  --Lz 260e-9 \
  --xmin=-250e-9 \
  --xmax=250e-9 \
  --ymin=-250e-9 \
  --ymax=250e-9 \
  --zmin=0.0 \
  --zmax=260e-9 \
  --n_diam 24 \
  --refine_band_R 6.0 \
  --far_h_factor 8.0 \
  --hard_exit
E. Charged 10 nm disk with visible dielectric layers
./run_in_docker.sh python3 main.py charged_disk_box \
  --disk="0.0,0.0,155e-9,10e-9,4e-9,1.602176634e-19" \
  --out out_mesh_fields/disk_R10nm_z155nm_layered \
  --p 1 \
  --epsr 12.0 \
  --Lx 500e-9 \
  --Ly 500e-9 \
  --Lz 260e-9 \
  --xmin=-250e-9 \
  --xmax=250e-9 \
  --ymin=-250e-9 \
  --ymax=250e-9 \
  --zmin=0.0 \
  --zmax=260e-9 \
  --n_diam 24 \
  --refine_band_R 6.0 \
  --far_h_factor 8.0 \
  --layer="0.0,10e-9,3.9" \
  --layer="10e-9,60e-9,12.0" \
  --layer="60e-9,140e-9,11.7" \
  --layer="140e-9,260e-9,12.0" \
  --hard_exit
F. Multiple disks
./run_in_docker.sh python3 main.py charged_disk_box \
  --disk="0.0,0.0,155e-9,10e-9,4e-9,1.602176634e-19" \
  --disk="40e-9,0.0,155e-9,10e-9,4e-9,1.602176634e-19" \
  --out out_mesh_fields/two_disk_test \
  --p 1 \
  --epsr 12.0 \
  --Lx 500e-9 \
  --Ly 500e-9 \
  --Lz 260e-9 \
  --xmin=-250e-9 \
  --xmax=250e-9 \
  --ymin=-250e-9 \
  --ymax=250e-9 \
  --zmin=0.0 \
  --zmax=260e-9 \
  --n_diam 24 \
  --refine_band_R 6.0 \
  --far_h_factor 8.0 \
  --hard_exit
Important CLI notes
Negative numbers

For arguments like bounds, use = when the value is negative:

Good:

--xmin=-250e-9

Bad:

--xmin -250e-9

The bad form can confuse argparse.

Repeated layers

Use repeated --layer entries:

--layer="0.0,10e-9,3.9" \
--layer="10e-9,60e-9,12.0"

Format is:

zmin,zmax,epsr
Repeated disks

Use repeated --disk entries:

--disk="xc,yc,zc,R,t,Q"

Example:

--disk="0.0,0.0,155e-9,10e-9,4e-9,1.602176634e-19"

Format is:

xc,yc,zc,R,t,Q

where:

xc, yc, zc are center coordinates

R is radius

t is thickness

Q is total charge on that disk

Resolution controls

For charged-disk runs, the key refinement knobs are:

--n_diam

target number of elements across disk diameter

--refine_band_R

how far the local refinement extends around the disk, measured in disk radii

--far_h_factor

how coarse the far field can become relative to the local disk mesh size

--p

polynomial degree of the finite element solve

Practical guidance:

start with p=1

keep n_diam moderate first

only increase p or n_diam after confirming the run finishes and the field is usable

very large n_diam on a tiny disk in a large 3D box can become extremely expensive

Output files

Typical outputs include:

pre_fields.xdmf

mesh + pre-solve fields like rho, epsilon, region_id

mesh_fields.xdmf

mesh + solved phi and associated DG0 fields

case-specific .xdmf files for benchmark or tag exports

.h5 sidecar files written automatically by XDMF

Use pre_fields.xdmf to verify:

the disk is placed where you think it is

dielectric layers occupy the expected z-ranges

Use solved output to:

compare slices

probe line cuts

validate against reference data

Current limitations

At the moment:

not every geometry family is fully unified under one generic config object yet

periodic boundary conditions are scaffolded, not fully generalized across all cases

the charged disk mesh refinement currently uses one reference disk for the gmsh refinement field

some builds require --hard_exit due to PETSc teardown behavior

gmsh-based tags and combined XDMF presentation can vary slightly by DOLFINx build

Recommended workflow right now
For verification

run a moderate case first

inspect pre_fields.xdmf

inspect solved output

compare slices or line probes against reference files

then increase resolution

For charged disk comparisons

place the disk at the exact physical z you want to compare

use explicit box bounds that match the comparison domain

keep p=1 first

increase mesh density only after the moderate case is usable

For future extension

Use the current code as a base for:

multiple disks / dots

multi-layer dielectric stacks

named stack presets

CIF or imported geometry pathways

periodic boundary condition integration

Planned extensions

The code is already moving toward the following:

one main interface for all runs

config-driven geometry/material/charge definitions

repeated --disk support for many charged regions

repeated --layer support for arbitrary dielectric stacks

MPI run wrappers for heavier cases

lazy imports in main.py

shared config objects across all solver families

future periodic boundary condition support through the MPC scaffold

Short version

If you are not sure what to run:

use square_gate_stack for clean benchmark electrostatics

use topdisk3d for gmsh facet-tag / Dirichlet patch tests

use charged_disk_box for charged-disk slice comparisons and layered dielectric tests

use --hard_exit for reliable completion in this environment
