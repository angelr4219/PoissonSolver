#!/usr/bin/env python3
import argparse, json
from geometry import (
    RodBoreParams, DiskParams, GateBoxParams,
    build_rod_with_bore, build_disk_with_hole,
    build_gate_box, load_gmsh_mesh,
    # you can add build_sphere, build_cone here
)
from physics import (
    assign_permittivity, gaussian_charge, constant_charge,
    dirichlet_bc_on_tag, solve_poisson
)
from mpi4py import MPI
from dolfinx import fem, io
import numpy as np
import ufl, os

def main():
    parser = argparse.ArgumentParser(description="General Poisson solver with arbitrary geometry")
    # Geometry selection
    parser.add_argument("--problem", choices=["rod", "disk", "gate", "mesh"], default="rod",
                        help="Predefined geometry (rod, disk, gate) or 'mesh' for custom .msh file")
    parser.add_argument("--mesh", type=str, help="Path to .msh file (used when --problem=mesh)")
    # Common geometry parameters
    parser.add_argument("--Lx", type=float, default=1.0)
    parser.add_argument("--Ly", type=float, default=1.0)
    parser.add_argument("--Lz", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.2, help="Used for rod/disk or sphere")
    # Charges and permittivity
    parser.add_argument("--charge_type", choices=["none","gaussian","constant"], default="gaussian")
    parser.add_argument("--total_charge", type=float, default=1e-12, help="Total Gaussian charge (C)")
    parser.add_argument("--rho0", type=float, default=0.0, help="Constant charge density (C/m^3)")
    parser.add_argument("--charge_center", type=str, default="0.5,0.5,0.5")
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--eps_map", type=str, default='{"101":1.0,"201":11.7,"202":13.0,"203":11.7}',
                        help="JSON mapping from cell tag to relative permittivity")
    # Solver parameters
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--output", type=str, default="results/solution")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    # Build geometry
    if args.problem == "mesh":
        if not args.mesh:
            raise ValueError("Specify --mesh=<path> when --problem=mesh")
        mesh, cell_tags, facet_tags = load_gmsh_mesh(args.mesh, dim=3, comm=comm)
    elif args.problem == "rod":
        params = RodBoreParams(Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, R=args.radius)
        mesh, cell_tags, facet_tags = build_rod_with_bore(params, comm=comm)
    elif args.problem == "disk":
        params = DiskParams(Lx=args.Lx, Ly=args.Ly, R=args.radius)
        mesh, cell_tags, facet_tags = build_disk_with_hole(params, comm=comm)
    elif args.problem == "gate":
        params = GateBoxParams(Lx=args.Lx, Ly=args.Ly, H=args.Lz)
        mesh, cell_tags, facet_tags = build_gate_box(params, comm=comm)
    else:
        raise NotImplementedError(args.problem)

    # Assign permittivity
    eps_map = {int(k): float(v) for k, v in json.loads(args.eps_map).items()}
    eps = assign_permittivity(mesh, cell_tags, eps_map)

    # Define charge density
    if args.charge_type == "gaussian":
        center = [float(x) for x in args.charge_center.split(",")]
        rho = gaussian_charge(mesh, args.total_charge, center, args.sigma)
    elif args.charge_type == "constant":
        rho = constant_charge(mesh, args.rho0)
    else:
        rho = constant_charge(mesh, 0.0)

    # Dirichlet BC on all external facets to zero (modify as needed)
    V_tmp = fem.FunctionSpace(mesh, ("CG", args.degree))
    bndry_facets = np.arange(0, facet_tags.indices.size, dtype=np.int32)
    # Example: apply 0 V on all facet tags; adapt if you want different voltages by tag
    bcs = []
    unique_tags = np.unique(facet_tags.values)
    for t in unique_tags:
        dofs = fem.locate_dofs_topological(V_tmp, mesh.topology.dim - 1,
                                           facet_tags.indices[facet_tags.values == t])
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V_tmp))

    # Solve
    V, phi = solve_poisson(mesh, eps, rho, bcs, degree=args.degree)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with io.XDMFFile(comm, args.output + ".xdmf", "w") as xdmf:
        mesh.name = "mesh"
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi)

if __name__ == "__main__":
    main()