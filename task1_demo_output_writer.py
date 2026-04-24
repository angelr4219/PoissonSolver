from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh

from poisson_solver.outputs import write_run_outputs


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Task 1 demo: canonical XDMF/H5 output writer for phi, eps, rho, and tags."
    )
    p.add_argument("--outdir", type=str, default="Results/task1_demo", help="Output directory")
    p.add_argument("--basename", type=str, default="task1_demo", help="Base name for all output files")

    p.add_argument("--Lx_nm", type=float, default=300.0, help="Domain size in x [nm]")
    p.add_argument("--Ly_nm", type=float, default=300.0, help="Domain size in y [nm]")
    p.add_argument("--Lz_nm", type=float, default=120.0, help="Domain size in z [nm]")
    p.add_argument("--tox_nm", type=float, default=15.0, help="Oxide thickness at top [nm]")

    p.add_argument("--nx", type=int, default=18, help="Cells in x")
    p.add_argument("--ny", type=int, default=18, help="Cells in y")
    p.add_argument("--nz", type=int, default=12, help="Cells in z")

    p.add_argument("--deg", type=int, default=2, help="Polynomial degree for demo phi")
    p.add_argument("--eps_semiconductor", type=float, default=11.7, help="Semiconductor relative permittivity")
    p.add_argument("--eps_oxide", type=float, default=9.0, help="Oxide relative permittivity")
    p.add_argument("--rho_semiconductor", type=float, default=0.0, help="Volumetric demo rho in semiconductor")
    p.add_argument("--rho_oxide", type=float, default=0.0, help="Volumetric demo rho in oxide")
    return p


def make_sorted_meshtags(msh, dim: int, tagged_entity_lists: list[tuple[np.ndarray, int]], name: str):
    """
    Build a meshtags object from a list of (entity_indices, tag_id).
    """
    all_entities = []
    all_values = []
    for entities, tag_id in tagged_entity_lists:
        if len(entities) == 0:
            continue
        all_entities.append(np.asarray(entities, dtype=np.int32))
        all_values.append(np.full(len(entities), tag_id, dtype=np.int32))

    if len(all_entities) == 0:
        raise RuntimeError(f"No tagged entities were provided for dim={dim}")

    entities = np.concatenate(all_entities)
    values = np.concatenate(all_values)

    order = np.argsort(entities)
    entities = entities[order]
    values = values[order]

    tags = mesh.meshtags(msh, dim, entities, values)
    tags.name = name
    return tags


def build_cell_tags(msh, Lz_m: float, tox_m: float):
    tdim = msh.topology.dim
    oxide_tag = 2
    semiconductor_tag = 1

    tol = 1.0e-15
    z_split = Lz_m - tox_m

    oxide_cells = mesh.locate_entities(msh, tdim, lambda x: x[2] >= z_split - tol)
    semiconductor_cells = mesh.locate_entities(msh, tdim, lambda x: x[2] < z_split - tol)

    cell_tags = make_sorted_meshtags(
        msh,
        tdim,
        [
            (semiconductor_cells, semiconductor_tag),
            (oxide_cells, oxide_tag),
        ],
        name="cell_tags",
    )

    volume_tag_map = {
        "semiconductor": semiconductor_tag,
        "oxide": oxide_tag,
    }
    return cell_tags, volume_tag_map


def build_facet_tags(msh, Lx_m: float, Ly_m: float, Lz_m: float):
    tdim = msh.topology.dim
    fdim = tdim - 1
    tol = 1.0e-15

    facet_tag_map = {
        "x_min": 11,
        "x_max": 12,
        "y_min": 21,
        "y_max": 22,
        "z_min": 31,
        "z_max": 32,
    }

    x_min = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0, atol=tol))
    x_max = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], Lx_m, atol=tol))
    y_min = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0, atol=tol))
    y_max = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], Ly_m, atol=tol))
    z_min = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], 0.0, atol=tol))
    z_max = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], Lz_m, atol=tol))

    facet_tags = make_sorted_meshtags(
        msh,
        fdim,
        [
            (x_min, facet_tag_map["x_min"]),
            (x_max, facet_tag_map["x_max"]),
            (y_min, facet_tag_map["y_min"]),
            (y_max, facet_tag_map["y_max"]),
            (z_min, facet_tag_map["z_min"]),
            (z_max, facet_tag_map["z_max"]),
        ],
        name="facet_tags",
    )

    return facet_tags, facet_tag_map


def build_piecewise_dg0_field(
    msh,
    cell_tags,
    volume_tag_map: dict[str, int],
    region_values: dict[str, float],
    name: str,
) -> fem.Function:
    V0 = fem.functionspace(msh, ("DG", 0))
    f = fem.Function(V0, name=name)
    f.x.array[:] = 0.0

    for region_name, tag_id in volume_tag_map.items():
        if region_name not in region_values:
            raise KeyError(f"Missing region value for {region_name}")
        cells = cell_tags.find(tag_id)
        f.x.array[cells] = region_values[region_name]

    return f


def main() -> None:
    args = build_argument_parser().parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank

    nm = 1.0e-9
    Lx_m = args.Lx_nm * nm
    Ly_m = args.Ly_nm * nm
    Lz_m = args.Lz_nm * nm
    tox_m = args.tox_nm * nm

    if tox_m <= 0.0 or tox_m >= Lz_m:
        raise ValueError("Require 0 < tox_nm < Lz_nm")

    msh = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx_m, Ly_m, Lz_m])],
        [args.nx, args.ny, args.nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    msh.name = "mesh"

    cell_tags, volume_tag_map = build_cell_tags(msh, Lz_m, tox_m)
    facet_tags, facet_tag_map = build_facet_tags(msh, Lx_m, Ly_m, Lz_m)

    V_phi = fem.functionspace(msh, ("Lagrange", args.deg))
    phi = fem.Function(V_phi, name="phi_native")

    # Deliberately use degree > 1 by default so the writer exercises CG1 output conversion.
    phi.interpolate(
        lambda x: (
            0.20 * (x[0] / Lx_m)
            - 0.15 * (x[1] / Ly_m)
            + 0.10 * (x[2] / Lz_m)
            + 0.05 * (x[0] / Lx_m) ** 2
            - 0.03 * (x[1] / Ly_m) * (x[2] / Lz_m)
        )
    )

    eps = build_piecewise_dg0_field(
        msh,
        cell_tags,
        volume_tag_map,
        {
            "semiconductor": args.eps_semiconductor,
            "oxide": args.eps_oxide,
        },
        name="eps",
    )

    rho = build_piecewise_dg0_field(
        msh,
        cell_tags,
        volume_tag_map,
        {
            "semiconductor": args.rho_semiconductor,
            "oxide": args.rho_oxide,
        },
        name="rho",
    )

    config = {
        "task": "task1_canonical_output_writer_demo",
        "domain_nm": {
            "Lx_nm": args.Lx_nm,
            "Ly_nm": args.Ly_nm,
            "Lz_nm": args.Lz_nm,
            "tox_nm": args.tox_nm,
        },
        "mesh": {
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "cell_type": "tetrahedron",
            "phi_degree_native": args.deg,
        },
        "materials": {
            "eps_semiconductor": args.eps_semiconductor,
            "eps_oxide": args.eps_oxide,
            "rho_semiconductor": args.rho_semiconductor,
            "rho_oxide": args.rho_oxide,
        },
        "volume_tag_map": volume_tag_map,
        "facet_tag_map": facet_tag_map,
        "output": {
            "outdir": args.outdir,
            "basename": args.basename,
            "phi_written_as_cg1": True,
        },
    }

    result = write_run_outputs(
        mesh=msh,
        phi=phi,
        eps=eps,
        rho=rho,
        outdir=Path(args.outdir),
        basename=args.basename,
        config=config,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        volume_tag_map=volume_tag_map,
        facet_tag_map=facet_tag_map,
        phi_as_cg1=True,
        extra_summary={
            "note": "Demo output writer run. Replace phi/eps/rho with real solver outputs in the next step."
        },
    )

    summary = result["summary"]
    phi_out = result["phi_out"]
    eps_out = result["eps_out"]
    rho_out = result["rho_out"]

    if rank == 0:
        print("\n=== Task 1 demo: canonical output writer ===")
        print(f"outdir   : {args.outdir}")
        print(f"basename : {args.basename}")
        print(f"mesh     : {summary['num_global_cells']} cells, {summary['num_global_vertices']} vertices")
        print("\nField ranges")
        print(f"  phi : [{summary['fields']['phi']['min']:.6e}, {summary['fields']['phi']['max']:.6e}]")
        print(f"  eps : [{summary['fields']['eps']['min']:.6e}, {summary['fields']['eps']['max']:.6e}]")
        print(f"  rho : [{summary['fields']['rho']['min']:.6e}, {summary['fields']['rho']['max']:.6e}]")
        print("\nTag counts")
        print(f"  cell tags : {summary.get('cell_tag_counts', {})}")
        print(f"  facet tags: {summary.get('facet_tag_counts', {})}")
        print("\n=== Wrote files ===")
        print(f"  fields      : {summary['files']['fields_xdmf']}")
        print(f"  cell tags   : {summary['files']['cell_tags_xdmf']}")
        print(f"  facet tags  : {summary['files']['facet_tags_xdmf']}")
        print(f"  config json : {summary['files']['config_json']}")
        print(f"  summary txt : {summary['files']['summary_txt']}")
        print(f"  summary json: {summary['files']['summary_json']}")
        print("\nParaView tips:")
        print("  - Open *_fields.xdmf to view phi, eps, and rho together.")
        print("  - Open *_cell_tags.xdmf and color by 'cell_tags' to inspect oxide/semiconductor regions.")
        print("  - Open *_facet_tags.xdmf and color by 'facet_tags' to inspect boundary tags.")
        print("  - This task writes phi in CG1 for consistent visualization output.")

    # Keep references alive to make intent obvious to future integration work.
    _ = (phi_out, eps_out, rho_out)


if __name__ == "__main__":
    main()
