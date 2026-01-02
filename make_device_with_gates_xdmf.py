#!/usr/bin/env python3
import json
import argparse
from mpi4py import MPI

from dolfinx.io import XDMFFile

# NOTE: your dolfinx image doesn't have dolfinx.io.gmshio, but it DOES have dolfinx.io.gmsh
from dolfinx.io import gmsh as dfx_gmsh

import gmsh


def add_outer_surface_tags(Lx: float, Ly: float, H: float, eps: float = 1e-12):
    # bottom plane z=H  -> tag 10
    bottom = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, -Ly/2 - eps, H - eps,
        +Lx/2 + eps, +Ly/2 + eps, H + eps, 2
    )
    bottom_ids = [s[1] for s in bottom]
    if bottom_ids:
        gmsh.model.addPhysicalGroup(2, bottom_ids, 10)
        gmsh.model.setPhysicalName(2, 10, "bottom_zH")

    # top plane z=0 -> tag 11
    top = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, -Ly/2 - eps, 0.0 - eps,
        +Lx/2 + eps, +Ly/2 + eps, 0.0 + eps, 2
    )
    top_ids = [s[1] for s in top]
    if top_ids:
        gmsh.model.addPhysicalGroup(2, top_ids, 11)
        gmsh.model.setPhysicalName(2, 11, "top_z0")

    # sides -> tag 12
    side_ids = set()

    x_minus = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, -Ly/2 - eps, 0.0 - eps,
        -Lx/2 + eps, +Ly/2 + eps, H + eps, 2
    )
    side_ids.update([s[1] for s in x_minus])

    x_plus = gmsh.model.getEntitiesInBoundingBox(
        +Lx/2 - eps, -Ly/2 - eps, 0.0 - eps,
        +Lx/2 + eps, +Ly/2 + eps, H + eps, 2
    )
    side_ids.update([s[1] for s in x_plus])

    y_minus = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, -Ly/2 - eps, 0.0 - eps,
        +Lx/2 + eps, -Ly/2 + eps, H + eps, 2
    )
    side_ids.update([s[1] for s in y_minus])

    y_plus = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, +Ly/2 - eps, 0.0 - eps,
        +Lx/2 + eps, +Ly/2 + eps, H + eps, 2
    )
    side_ids.update([s[1] for s in y_plus])

    side_ids = list(side_ids)
    if side_ids:
        gmsh.model.addPhysicalGroup(2, side_ids, 12)
        gmsh.model.setPhysicalName(2, 12, "sides_outer")


def build_model(Lx, Ly, H, gate_boxes, lc_min, lc_max):
    gmsh.initialize()
    gmsh.model.add("device_with_embedded_gates")

    # Device box: x,y in [-L/2, L/2], z in [0,H]
    dev = gmsh.model.occ.addBox(-Lx/2, -Ly/2, 0.0, Lx, Ly, H)

    tools = []
    for g in gate_boxes:
        tools.append(
            gmsh.model.occ.addBox(
                float(g["x0"]), float(g["y0"]), float(g["z0"]),
                float(g["dx"]), float(g["dy"]), float(g["dz"])
            )
        )

    out_dimtags, _ = gmsh.model.occ.cut(
        [(3, dev)],
        [(3, t) for t in tools],
        removeObject=True,
        removeTool=True
    )
    gmsh.model.occ.synchronize()

    # Dielectric volume(s) -> tag 1
    vols = [tag for (dim, tag) in out_dimtags if dim == 3]
    if not vols:
        raise RuntimeError("Boolean cut produced no 3D volumes.")
    gmsh.model.addPhysicalGroup(3, vols, 1)
    gmsh.model.setPhysicalName(3, 1, "dielectric")

    # Outer boundary tags
    add_outer_surface_tags(Lx, Ly, H, eps=1e-12)

    # Gate cavity surfaces: physical tags = 101, 102, 103, ...
    eps = 1e-12
    for g in gate_boxes:
        tag = int(g["tag"])
        x0, y0, z0 = float(g["x0"]), float(g["y0"]), float(g["z0"])
        dx, dy, dz = float(g["dx"]), float(g["dy"]), float(g["dz"])

        surfs = gmsh.model.getEntitiesInBoundingBox(
            x0 - eps, y0 - eps, z0 - eps,
            x0 + dx + eps, y0 + dy + eps, z0 + dz + eps, 2
        )
        surf_ids = [s[1] for s in surfs]
        if not surf_ids:
            raise RuntimeError(f"No cavity surfaces found for gate tag {tag}. Check gate box coords.")
        gmsh.model.addPhysicalGroup(2, surf_ids, tag)
        gmsh.model.setPhysicalName(2, tag, f"gate_cavity_{tag}")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(lc_min))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(lc_max))

    gmsh.model.mesh.generate(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output XDMF base name (no extension).")
    p.add_argument("--Lx", type=float, required=True)
    p.add_argument("--Ly", type=float, required=True)
    p.add_argument("--H",  type=float, required=True)
    p.add_argument("--gates", type=str, required=True, help="JSON list of gate void boxes with tags.")
    p.add_argument("--lc_min", type=float, default=5e-9)
    p.add_argument("--lc_max", type=float, default=20e-9)
    args = p.parse_args()

    gate_boxes = json.loads(args.gates)

    build_model(args.Lx, args.Ly, args.H, gate_boxes, args.lc_min, args.lc_max)

    # Convert gmsh.model -> dolfinx mesh + tags (this is the key step)
    comm = MPI.COMM_WORLD
    domain, cell_tags, facet_tags = dfx_gmsh.model_to_mesh(gmsh.model, comm, 0, gdim=3)

    # Write mesh + cell tags
    with XDMFFile(comm, f"{args.out}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        if cell_tags is not None:
            xf.write_meshtags(cell_tags, domain.geometry)

    # Write facet tags (debug + for BCs)
    if facet_tags is not None:
        with XDMFFile(comm, f"{args.out}_facets.xdmf", "w") as xf:
            xf.write_mesh(domain)
            xf.write_meshtags(facet_tags, domain.geometry)

    gmsh.finalize()

    if comm.rank == 0:
        print(f"[OK] wrote {args.out}.xdmf and {args.out}_facets.xdmf")


if __name__ == "__main__":
    main()
