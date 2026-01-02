#!/usr/bin/env python3
"""
Build a dielectric box with embedded rectangular voids ("gate cavities") using gmsh OCC boolean cut.

- Device box: x,y in [-L/2, L/2], z in [0, H]
- Gate void boxes are CUT OUT of the dielectric volume.

Physical groups created:
  Volume:
    1    -> dielectric volume(s)

  Outer boundary surfaces (optional but recommended):
    10   -> bottom outer surface (z=H)
    11   -> top outer surface    (z=0)
    12   -> side outer surfaces  (x=±L/2 and y=±Ly/2)

  Gate cavity surfaces:
    101, 102, ... as provided by gate JSON "tag"

Notes:
- Bounding-box surface selection assumes voids do NOT touch the outer boundary planes.
"""

import json
import gmsh


def _add_outer_surface_tags(Lx: float, Ly: float, H: float, eps: float = 1e-12):
    """Tag OUTER boundary surfaces by bounding-box selection."""
    # bottom plane z=H
    bottom = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, -Ly/2 - eps, H - eps,
        +Lx/2 + eps, +Ly/2 + eps, H + eps, 2
    )
    bottom_ids = [s[1] for s in bottom]
    if bottom_ids:
        gmsh.model.addPhysicalGroup(2, bottom_ids, 10)
        gmsh.model.setPhysicalName(2, 10, "bottom_zH")

    # top plane z=0
    top = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, -Ly/2 - eps, 0.0 - eps,
        +Lx/2 + eps, +Ly/2 + eps, 0.0 + eps, 2
    )
    top_ids = [s[1] for s in top]
    if top_ids:
        gmsh.model.addPhysicalGroup(2, top_ids, 11)
        gmsh.model.setPhysicalName(2, 11, "top_z0")

    # sides: x=±L/2 and y=±Ly/2
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


def build_device_with_gate_voids(msh_path: str, Lx: float, Ly: float, H: float,
                                 gate_boxes, lc_min: float, lc_max: float):
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

    # Tag dielectric volume(s)
    vols = [tag for (dim, tag) in out_dimtags if dim == 3]
    if not vols:
        raise RuntimeError("Boolean cut produced no 3D volumes (unexpected).")
    gmsh.model.addPhysicalGroup(3, vols, 1)
    gmsh.model.setPhysicalName(3, 1, "dielectric")

    # Tag outer boundary surfaces
    _add_outer_surface_tags(Lx, Ly, H, eps=1e-12)

    # Tag each gate cavity surface by bounding-box selection
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

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    gmsh.finalize()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--Lx", type=float, required=True)
    p.add_argument("--Ly", type=float, required=True)
    p.add_argument("--H",  type=float, required=True)
    p.add_argument("--gates", type=str, required=True,
                   help='JSON list of gate boxes with tags, e.g. \'[{"x0":..., "tag":101}, ...]\'')
    p.add_argument("--lc_min", type=float, default=5e-9)
    p.add_argument("--lc_max", type=float, default=20e-9)
    args = p.parse_args()

    gate_boxes = json.loads(args.gates)
    build_device_with_gate_voids(args.out, args.Lx, args.Ly, args.H,
                                 gate_boxes, args.lc_min, args.lc_max)
