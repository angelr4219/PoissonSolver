from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dolfinx import fem, io, mesh as dmesh

from .common import COMM, RANK, degree_compat, make_function_space


def _write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


def interpolate_to_cg1_if_needed(fn: fem.Function) -> fem.Function:
    msh = fn.function_space.mesh

    try:
        deg = int(fn.function_space.ufl_element().degree())
    except Exception:
        try:
            deg = int(fn.function_space.ufl_element().degree)
        except Exception:
            deg = 1

    if deg == 1:
        return fn

    V1 = make_function_space(msh, ("CG", 1))
    fn_out = fem.Function(V1, name=fn.name)

    try:
        fn_out.interpolate(fn)
    except Exception:
        expr = fem.Expression(fn, V1.element.interpolation_points())
        fn_out.interpolate(expr)

    return fn_out


def write_phi_file(
    msh: dmesh.Mesh,
    phi: fem.Function,
    outdir: str | Path,
    basename: str = "phi_solution",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xdmf_phi = outdir / f"{basename}.xdmf"

    phi_out = interpolate_to_cg1_if_needed(phi)
    if RANK == 0 and phi_out is not phi:
        try:
            sol_deg = int(phi.function_space.ufl_element().degree())
        except Exception:
            try:
                sol_deg = int(phi.function_space.ufl_element().degree)
            except Exception:
                sol_deg = -1
        print(f"[INFO] Writing interpolated CG1 phi for XDMF compatibility (input degree was {sol_deg}).")

    with io.XDMFFile(COMM, str(xdmf_phi), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi_out)

    return xdmf_phi


def write_cell_fields_file(
    msh: dmesh.Mesh,
    outdir: str | Path,
    fields: Iterable[fem.Function],
    basename: str = "cell_fields",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xdmf_aux = outdir / f"{basename}.xdmf"

    with io.XDMFFile(COMM, str(xdmf_aux), "w") as xdmf:
        xdmf.write_mesh(msh)
        for fn in fields:
            xdmf.write_function(fn)

    return xdmf_aux


def write_meshtags_file(
    msh: dmesh.Mesh,
    mt: dmesh.MeshTags,
    outdir: str | Path,
    basename: str = "mesh_tags",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xdmf_tags = outdir / f"{basename}.xdmf"

    with io.XDMFFile(COMM, str(xdmf_tags), "w") as xdmf:
        xdmf.write_mesh(msh)
        _write_meshtags_compat(xdmf, mt, msh)

    return xdmf_tags


def print_written_files(*paths) -> None:
    if RANK == 0:
        print("\n=== Wrote files ===")
        for p in paths:
            if p is not None:
                print(f"  {p}")
        print("\nParaView tips:")
        print("  - Open the phi_solution.xdmf file for the scalar potential.")
        print("  - Open the cell_fields.xdmf file for rho / epsilon / region_id / shape_mask.")
