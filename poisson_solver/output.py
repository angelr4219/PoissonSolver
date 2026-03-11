from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dolfinx import fem, io, mesh as dmesh

from .common import RANK, degree_compat, make_function_space


def should_write_outputs() -> bool:
    return os.environ.get("NOWRITE", "0") != "1"


def write_meshtags_compat(xdmf: io.XDMFFile, mt: dmesh.MeshTags, msh: dmesh.Mesh) -> None:
    try:
        xdmf.write_meshtags(mt)
    except TypeError:
        xdmf.write_meshtags(mt, msh.geometry)


def function_for_xdmf(phi: fem.Function, msh: dmesh.Mesh) -> fem.Function:
    mesh_deg = degree_compat(msh.geometry.cmap.degree)
    sol_deg = degree_compat(phi.function_space.ufl_element().degree)
    if sol_deg == mesh_deg:
        return phi

    Vout = make_function_space(msh, ("CG", mesh_deg))
    phi_out = fem.Function(Vout, name=phi.name)
    try:
        phi_out.interpolate(phi)
    except Exception:
        expr = fem.Expression(phi, Vout.element.interpolation_points())
        phi_out.interpolate(expr)
    return phi_out


def write_mesh_and_functions(comm, msh: dmesh.Mesh, path: str | Path, functions: Iterable[fem.Function]) -> Path | None:
    if not should_write_outputs():
        if RANK == 0:
            print("NOTE: NOWRITE=1, skipping output")
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with io.XDMFFile(comm, str(path), "w") as xdmf:
        xdmf.write_mesh(msh)
        for fn in functions:
            xdmf.write_function(fn)

    if RANK == 0:
        print(f"Wrote: {path}")
    return path


def write_mesh_and_meshtags(comm, msh: dmesh.Mesh, path: str | Path, mt: dmesh.MeshTags) -> Path | None:
    if not should_write_outputs():
        if RANK == 0:
            print("NOTE: NOWRITE=1, skipping output")
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with io.XDMFFile(comm, str(path), "w") as xdmf:
        xdmf.write_mesh(msh)
        write_meshtags_compat(xdmf, mt, msh)

    if RANK == 0:
        print(f"Wrote: {path}")
    return path


def write_pre_fields(comm, msh, outdir: str | Path, rho, epsilon, region_id) -> Path | None:
    if not should_write_outputs():
        if RANK == 0:
            print("NOTE: NOWRITE=1, skipping output")
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / "pre_fields.xdmf"

    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(rho)
        xdmf.write_function(epsilon)
        xdmf.write_function(region_id)

    if RANK == 0:
        print(f"Wrote: {xdmf_path} (and .h5 sidecar)  [pre-solve]")
    return xdmf_path
