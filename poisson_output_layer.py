#!/usr/bin/env python3
"""
Canonical output layer for DOLFINx electrostatics runs.

What this module does
- Writes one canonical fields file:
    <basename>_fields.xdmf
    <basename>_fields.h5
  containing:
    - mesh
    - phi
    - eps
    - rho

- Optionally writes:
    <basename>_cell_tags.xdmf
    <basename>_facet_tags.xdmf

- Always writes:
    <basename>_config.json
    <basename>_summary.txt
    <basename>_summary.json
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
from typing import Any, Optional

import numpy as np
from mpi4py import MPI

from dolfinx import fem, io


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _global_min_max(comm: MPI.Intracomm, values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        local_min = np.inf
        local_max = -np.inf
    else:
        local_min = float(np.min(values))
        local_max = float(np.max(values))

    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    return gmin, gmax


def function_global_min_max(fn: fem.Function) -> tuple[float, float]:
    return _global_min_max(fn.function_space.mesh.comm, fn.x.array)


def interpolate_to_cg1(fn: fem.Function, name: Optional[str] = None) -> fem.Function:
    """
    Robust version: always interpolate into scalar CG1.
    This avoids DOLFINx/Basix API-version checks.
    """
    mesh_obj = fn.function_space.mesh
    V_out = fem.functionspace(mesh_obj, ("Lagrange", 1))
    out = fem.Function(V_out, name=name or fn.name or "phi")
    out.interpolate(fn)
    out.x.scatter_forward()
    return out


def ensure_dg0_field(fn_or_expr: Any, mesh_obj, name: str) -> fem.Function:
    """
    Convert input into a scalar DG0 field on mesh_obj.

    Supported:
    - fem.Function
    - scalar constant
    - interpolatable expression/callable
    """
    V0 = fem.functionspace(mesh_obj, ("DG", 0))
    out = fem.Function(V0, name=name)

    if isinstance(fn_or_expr, fem.Function):
        out.interpolate(fn_or_expr)
        out.x.scatter_forward()
        return out

    if np.isscalar(fn_or_expr):
        out.x.array[:] = float(fn_or_expr)
        out.x.scatter_forward()
        return out

    out.interpolate(fn_or_expr)
    out.x.scatter_forward()
    return out


class XDMFRunWriter:
    def __init__(self, outdir: str | Path, basename: str):
        self.outdir = Path(outdir)
        self.basename = basename
        _ensure_dir(self.outdir)

    @property
    def fields_xdmf(self) -> Path:
        return self.outdir / f"{self.basename}_fields.xdmf"

    @property
    def cell_tags_xdmf(self) -> Path:
        return self.outdir / f"{self.basename}_cell_tags.xdmf"

    @property
    def facet_tags_xdmf(self) -> Path:
        return self.outdir / f"{self.basename}_facet_tags.xdmf"

    @property
    def config_json(self) -> Path:
        return self.outdir / f"{self.basename}_config.json"

    @property
    def summary_txt(self) -> Path:
        return self.outdir / f"{self.basename}_summary.txt"

    @property
    def summary_json(self) -> Path:
        return self.outdir / f"{self.basename}_summary.json"

    def write_fields(
        self,
        phi: fem.Function,
        eps: Any,
        rho: Any,
        *,
        force_phi_cg1: bool = True,
    ) -> dict[str, Any]:
        mesh_obj = phi.function_space.mesh

        phi_out = interpolate_to_cg1(phi, name="phi") if force_phi_cg1 else phi
        eps_out = ensure_dg0_field(eps, mesh_obj, name="eps")
        rho_out = ensure_dg0_field(rho, mesh_obj, name="rho")

        with io.XDMFFile(mesh_obj.comm, str(self.fields_xdmf), "w") as xdmf:
            xdmf.write_mesh(mesh_obj)
            xdmf.write_function(phi_out)
            xdmf.write_function(eps_out)
            xdmf.write_function(rho_out)

        return {"phi": phi_out, "eps": eps_out, "rho": rho_out}

    def write_meshtags(
        self,
        mesh_obj,
        *,
        cell_tags=None,
        facet_tags=None,
    ) -> None:
        if cell_tags is not None:
            with io.XDMFFile(mesh_obj.comm, str(self.cell_tags_xdmf), "w") as xdmf:
                xdmf.write_mesh(mesh_obj)
                xdmf.write_meshtags(cell_tags, mesh_obj.geometry)

        if facet_tags is not None:
            with io.XDMFFile(mesh_obj.comm, str(self.facet_tags_xdmf), "w") as xdmf:
                xdmf.write_mesh(mesh_obj)
                xdmf.write_meshtags(facet_tags, mesh_obj.geometry)

    def write_config(self, cfg: dict[str, Any]) -> None:
        data = _to_serializable(cfg)
        if MPI.COMM_WORLD.rank == 0:
            with open(self.config_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)

    def build_summary(
        self,
        *,
        phi: fem.Function,
        eps: fem.Function,
        rho: fem.Function,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        phi_min, phi_max = function_global_min_max(phi)
        eps_min, eps_max = function_global_min_max(eps)
        rho_min, rho_max = function_global_min_max(rho)

        summary = {
            "phi_min": phi_min,
            "phi_max": phi_max,
            "eps_min": eps_min,
            "eps_max": eps_max,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "fields_xdmf": str(self.fields_xdmf),
        }
        if self.cell_tags_xdmf.exists():
            summary["cell_tags_xdmf"] = str(self.cell_tags_xdmf)
        if self.facet_tags_xdmf.exists():
            summary["facet_tags_xdmf"] = str(self.facet_tags_xdmf)
        if extra:
            summary.update(_to_serializable(extra))
        return summary

    def write_summary(
        self,
        *,
        phi: fem.Function,
        eps: fem.Function,
        rho: fem.Function,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        mesh_obj = phi.function_space.mesh
        summary = self.build_summary(phi=phi, eps=eps, rho=rho, extra=extra)

        if mesh_obj.comm.rank == 0:
            with open(self.summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            lines = [
                f"basename: {self.basename}",
                f"fields_xdmf: {summary['fields_xdmf']}",
                f"phi_min: {summary['phi_min']:.12e}",
                f"phi_max: {summary['phi_max']:.12e}",
                f"eps_min: {summary['eps_min']:.12e}",
                f"eps_max: {summary['eps_max']:.12e}",
                f"rho_min: {summary['rho_min']:.12e}",
                f"rho_max: {summary['rho_max']:.12e}",
            ]
            if "cell_tags_xdmf" in summary:
                lines.append(f"cell_tags_xdmf: {summary['cell_tags_xdmf']}")
            if "facet_tags_xdmf" in summary:
                lines.append(f"facet_tags_xdmf: {summary['facet_tags_xdmf']}")

            for key, value in summary.items():
                if key in {
                    "fields_xdmf",
                    "phi_min",
                    "phi_max",
                    "eps_min",
                    "eps_max",
                    "rho_min",
                    "rho_max",
                    "cell_tags_xdmf",
                    "facet_tags_xdmf",
                }:
                    continue
                lines.append(f"{key}: {value}")

            with open(self.summary_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        return summary


def print_written_files(writer: XDMFRunWriter, comm: MPI.Intracomm) -> None:
    if comm.rank == 0:
        print("\n=== Wrote files ===")
        print(f"  fields:        {writer.fields_xdmf}")
        if writer.cell_tags_xdmf.exists():
            print(f"  cell tags:     {writer.cell_tags_xdmf}")
        if writer.facet_tags_xdmf.exists():
            print(f"  facet tags:    {writer.facet_tags_xdmf}")
        print(f"  config:        {writer.config_json}")
        print(f"  summary txt:   {writer.summary_txt}")
        print(f"  summary json:  {writer.summary_json}")
        print("\nParaView tips:")
        print("  - Open *_fields.xdmf to inspect phi, eps, and rho together.")
        print("  - Open *_facet_tags.xdmf to inspect boundary tag IDs.")
        print("  - Open *_cell_tags.xdmf to inspect region IDs.")
