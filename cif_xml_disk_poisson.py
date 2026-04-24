#!/usr/bin/env python3
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, io, mesh
from dolfinx.mesh import CellType, meshtags
from dolfinx.fem.petsc import LinearProblem


EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19


@dataclass
class DiskGeometry:
    Lx_nm: float
    Ly_nm: float
    disk_radius_nm: float
    disk_cx_nm_box: float
    disk_cy_nm_box: float

    @property
    def disk_cx_nm_phys(self) -> float:
        return self.disk_cx_nm_box - 0.5 * self.Lx_nm

    @property
    def disk_cy_nm_phys(self) -> float:
        return self.disk_cy_nm_box - 0.5 * self.Ly_nm


@dataclass
class LayerSpec:
    name: str
    material: str
    height_nm: float


def parse_cif_optional(cif_path: str) -> DiskGeometry:
    text = Path(cif_path).read_text()

    box_m = re.search(r"Box\s+([\d.+-Ee]+)\s+([\d.+-Ee]+)\s+([\d.+-Ee]+)\s+([\d.+-Ee]+)\s*;", text)
    disk_m = re.search(r"\bR\s+([\d.+-Ee]+)\s+([\d.+-Ee]+)\s+([\d.+-Ee]+)\s*;", text)

    if box_m is None:
        raise ValueError(f"Could not parse 'Box ...;' from {cif_path}")
    if disk_m is None:
        raise ValueError(f"Could not parse 'R radius cx cy;' from {cif_path}")

    Lx_nm = float(box_m.group(1))
    Ly_nm = float(box_m.group(2))
    x0_nm = float(box_m.group(3))
    y0_nm = float(box_m.group(4))
    if abs(x0_nm) > 1e-12 or abs(y0_nm) > 1e-12:
        raise NotImplementedError("This parser assumes Box ... 0 0; style CIFs.")

    return DiskGeometry(
        Lx_nm=Lx_nm,
        Ly_nm=Ly_nm,
        disk_radius_nm=float(disk_m.group(1)),
        disk_cx_nm_box=float(disk_m.group(2)),
        disk_cy_nm_box=float(disk_m.group(3)),
    )


def xml_get(root, path, default=None, cast=str, attr="value"):
    node = root.find(path)
    if node is None:
        return default
    val = node.get(attr)
    if val is None:
        return default
    if cast is bool:
        return str(val).strip().lower() == "true"
    return cast(val)


def parse_xml_optional(xml_path: str):
    root = ET.parse(xml_path).getroot()

    gate_bias = xml_get(root, "./GateBias/P1", None, float)
    bottom_bias = xml_get(root, "./LowerSurfaceBoundaryConditions/bias", None, float)
    bottom_bc_type = xml_get(root, "./LowerSurfaceBoundaryConditions/botBiasType", "DIRICHLET", str)
    interface_bc_flag = xml_get(root, "./InterfaceBCparameters/interfaceBCflag", False, bool)
    surface_src_density = xml_get(root, "./InterfaceBCparameters/surfaceSrcDensity", 0.0, float)
    cap_material = xml_get(root, "./InterfaceBCparameters/capMaterialName", "Vacuum", str)
    cap_gate_dist = xml_get(root, "./InterfaceBCparameters/capGateDist", 0.0, float)

    layers = []
    for layer in root.findall("./LayeredStructure/Layer"):
        layers.append(
            LayerSpec(
                name=layer.find("name").get("value"),
                material=layer.find("materialType").get("value"),
                height_nm=float(layer.find("height").get("value")),
            )
        )

    dielectric_map = {}
    for mat in root.findall("./MaterialList/Material"):
        dielectric_map[mat.find("name").get("value")] = float(mat.find("dielectricConstant").get("value"))

    return {
        "gate_bias": gate_bias,
        "bottom_bias": bottom_bias,
        "bottom_bc_type": bottom_bc_type,
        "interface_bc_flag": interface_bc_flag,
        "surface_src_density_cm2": surface_src_density,
        "cap_material": cap_material,
        "cap_gate_dist_nm": cap_gate_dist,
        "layers": layers,
        "dielectric_map": dielectric_map,
    }


def parse_csv_floats(s: str):
    if s is None or s.strip() == "":
        return []
    return [float(x.strip()) for x in s.split(",")]


def parse_csv_strings(s: str):
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",")]


def fill_or_validate(values, n, default):
    if len(values) == 0:
        return [default] * n
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"Expected {n} entries, got {len(values)}")
    return values


def make_box_mesh(comm, Lx_m, Ly_m, Lz_m, h_nm, celltype_name):
    Nx = max(1, math.ceil((Lx_m * 1e9) / h_nm))
    Ny = max(1, math.ceil((Ly_m * 1e9) / h_nm))
    Nz = max(1, math.ceil((Lz_m * 1e9) / h_nm))

    if celltype_name == "hex":
        celltype = CellType.hexahedron
    elif celltype_name == "tet":
        celltype = CellType.tetrahedron
    else:
        raise ValueError("celltype must be 'hex' or 'tet'")

    msh = mesh.create_box(
        comm,
        [[-0.5 * Lx_m, -0.5 * Ly_m, 0.0], [0.5 * Lx_m, 0.5 * Ly_m, Lz_m]],
        [Nx, Ny, Nz],
        cell_type=celltype,
    )
    return msh, Nx, Ny, Nz


def build_cell_fields(msh, layers, dielectric_map, rho_layers_C_m3):
    Q0 = fem.functionspace(msh, ("DG", 0))

    eps_r = fem.Function(Q0)
    eps_r.name = "eps_r_dg0"

    rho = fem.Function(Q0)
    rho.name = "rho_dg0"

    mat_id = fem.Function(Q0)
    mat_id.name = "mat_id_dg0"

    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    cells = np.arange(ncells, dtype=np.int32)
    mids = mesh.compute_midpoints(msh, tdim, cells)
    zmid_nm = mids[:, 2] * 1e9

    z0 = 0.0
    spans = []
    for layer in layers:
        z1 = z0 + layer.height_nm
        spans.append((z0, z1, layer))
        z0 = z1

    eps_arr = np.zeros(ncells, dtype=np.float64)
    rho_arr = np.zeros(ncells, dtype=np.float64)
    mat_arr = np.zeros(ncells, dtype=np.float64)

    for i, (za, zb, layer) in enumerate(spans, start=1):
        sel = (zmid_nm >= za - 1e-12) & (zmid_nm < zb + 1e-12)
        eps_arr[sel] = dielectric_map[layer.material]
        rho_arr[sel] = rho_layers_C_m3[i - 1]
        mat_arr[sel] = float(i)

    eps_r.x.array[:] = eps_arr
    rho.x.array[:] = rho_arr
    mat_id.x.array[:] = mat_arr
    return eps_r, rho, mat_id, spans


def locate_boundary_sets(msh, disk_cx_m, disk_cy_m, disk_r_m, Lz_m):
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)

    all_bfacets = mesh.exterior_facet_indices(msh.topology)
    mids = mesh.compute_midpoints(msh, fdim, all_bfacets)

    x = mids[:, 0]
    y = mids[:, 1]
    z = mids[:, 2]
    r2 = (x - disk_cx_m) ** 2 + (y - disk_cy_m) ** 2

    top_mask = np.isclose(z, 0.0)
    bottom_mask = np.isclose(z, Lz_m)
    disk_mask = top_mask & (r2 <= disk_r_m * disk_r_m)
    top_free_mask = top_mask & ~disk_mask
    side_mask = ~(disk_mask | top_free_mask | bottom_mask)

    disk_facets = all_bfacets[disk_mask].astype(np.int32)
    top_free_facets = all_bfacets[top_free_mask].astype(np.int32)
    bottom_facets = all_bfacets[bottom_mask].astype(np.int32)
    side_facets = all_bfacets[side_mask].astype(np.int32)

    if disk_facets.size == 0:
        raise RuntimeError("No disk-gate facets found. Increase resolution or disk radius.")

    facet_indices = np.hstack([disk_facets, top_free_facets, bottom_facets, side_facets])
    facet_values = np.hstack([
        np.full(disk_facets.size, 1, dtype=np.int32),
        np.full(top_free_facets.size, 2, dtype=np.int32),
        np.full(bottom_facets.size, 3, dtype=np.int32),
        np.full(side_facets.size, 4, dtype=np.int32),
    ])
    order = np.argsort(facet_indices)
    facet_tags = meshtags(msh, fdim, facet_indices[order], facet_values[order])

    return facet_tags, disk_facets, top_free_facets, bottom_facets, side_facets


def build_bcs(V, facet_tags, V_gate, V_bottom, bottom_bc_type, side_bc_type, V_side):
    fdim = V.mesh.topology.dim - 1
    bcs = []

    gate_facets = facet_tags.find(1)
    gate_dofs = fem.locate_dofs_topological(V, fdim, gate_facets)
    bcs.append(fem.dirichletbc(PETSc.ScalarType(V_gate), gate_dofs, V))

    if bottom_bc_type.upper() == "DIRICHLET":
        bottom_facets = facet_tags.find(3)
        bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(V_bottom), bottom_dofs, V))

    if side_bc_type == "dirichlet":
        side_facets = facet_tags.find(4)
        side_dofs = fem.locate_dofs_topological(V, fdim, side_facets)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(V_side), side_dofs, V))

    return bcs


def interpolate_to_cg1(msh, f_in, name):
    V1 = fem.functionspace(msh, ("CG", 1))
    f_out = fem.Function(V1)
    f_out.name = name
    f_out.interpolate(f_in)
    f_out.x.scatter_forward()
    return f_out


def sample_center_probe_z(msh, phi, npts=101):
    zmin = 0.0
    zmax = float(np.max(msh.geometry.x[:, 2]))
    zs = np.linspace(zmin, zmax, npts)

    from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

    pts = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])
    tree = bb_tree(msh, msh.topology.dim)
    candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh, candidates, pts)

    vals = np.full(npts, np.nan)
    cells = np.full(npts, -1, dtype=np.int32)
    inside = np.zeros(npts, dtype=bool)
    for i in range(npts):
        links = colliding.links(i)
        if len(links) > 0:
            inside[i] = True
            cells[i] = links[0]

    if np.any(inside):
        vals[inside] = np.asarray(phi.eval(pts[inside], cells[inside])).reshape(-1)

    return zs * 1e9, vals


def main():
    ap = argparse.ArgumentParser(description="Editable disk-gate Poisson/Laplace benchmark with optional XML/CIF import.")

    ap.add_argument("--xml", default=None)
    ap.add_argument("--cif", default=None)

    ap.add_argument("--Lx_nm", type=float, default=500.0)
    ap.add_argument("--Ly_nm", type=float, default=500.0)
    ap.add_argument("--disk_radius_nm", type=float, default=10.0)
    ap.add_argument("--disk_cx_nm", type=float, default=250.0)
    ap.add_argument("--disk_cy_nm", type=float, default=250.0)

    ap.add_argument("--layer_materials", type=str, default="Si70Ge30,sSi,Si70Ge30")
    ap.add_argument("--layer_heights_nm", type=str, default="50,5,250")
    ap.add_argument("--layer_epsr", type=str, default="13.05,11.7,13.05")
    ap.add_argument("--rho_layers_C_m3", type=str, default="0,0,0")

    ap.add_argument("--add_cap", action="store_true")
    ap.add_argument("--cap_material", type=str, default="Vacuum")
    ap.add_argument("--cap_epsr", type=float, default=1.0)
    ap.add_argument("--cap_thickness_nm", type=float, default=0.0)

    ap.add_argument("--V_gate", type=float, default=1.0)
    ap.add_argument("--V_bottom", type=float, default=0.0)
    ap.add_argument("--bottom_bc_type", choices=["DIRICHLET", "NATURAL"], default="DIRICHLET")
    ap.add_argument("--side_bc_type", choices=["natural", "dirichlet"], default="natural")
    ap.add_argument("--side_bc_value", type=float, default=0.0)
    ap.add_argument("--sigma_top_cm2", type=float, default=0.0)
    ap.add_argument("--interface_bc_flag", action="store_true")

    ap.add_argument("--celltype", choices=["hex", "tet"], default="tet")
    ap.add_argument("--h_nm", type=float, default=6.0)
    ap.add_argument("--periodic", choices=["none", "x", "y", "xy"], default="none")
    ap.add_argument("--petsc_ksp", default="cg")
    ap.add_argument("--petsc_pc", default="hypre")

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--basename", default=None)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD

    geom = DiskGeometry(
        Lx_nm=args.Lx_nm,
        Ly_nm=args.Ly_nm,
        disk_radius_nm=args.disk_radius_nm,
        disk_cx_nm_box=args.disk_cx_nm,
        disk_cy_nm_box=args.disk_cy_nm,
    )

    layer_materials = parse_csv_strings(args.layer_materials)
    layer_heights_nm = parse_csv_floats(args.layer_heights_nm)
    layer_epsr = parse_csv_floats(args.layer_epsr)
    rho_layers = parse_csv_floats(args.rho_layers_C_m3)

    if len(layer_materials) != len(layer_heights_nm):
        raise ValueError("layer_materials and layer_heights_nm must have the same length")

    layer_epsr = fill_or_validate(layer_epsr, len(layer_materials), 11.7)
    rho_layers = fill_or_validate(rho_layers, len(layer_materials), 0.0)

    dielectric_map = {mat: eps for mat, eps in zip(layer_materials, layer_epsr)}
    layers = [LayerSpec(name=f"layer_{i+1}", material=mat, height_nm=h)
              for i, (mat, h) in enumerate(zip(layer_materials, layer_heights_nm))]

    if args.xml is not None:
        xml_cfg = parse_xml_optional(args.xml)
        if xml_cfg["gate_bias"] is not None:
            args.V_gate = xml_cfg["gate_bias"]
        if xml_cfg["bottom_bias"] is not None:
            args.V_bottom = xml_cfg["bottom_bias"]
        if xml_cfg["bottom_bc_type"] is not None:
            args.bottom_bc_type = xml_cfg["bottom_bc_type"].upper()
        args.interface_bc_flag = bool(xml_cfg["interface_bc_flag"])
        args.sigma_top_cm2 = xml_cfg["surface_src_density_cm2"]

        if len(xml_cfg["layers"]) > 0:
            layers = xml_cfg["layers"]
            dielectric_map = xml_cfg["dielectric_map"]
            rho_layers = fill_or_validate(rho_layers, len(layers), 0.0)

        if args.add_cap is False and xml_cfg["cap_gate_dist_nm"] > 0:
            if args.cap_thickness_nm == 0.0:
                args.cap_thickness_nm = xml_cfg["cap_gate_dist_nm"]
            if args.cap_material == "Vacuum":
                args.cap_material = xml_cfg["cap_material"]
                args.cap_epsr = dielectric_map.get(args.cap_material, args.cap_epsr)

    if args.cif is not None:
        geom = parse_cif_optional(args.cif)

    if args.add_cap and args.cap_thickness_nm > 0:
        dielectric_map[args.cap_material] = args.cap_epsr
        layers = [LayerSpec(name="cap", material=args.cap_material, height_nm=args.cap_thickness_nm)] + layers
        rho_layers = [0.0] + rho_layers

    rho_layers = fill_or_validate(rho_layers, len(layers), 0.0)

    if args.periodic != "none":
        raise RuntimeError("This write-checked version is for non-periodic runs first. Once this is verified, periodic can be re-enabled cleanly.")

    if args.side_bc_type == "dirichlet" and args.periodic != "none":
        raise ValueError("Use either periodic sides or side Dirichlet BCs, not both.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    basename = args.basename if args.basename else outdir.name
    xdmf_path = outdir / f"{basename}.xdmf"
    h5_path = outdir / f"{basename}.h5"
    json_path = outdir / f"{basename}_config.json"
    summary_path = outdir / f"{basename}_summary.txt"
    probe_path = outdir / f"{basename}_probe_z.csv"

    Lx_m = geom.Lx_nm * 1e-9
    Ly_m = geom.Ly_nm * 1e-9
    Lz_m = sum(layer.height_nm for layer in layers) * 1e-9

    disk_r_m = geom.disk_radius_nm * 1e-9
    disk_cx_m = geom.disk_cx_nm_phys * 1e-9
    disk_cy_m = geom.disk_cy_nm_phys * 1e-9

    msh, Nx, Ny, Nz = make_box_mesh(comm, Lx_m, Ly_m, Lz_m, args.h_nm, args.celltype)
    eps_r_dg0, rho_dg0, mat_id_dg0, spans = build_cell_fields(msh, layers, dielectric_map, rho_layers)
    facet_tags, disk_facets, top_free_facets, bottom_facets, side_facets = locate_boundary_sets(
        msh, disk_cx_m, disk_cy_m, disk_r_m, Lz_m
    )

    V = fem.functionspace(msh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bcs = build_bcs(V, facet_tags, args.V_gate, args.V_bottom, args.bottom_bc_type, args.side_bc_type, args.side_bc_value)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    sigma_C_m2 = args.sigma_top_cm2 * 1.0e4 * E_CHARGE if args.interface_bc_flag else 0.0
    zero = fem.Constant(msh, PETSc.ScalarType(0.0))

    a = ufl.inner(EPS0 * eps_r_dg0 * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rho_dg0 * v * ufl.dx + zero * v * ufl.dx + PETSc.ScalarType(sigma_C_m2) * v * ds(2)

    phi = fem.Function(V)
    phi.name = "phi"

    problem = LinearProblem(
        a,
        L,
        u=phi,
        bcs=bcs,
        petsc_options_prefix="diskbox_",
        petsc_options={
            "ksp_type": args.petsc_ksp,
            "pc_type": args.petsc_pc,
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
            "ksp_error_if_not_converged": True,
        },
    )
    problem.solve()
    phi.x.scatter_forward()

    rho = interpolate_to_cg1(msh, rho_dg0, "rho")
    eps_r = interpolate_to_cg1(msh, eps_r_dg0, "eps_r")
    mat_id = interpolate_to_cg1(msh, mat_id_dg0, "mat_id")

    local_phi_min = float(np.min(phi.x.array))
    local_phi_max = float(np.max(phi.x.array))
    gphi_min = comm.allreduce(local_phi_min, op=MPI.MIN)
    gphi_max = comm.allreduce(local_phi_max, op=MPI.MAX)

    local_rho_min = float(np.min(rho.x.array))
    local_rho_max = float(np.max(rho.x.array))
    grho_min = comm.allreduce(local_rho_min, op=MPI.MIN)
    grho_max = comm.allreduce(local_rho_max, op=MPI.MAX)

    local_eps_min = float(np.min(eps_r.x.array))
    local_eps_max = float(np.max(eps_r.x.array))
    geps_min = comm.allreduce(local_eps_min, op=MPI.MIN)
    geps_max = comm.allreduce(local_eps_max, op=MPI.MAX)

    local_mat_min = float(np.min(mat_id.x.array))
    local_mat_max = float(np.max(mat_id.x.array))
    gmat_min = comm.allreduce(local_mat_min, op=MPI.MIN)
    gmat_max = comm.allreduce(local_mat_max, op=MPI.MAX)

    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi)
        xdmf.write_function(rho)
        xdmf.write_function(eps_r)
        xdmf.write_function(mat_id)

    if comm.rank == 0:
        zs_nm, phi_probe = sample_center_probe_z(msh, phi, npts=101)
        np.savetxt(
            probe_path,
            np.column_stack([zs_nm, phi_probe]),
            delimiter=",",
            header="z_nm,phi_center",
            comments=""
        )

        cfg = {
            "xml": args.xml,
            "cif": args.cif,
            "basename": basename,
            "Lx_nm": geom.Lx_nm,
            "Ly_nm": geom.Ly_nm,
            "Lz_nm": Lz_m * 1e9,
            "disk_radius_nm": geom.disk_radius_nm,
            "disk_center_nm_box": [geom.disk_cx_nm_box, geom.disk_cy_nm_box],
            "disk_center_nm_phys": [geom.disk_cx_nm_phys, geom.disk_cy_nm_phys],
            "celltype": args.celltype,
            "h_nm": args.h_nm,
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "V_gate": args.V_gate,
            "V_bottom": args.V_bottom,
            "bottom_bc_type": args.bottom_bc_type,
            "side_bc_type": args.side_bc_type,
            "side_bc_value": args.side_bc_value,
            "interface_bc_flag": args.interface_bc_flag,
            "sigma_top_cm2": args.sigma_top_cm2 if args.interface_bc_flag else 0.0,
            "layers": [
                {
                    "name": layer.name,
                    "material": layer.material,
                    "height_nm": layer.height_nm,
                    "eps_r": dielectric_map[layer.material],
                    "rho_C_m3": rho_layers[i],
                }
                for i, layer in enumerate(layers)
            ],
            "phi_min": gphi_min,
            "phi_max": gphi_max,
            "rho_min": grho_min,
            "rho_max": grho_max,
            "eps_r_min": geps_min,
            "eps_r_max": geps_max,
            "mat_id_min": gmat_min,
            "mat_id_max": gmat_max,
            "disk_facets": int(disk_facets.size),
            "top_free_facets": int(top_free_facets.size),
            "bottom_facets": int(bottom_facets.size),
            "side_facets": int(side_facets.size),
        }
        json_path.write_text(json.dumps(cfg, indent=2))

        with open(summary_path, "w") as f:
            f.write("=== Disk benchmark summary ===\n")
            f.write(f"basename     : {basename}\n")
            f.write(f"mesh         : {args.celltype}, h={args.h_nm} nm -> ({Nx},{Ny},{Nz})\n")
            f.write(f"box [nm]     : ({geom.Lx_nm:.3f}, {geom.Ly_nm:.3f}, {Lz_m*1e9:.3f})\n")
            f.write(f"disk [nm]    : R={geom.disk_radius_nm:.3f}, center_box=({geom.disk_cx_nm_box:.3f}, {geom.disk_cy_nm_box:.3f})\n")
            f.write(f"disk [nm]    : center_phys=({geom.disk_cx_nm_phys:.3f}, {geom.disk_cy_nm_phys:.3f})\n")
            f.write(f"BCs          : gate={args.V_gate:.6f} V, bottom={args.V_bottom:.6f} V ({args.bottom_bc_type}), side={args.side_bc_type}\n")
            f.write(f"sigma_top    : {(args.sigma_top_cm2 if args.interface_bc_flag else 0.0):.6e} e/cm^2\n")
            for i, (za, zb, layer) in enumerate(spans):
                f.write(
                    f"layer {i+1}     : {layer.material:12s} z=[{za:.3f}, {zb:.3f}] nm, "
                    f"eps_r={dielectric_map[layer.material]:.4f}, rho={rho_layers[i]:.6e} C/m^3\n"
                )
            f.write(f"facets       : disk={disk_facets.size}, top_free={top_free_facets.size}, bottom={bottom_facets.size}, side={side_facets.size}\n")
            f.write(f"phi min/max  : {gphi_min:.12e}, {gphi_max:.12e}\n")
            f.write(f"rho min/max  : {grho_min:.12e}, {grho_max:.12e}\n")
            f.write(f"eps min/max  : {geps_min:.12e}, {geps_max:.12e}\n")
            f.write(f"mat min/max  : {gmat_min:.12e}, {gmat_max:.12e}\n")
            f.write(f"xdmf         : {xdmf_path.name}\n")
            f.write(f"h5           : {h5_path.name}\n")
            f.write(f"probe        : {probe_path.name}\n")

        print("=== Disk benchmark summary ===")
        print(f"basename     : {basename}")
        print(f"mesh         : {args.celltype}, h={args.h_nm} nm -> ({Nx},{Ny},{Nz})")
        print(f"box [nm]     : ({geom.Lx_nm:.3f}, {geom.Ly_nm:.3f}, {Lz_m*1e9:.3f})")
        print(f"disk [nm]    : R={geom.disk_radius_nm:.3f}, center_box=({geom.disk_cx_nm_box:.3f}, {geom.disk_cy_nm_box:.3f})")
        print(f"disk [nm]    : center_phys=({geom.disk_cx_nm_phys:.3f}, {geom.disk_cy_nm_phys:.3f})")
        print(f"BCs          : gate={args.V_gate:.6f} V, bottom={args.V_bottom:.6f} V ({args.bottom_bc_type}), side={args.side_bc_type}")
        print(f"phi min/max  : {gphi_min:.12e}, {gphi_max:.12e}")
        print(f"rho min/max  : {grho_min:.12e}, {grho_max:.12e}")
        print(f"eps min/max  : {geps_min:.12e}, {geps_max:.12e}")
        print(f"mat min/max  : {gmat_min:.12e}, {gmat_max:.12e}")
        print("Wrote:")
        print(f"  {xdmf_path}")
        print(f"  {h5_path}")
        print(f"  {json_path}")
        print(f"  {summary_path}")
        print(f"  {probe_path}")


if __name__ == "__main__":
    main()
