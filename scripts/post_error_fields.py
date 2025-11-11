import os, argparse, importlib, xml.etree.ElementTree as ET, ufl
from mpi4py import MPI
from dolfinx import fem, io
from src.error_fields import write_error_fields_xdmf

def load_builder(mod_func: str):
    mod, func = mod_func.split(":", 1)
    return getattr(importlib.import_module(mod), func)

def autodetect_names(xdmf_path: str):
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    grid = root.find('.//{*}Grid')
    if grid is None:
        raise RuntimeError("No <Grid> found in XDMF.")
    mesh_name = grid.attrib.get("Name") or grid.attrib.get("name") or "mesh"
    attr = grid.find('.//{*}Attribute')
    field_name = None if attr is None else (attr.attrib.get("Name") or attr.attrib.get("name"))
    return mesh_name, field_name

def main():
    ap = argparse.ArgumentParser(description="Create *_error.xdmf next to phi.xdmf")
    ap.add_argument("--phi-xdmf", required=True)
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--analytic", default="exact_rect_gates:rect_gates_phi")
    ap.add_argument("--mesh-name")
    ap.add_argument("--field-name")
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--vp1", type=float, required=True)
    ap.add_argument("--vx", type=float, required=True)
    ap.add_argument("--vp2", type=float, required=True)
    ap.add_argument("--tau", type=float, default=1e-9)
    ap.add_argument("--out")
    args = ap.parse_args()

    # Detect names if missing
    mesh_name, field_name = args.mesh_name, args.field_name
    if mesh_name is None or field_name is None:
        m_auto, f_auto = autodetect_names(args.phi_xdmf)
        mesh_name = mesh_name or m_auto
        field_name = field_name or f_auto
        if field_name is None:
            raise RuntimeError("Couldn't auto-detect a field/Attribute name in XDMF. Pass --field-name.")

    # Read mesh + phi (note: functionspace(), not FunctionSpace())
    with io.XDMFFile(MPI.COMM_WORLD, args.phi_xdmf, "r") as xdmf:
        mesh = xdmf.read_mesh(name=mesh_name)
    V = fem.functionspace(mesh, ("CG", args.degree))

    phi = fem.Function(V, name=field_name)
    with io.XDMFFile(mesh.comm, args.phi_xdmf, "r") as xdmf:
        xdmf.read_function(phi, name=field_name)

    # Analytic UFL
    builder = load_builder(args.analytic)
    x = ufl.SpatialCoordinate(mesh)
    phi_exact_ufl = builder(x[0], x[1], x[2], a=args.a, vp1=args.vp1, vx=args.vx, vp2=args.vp2)

    # Write error fields
    out = args.out or (os.path.splitext(args.phi_xdmf)[0] + "_error.xdmf")
    write_error_fields_xdmf(mesh, V, phi, phi_exact_ufl, out, tau=args.tau)

if __name__ == "__main__":
    main()
