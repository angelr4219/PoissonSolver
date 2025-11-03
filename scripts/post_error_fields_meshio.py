import os, argparse, importlib, inspect, ast, json
import numpy as np
import ufl, meshio
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from basix.ufl import element as basix_element
from src.error_fields import write_error_fields_xdmf

def _load_builder(mod_func: str):
    mod, func = mod_func.split(":", 1)
    return getattr(importlib.import_module(mod), func)

def _auto_builder():
    mod = importlib.import_module("exact_rect_gates")
    best = None
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        nl = name.lower()
        if any(k in nl for k in ("phi", "rect", "gate")):
            best = fn
            if "rect" in nl and "phi" in nl:
                break
    if best is None:
        raise RuntimeError("Could not auto-detect analytic builder in exact_rect_gates. Pass --analytic module:function.")
    return best

def _read_mesh_and_phi_timeseries(xdmf_path: str, degree: int, field_name: str):
    with meshio.xdmf.TimeSeriesReader(xdmf_path) as reader:
        points, cells = reader.read_points_cells()
        _, point_data, _ = reader.read_data(0)
    if not cells:
        raise RuntimeError("No cells found in XDMF.")
    ctype = cells[0].type
    cdata = cells[0].data
    if "tetra" in ctype:
        cellname = "tetrahedron"
    elif "hexa" in ctype or "hex" in ctype:
        cellname = "hexahedron"
    else:
        raise RuntimeError(f"Unsupported cell type: {ctype}")
    ve = basix_element("Lagrange", cellname, 1, shape=(3,))
    ufl_mesh = ufl.Mesh(ve)
    domain = dmesh.create_mesh(MPI.COMM_WORLD, cdata, points, ufl_mesh)
    V = fem.functionspace(domain, ("CG", degree))
    if field_name not in point_data:
        raise RuntimeError(f"Field '{field_name}' not in point_data. Available: {list(point_data.keys())}")
    vals = np.asarray(point_data[field_name], dtype=float).squeeze()
    dof_xyz = V.tabulate_dof_coordinates().reshape((-1, domain.geometry.dim))
    try:
        from scipy.spatial import cKDTree as KDTree
        _, idx = KDTree(points).query(dof_xyz, k=1)
        mapped = vals[idx].squeeze()
    except Exception:
        mapped = np.empty(dof_xyz.shape[0])
        for i, p in enumerate(dof_xyz):
            j = np.argmin(np.linalg.norm(points - p, axis=1))
            mapped[i] = vals[j]
    phi = fem.Function(V, name=field_name)
    phi.x.array[:] = mapped.astype(float)
    return domain, V, phi

def _call_builder_flex(builder, x, y, z, core_kwargs, extra_kwargs):
    """Support direct builders and factories; handle vector vs scalar arg styles."""
    sig = inspect.signature(builder)
    params = list(sig.parameters.keys())
    kw_all = {**{k: v for k, v in core_kwargs.items() if k in params}, **extra_kwargs}
    X = ufl.as_vector((x, y, z))

    # Try direct-style first if it looks like (x,y,z,...) signature
    if len(params) >= 3 and params[0].lower() in ("x","x0") and params[1].lower() in ("y","x1") and params[2].lower() in ("z","x2"):
        try:
            return builder(x, y, z, **kw_all)
        except TypeError:
            return builder(X, **kw_all)

    # Otherwise factory-style: builder(**kw) -> callable
    factory = builder(**kw_all)
    try:
        return factory(X)             # expects a single coordinate vector
    except TypeError:
        return factory(x, y, z)       # fallback: three scalars

def main():
    ap = argparse.ArgumentParser(description="Make *_error.xdmf from phi.xdmf (meshio time series)")
    ap.add_argument("--phi-xdmf", required=True)
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--field-name", default="phi")
    ap.add_argument("--analytic", default=None, help="module:function; if omitted, auto-detect in exact_rect_gates")
    ap.add_argument("--extra", default=None, help="Python dict literal of extra kwargs")
    ap.add_argument("--extra-json", default=None, help="JSON dict of extra kwargs (preferred for shells)")
    # core params
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--vp1", type=float, required=True)
    ap.add_argument("--vx", type=float, required=True)
    ap.add_argument("--vp2", type=float, required=True)
    ap.add_argument("--tau", type=float, default=1e-9)
    ap.add_argument("--out")
    args = ap.parse_args()

    domain, V, phi = _read_mesh_and_phi_timeseries(args.phi_xdmf, args.degree, args.field_name)
    builder = _load_builder(args.analytic) if args.analytic else _auto_builder()

    # extra kwargs
    extra_kwargs = {}
    if args.extra_json:
        extra_kwargs = json.loads(args.extra_json)
        if not isinstance(extra_kwargs, dict):
            raise SystemExit("ERROR: --extra-json must be a JSON object (dict).")
    elif args.extra:
        try:
            extra_kwargs = ast.literal_eval(args.extra)
            if not isinstance(extra_kwargs, dict):
                raise ValueError
        except Exception:
            raise SystemExit("ERROR: --extra must be a Python dict literal.")

    core_kwargs = dict(a=args.a, vp1=args.vp1, vx=args.vx, vp2=args.vp2)
    xcrd = ufl.SpatialCoordinate(domain)
    phi_exact_ufl = _call_builder_flex(builder, xcrd[0], xcrd[1], xcrd[2], core_kwargs, extra_kwargs)

    out = args.out or (os.path.splitext(args.phi_xdmf)[0] + "_error.xdmf")
    write_error_fields_xdmf(domain, V, phi, phi_exact_ufl, out, tau=args.tau)

if __name__ == "__main__":
    main()
