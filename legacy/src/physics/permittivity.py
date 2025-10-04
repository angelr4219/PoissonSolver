from dolfinx import fem
import numpy as np

EPS0 = 8.854187817e-12  # F/m

def eps_from_materials(
    domain,
    cell_tags,
    tag_map=None,                 # {tag:int -> epsr:float} OR {tag:int -> name:str}
    material_epsr=None,           # optional {name:str -> epsr:float}; used if tag_map has names
    eps0: float = EPS0,
    space: str = "DG",
    degree: int = 0,
    **kwargs
) -> fem.Function:
    """
    Build a piecewise permittivity field: ε(x) = ε0 * εr(tag) per cell (DG0 by default).
    - domain: dolfinx.mesh.Mesh
    - cell_tags: dolfinx.mesh.meshtags on cells (.indices, .values)
    """
    # default name->epsr table (override via material_epsr if desired)
    default_name_epsr = {
        "si": 11.7, "silicon": 11.7,
        "sige": 13.0, "silicon-germanium": 13.0, "silicon germanium": 13.0,
        "ge": 16.0, "germanium": 16.0,
        "sio2": 3.9, "oxide": 3.9,
        "air": 1.0, "vacuum": 1.0, "bore": 1.0,
        "outer": 11.7, "inclusion": 13.0, "solid": 11.7,
    }
    if material_epsr is None:
        material_epsr = default_name_epsr

    fam = "DG" if space and space.upper().startswith("DG") else "CG"
    # Use factory (lowercase) in new dolfinx
    V = fem.functionspace(domain, (fam, degree))
    eps = fem.Function(V, name="epsilon")

    tdim = domain.topology.dim
    n_cells_local = domain.topology.index_map(tdim).size_local

    # start as vacuum everywhere
    eps_local = np.full(n_cells_local, eps0 * 1.0, dtype=np.float64)

    tags = cell_tags.values.astype(np.int64)
    cells = cell_tags.indices.astype(np.int64)

    # Build tag -> eps_r map
    tag_to_epsr = {}
    if isinstance(tag_map, dict) and len(tag_map) > 0:
        sample = next(iter(tag_map.values()))
        if isinstance(sample, (int, float, np.floating)):
            tag_to_epsr = {int(k): float(v) for k, v in tag_map.items()}
        else:
            # names -> epsr
            for k, name in tag_map.items():
                key = str(name).strip().lower()
                tag_to_epsr[int(k)] = float(material_epsr.get(key, 1.0))

    # apply by cell tag (fallback to 1.0 if tag not mapped)
    for tag in np.unique(tags):
        er = float(tag_to_epsr.get(int(tag), 1.0))
        eps_local[cells[tags == tag]] = eps0 * er

    # assign to function dofs
    if eps.x.array.size == eps_local.size:
        eps.x.array[:] = eps_local
    else:
        # if user requested a higher-order space, keep cellwise-constant values
        rep = eps.x.array.size // eps_local.size
        eps.x.array[:] = np.repeat(eps_local, rep)
    eps.x.scatter_forward()
    return eps
