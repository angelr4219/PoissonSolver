def build_2d(mode: str, Lx: float, Ly: float, h: float, R: float = None):
    """
    Build a unit rectangle with a disk/rect inclusion using OCC.
    Returns (domain, cell_tags, facet_tags).
    """
    import gmsh
    from mpi4py import MPI

    gmsh.initialize() if not gmsh.isInitialized() else None
    gmsh.model.add("rod2d")

    # ---- 1) Make surfaces in OCC and keep THEIR TAGS
    rect_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)

    if mode in ("rod2d", "disk2d"):
        assert R is not None and R > 0.0
        inc_tag = gmsh.model.occ.addDisk(Lx/2.0, Ly/2.0, 0.0, R, R)
    elif mode == "rect_hole2d":
        w = h if R is None else R
        inc_tag = gmsh.model.occ.addRectangle(Lx/2.0 - w/2.0, Ly/2.0 - w/2.0, 0.0, w, w)
    else:
        raise ValueError(f"Unknown 2D mode: {mode}")

    # ---- 2) Keep BOTH regions (outer + inclusion) as separate materials
    gmsh.model.occ.fragment([(2, rect_tag)], [(2, inc_tag)])
    gmsh.model.occ.synchronize()

    # ---- 3) Tag physical groups for cells (dim=2)
    ents2 = gmsh.model.occ.getEntities(2)
    areas = [(tag, gmsh.model.occ.getMass(2, tag)) for (_, tag) in ents2]
    areas.sort(key=lambda t: t[1])
    inc_surf = areas[0][0]
    outer_surf = areas[-1][0]

    gmsh.model.addPhysicalGroup(2, [outer_surf], 10)
    gmsh.model.setPhysicalName(2, 10, "outer")
    gmsh.model.addPhysicalGroup(2, [inc_surf], 20)
    gmsh.model.setPhysicalName(2, 20, "inclusion")

    # Tag outer boundary curves
    loops = gmsh.model.getBoundary([(2, outer_surf)], oriented=False, recursive=False)
    outer_curves = [t for (dim, t) in loops if dim == 1]
    if outer_curves:
        gmsh.model.addPhysicalGroup(1, outer_curves, 101)
        gmsh.model.setPhysicalName(1, 101, "outer_boundary")

    # ---- 4) Mesh size + generate
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(2)

    # ---- 5) Convert to dolfinx
    return gmsh.model
def build_3d(mode: str, Lx: float, Ly: float, L: float, h: float, R: float = None):
    """
    3D prism with a through cylindrical inclusion (bore/air) using OCC.
    Returns gmsh.model; conversion is handled elsewhere (helpers).
    Tags:
      volumes: "solid"(110), "bore"(120)
      faces:   "outer_boundary"(201), "bore_boundary"(202)
    """
    import gmsh
    assert mode in ("cylinder3d", "rod3d"), f"Unknown 3D mode: {mode}"
    assert R is not None and R > 0.0

    # Safe init if not already
    gmsh.initialize() if not gmsh.isInitialized() else None
    gmsh.model.add("rod3d")

    # Geometry (OCC)
    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, L)
    cyl = gmsh.model.occ.addCylinder(Lx/2.0, Ly/2.0, 0.0, 0.0, 0.0, L, R)

    # Keep both domains (solid + bore)
    gmsh.model.occ.fragment([(3, box)], [(3, cyl)])
    gmsh.model.occ.synchronize()

    # Volume tags by mass (smaller = bore)
    vols = gmsh.model.occ.getEntities(3)
    masses = [(tag, gmsh.model.occ.getMass(3, tag)) for (_, tag) in vols]
    masses.sort(key=lambda t: t[1])
    bore_vol  = masses[0][0]
    solid_vol = masses[-1][0]

    gmsh.model.addPhysicalGroup(3, [solid_vol], 110)
    gmsh.model.setPhysicalName(3, 110, "solid")
    gmsh.model.addPhysicalGroup(3, [bore_vol], 120)
    gmsh.model.setPhysicalName(3, 120, "bore")

    # Face groups
    solid_faces = gmsh.model.getBoundary([(3, solid_vol)], oriented=False, recursive=False)
    bore_faces  = gmsh.model.getBoundary([(3, bore_vol)],   oriented=False, recursive=False)
    bore_face_tags  = {t for (d, t) in bore_faces if d == 2}
    solid_face_tags = [t for (d, t) in solid_faces if d == 2]
    outer_faces = [t for t in solid_face_tags if t not in bore_face_tags]

    if outer_faces:
        gmsh.model.addPhysicalGroup(2, outer_faces, 201)
        gmsh.model.setPhysicalName(2, 201, "outer_boundary")
    if bore_face_tags:
        gmsh.model.addPhysicalGroup(2, list(bore_face_tags), 202)
        gmsh.model.setPhysicalName(2, 202, "bore_boundary")

    # Mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(3)

    return gmsh.model

def build_3d(mode: str, Lx: float, Ly: float, L: float, h: float, R: float = None):
    """
    3D prism with a through cylindrical inclusion (bore/air) using OCC.
    Returns gmsh.model; conversion is handled elsewhere (helpers).
    Tags:
      volumes: "solid"(110), "bore"(120)
      faces:   "outer_boundary"(201), "bore_boundary"(202)
    """
    import gmsh
    assert mode in ("cylinder3d", "rod3d"), f"Unknown 3D mode: {mode}"
    assert R is not None and R > 0.0

    # Safe init if not already
    gmsh.initialize() if not gmsh.isInitialized() else None
    gmsh.model.add("rod3d")

    # Geometry (OCC)
    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, L)
    cyl = gmsh.model.occ.addCylinder(Lx/2.0, Ly/2.0, 0.0, 0.0, 0.0, L, R)

    # Keep both domains (solid + bore)
    gmsh.model.occ.fragment([(3, box)], [(3, cyl)])
    gmsh.model.occ.synchronize()

    # Volume tags by mass (smaller = bore)
    vols = gmsh.model.occ.getEntities(3)
    masses = [(tag, gmsh.model.occ.getMass(3, tag)) for (_, tag) in vols]
    masses.sort(key=lambda t: t[1])
    bore_vol  = masses[0][0]
    solid_vol = masses[-1][0]

    gmsh.model.addPhysicalGroup(3, [solid_vol], 110)
    gmsh.model.setPhysicalName(3, 110, "solid")
    gmsh.model.addPhysicalGroup(3, [bore_vol], 120)
    gmsh.model.setPhysicalName(3, 120, "bore")

    # Face groups
    solid_faces = gmsh.model.getBoundary([(3, solid_vol)], oriented=False, recursive=False)
    bore_faces  = gmsh.model.getBoundary([(3, bore_vol)],   oriented=False, recursive=False)
    bore_face_tags  = {t for (d, t) in bore_faces if d == 2}
    solid_face_tags = [t for (d, t) in solid_faces if d == 2]
    outer_faces = [t for t in solid_face_tags if t not in bore_face_tags]

    if outer_faces:
        gmsh.model.addPhysicalGroup(2, outer_faces, 201)
        gmsh.model.setPhysicalName(2, 201, "outer_boundary")
    if bore_face_tags:
        gmsh.model.addPhysicalGroup(2, list(bore_face_tags), 202)
        gmsh.model.setPhysicalName(2, 202, "bore_boundary")

    # Mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(3)

    return gmsh.model

def build_3d(mode: str, Lx: float, Ly: float, L: float, h: float, R: float = None):
    """
    3D prism with a through cylindrical inclusion (bore/air) using OCC.
    Returns gmsh.model; conversion is handled elsewhere (helpers).
    Tags:
      volumes: "solid"(110), "bore"(120)
      faces:   "outer_boundary"(201), "bore_boundary"(202)
    """
    import gmsh
    assert mode in ("cylinder3d", "rod3d"), f"Unknown 3D mode: {mode}"
    assert R is not None and R > 0.0

    # Safe init if not already
    gmsh.initialize() if not gmsh.isInitialized() else None
    gmsh.model.add("rod3d")

    # Geometry (OCC)
    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, L)
    cyl = gmsh.model.occ.addCylinder(Lx/2.0, Ly/2.0, 0.0, 0.0, 0.0, L, R)

    # Keep both domains (solid + bore)
    gmsh.model.occ.fragment([(3, box)], [(3, cyl)])
    gmsh.model.occ.synchronize()

    # Volume tags by mass (smaller = bore)
    vols = gmsh.model.occ.getEntities(3)
    masses = [(tag, gmsh.model.occ.getMass(3, tag)) for (_, tag) in vols]
    masses.sort(key=lambda t: t[1])
    bore_vol  = masses[0][0]
    solid_vol = masses[-1][0]

    gmsh.model.addPhysicalGroup(3, [solid_vol], 110)
    gmsh.model.setPhysicalName(3, 110, "solid")
    gmsh.model.addPhysicalGroup(3, [bore_vol], 120)
    gmsh.model.setPhysicalName(3, 120, "bore")

    # Face groups
    solid_faces = gmsh.model.getBoundary([(3, solid_vol)], oriented=False, recursive=False)
    bore_faces  = gmsh.model.getBoundary([(3, bore_vol)],   oriented=False, recursive=False)
    bore_face_tags  = {t for (d, t) in bore_faces if d == 2}
    solid_face_tags = [t for (d, t) in solid_faces if d == 2]
    outer_faces = [t for t in solid_face_tags if t not in bore_face_tags]

    if outer_faces:
        gmsh.model.addPhysicalGroup(2, outer_faces, 201)
        gmsh.model.setPhysicalName(2, 201, "outer_boundary")
    if bore_face_tags:
        gmsh.model.addPhysicalGroup(2, list(bore_face_tags), 202)
        gmsh.model.setPhysicalName(2, 202, "bore_boundary")

    # Mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(3)

    return gmsh.model
