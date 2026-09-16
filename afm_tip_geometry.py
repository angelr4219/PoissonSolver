import gmsh


def add_afm_tip(
    occ,
    x_center=0.0,
    y_center=0.0,
    surface_z=0.0,
    gap=10.0,
    tip_radius=10.0,
    shank_height=160.0,
    shank_top_radius=120.0,
    shaft_radius=120.0,
    shaft_height=100.0,
):
    """
    Add a fused AFM-tip conductor using Gmsh OpenCASCADE.

    Geometry, all dimensions in nm:

        cylindrical shaft
        radius = 120 nm
        height = 100 nm
              ||
              ||
        ______||______
         \          /
          \        /
           \      /      conical shank
            \    /       height = 160 nm
             \  /        radius 10 -> 120 nm
              \/
             (  )        spherical apex
              \/         radius = 10 nm
               |
          gap = 10 nm
    ---------------------- sample surface, z = 0

    Returns
    -------
    tip : list[(dim, tag)]
        Fused 3D OpenCASCADE volume representing the conductor.

    info : dict
        Useful z positions and geometry dimensions.
    """

    # Lowest point of the sphere:
    #
    # z_apex = surface_z + gap
    #
    # Therefore the sphere center is one radius higher.
    z_sphere_center = surface_z + gap + tip_radius

    # ---------------------------------------------------------
    # 1. Spherical apex
    # ---------------------------------------------------------
    sphere = occ.addSphere(
        x_center,
        y_center,
        z_sphere_center,
        tip_radius,
    )

    # ---------------------------------------------------------
    # 2. Conical shank
    #
    # Bottom radius = tip_radius = 10 nm
    # Top radius    = 120 nm
    # Height        = 160 nm
    #
    # The cone starts at the sphere-center plane so it overlaps
    # the upper portion of the sphere. The Boolean fuse makes
    # the result one continuous conductor.
    # ---------------------------------------------------------
    cone = occ.addCone(
        x_center,
        y_center,
        z_sphere_center,
        0.0,
        0.0,
        shank_height,
        tip_radius,
        shank_top_radius,
    )

    # ---------------------------------------------------------
    # 3. Cylindrical shaft
    #
    # Starts directly at the wide end of the cone.
    # Its radius matches the cone top radius.
    # ---------------------------------------------------------
    z_shaft_bottom = z_sphere_center + shank_height

    shaft = occ.addCylinder(
        x_center,
        y_center,
        z_shaft_bottom,
        0.0,
        0.0,
        shaft_height,
        shaft_radius,
    )

    # ---------------------------------------------------------
    # Fuse sphere + cone
    # ---------------------------------------------------------
    sphere_cone, _ = occ.fuse(
        [(3, sphere)],
        [(3, cone)],
        removeObject=True,
        removeTool=True,
    )

    # ---------------------------------------------------------
    # Fuse with cylindrical shaft
    # ---------------------------------------------------------
    tip, _ = occ.fuse(
        sphere_cone,
        [(3, shaft)],
        removeObject=True,
        removeTool=True,
    )

    info = {
        "surface_z": surface_z,
        "gap": gap,
        "apex_z": surface_z + gap,
        "sphere_center_z": z_sphere_center,
        "tip_radius": tip_radius,
        "shank_height": shank_height,
        "shank_bottom_radius": tip_radius,
        "shank_top_radius": shank_top_radius,
        "shaft_bottom_z": z_shaft_bottom,
        "shaft_radius": shaft_radius,
        "shaft_height": shaft_height,
        "tip_top_z": z_shaft_bottom + shaft_height,
    }

    return tip, info
