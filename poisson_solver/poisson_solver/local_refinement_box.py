import numpy as np
import gmsh as gmsh_api
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from dolfinx import fem


def make_function_space(mesh, family, degree):
    """
    Compatibility helper for newer/older DOLFINx versions.
    """
    try:
        return fem.functionspace(mesh, (family, degree))
    except AttributeError:
        return fem.FunctionSpace(mesh, (family, degree))


def main():
    comm = MPI.COMM_WORLD
    rank = 0

    gmsh_api.initialize()
    gmsh_api.model.add("local_refinement_box")

    # ----------------------------
    # Geometry sizes
    # ----------------------------
    Lx, Ly, Lz = 500e-9, 500e-9, 150e-9
    cx, cy, cz = 0.0, 0.0, 60e-9

    fine_Lx, fine_Ly, fine_Lz = 80e-9, 80e-9, 30e-9

    h_coarse = 20e-9
    h_fine = 4e-9

    x_min_f = cx - fine_Lx / 2
    x_max_f = cx + fine_Lx / 2
    y_min_f = cy - fine_Ly / 2
    y_max_f = cy + fine_Ly / 2
    z_min_f = cz - fine_Lz / 2
    z_max_f = cz + fine_Lz / 2

    # ----------------------------
    # Large simulation box
    # ----------------------------
    box = gmsh_api.model.occ.addBox(
        -Lx / 2, -Ly / 2, 0.0,
        Lx, Ly, Lz
    )

    gmsh_api.model.occ.synchronize()

    # ----------------------------
    # Physical volume
    # This is one real volume. The refinement region is NOT
    # a physical subvolume yet. It is a mesh-size field.
    # ----------------------------
    domain_tag = 1
    gmsh_api.model.addPhysicalGroup(3, [box], domain_tag)
    gmsh_api.model.setPhysicalName(3, domain_tag, "domain")

    # ----------------------------
    # Local refinement box as mesh-size field only
    # ----------------------------
    field = gmsh_api.model.mesh.field.add("Box")

    gmsh_api.model.mesh.field.setNumber(field, "VIn", h_fine)
    gmsh_api.model.mesh.field.setNumber(field, "VOut", h_coarse)

    gmsh_api.model.mesh.field.setNumber(field, "XMin", x_min_f)
    gmsh_api.model.mesh.field.setNumber(field, "XMax", x_max_f)
    gmsh_api.model.mesh.field.setNumber(field, "YMin", y_min_f)
    gmsh_api.model.mesh.field.setNumber(field, "YMax", y_max_f)
    gmsh_api.model.mesh.field.setNumber(field, "ZMin", z_min_f)
    gmsh_api.model.mesh.field.setNumber(field, "ZMax", z_max_f)

    gmsh_api.model.mesh.field.setAsBackgroundMesh(field)

    # Keep mesh sizes controlled by the field
    gmsh_api.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh_api.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh_api.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh_api.option.setNumber("Mesh.CharacteristicLengthMin", h_fine)
    gmsh_api.option.setNumber("Mesh.CharacteristicLengthMax", h_coarse)

    # ----------------------------
    # Generate mesh
    # ----------------------------
    gmsh_api.model.mesh.generate(3)

    # ----------------------------
    # Convert to DOLFINx mesh
    # ----------------------------
    mesh_data = gmshio.model_to_mesh(
        gmsh_api.model,
        comm,
        rank=rank,
        gdim=3
    )

    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    mesh.name = "locally_refined_box_mesh"

    # ----------------------------
    # Create visible cell-wise fields
    #
    # material_id:
    #   1 = refined/fine region
    #   2 = coarse outer region
    #
    # target_h_nm:
    #   4  = target h in fine region, nm
    #   20 = target h in coarse region, nm
    # ----------------------------
    V0 = make_function_space(mesh, "DG", 0)

    def inside_refinement_box(x):
        return (
            (x[0] >= x_min_f) & (x[0] <= x_max_f) &
            (x[1] >= y_min_f) & (x[1] <= y_max_f) &
            (x[2] >= z_min_f) & (x[2] <= z_max_f)
        )

    material_id = fem.Function(V0)
    material_id.name = "material_id"
    material_id.interpolate(
        lambda x: np.where(inside_refinement_box(x), 1.0, 2.0)
    )

    target_h_nm = fem.Function(V0)
    target_h_nm.name = "target_h_nm"
    target_h_nm.interpolate(
        lambda x: np.where(inside_refinement_box(x), h_fine * 1e9, h_coarse * 1e9)
    )

    material_id.x.scatter_forward()
    target_h_nm.x.scatter_forward()

    # ----------------------------
    # Count cells in each region using actual cell midpoints
    # ----------------------------
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)

    num_cells_local = mesh.topology.index_map(tdim).size_local
    cell_to_vertex = mesh.topology.connectivity(tdim, 0)
    cell_vertices = cell_to_vertex.array.reshape(num_cells_local, -1)
    midpoints = mesh.geometry.x[cell_vertices].mean(axis=1)

    xmid = midpoints[:, 0]
    ymid = midpoints[:, 1]
    zmid = midpoints[:, 2]

    fine_mask = (
        (xmid >= x_min_f) & (xmid <= x_max_f) &
        (ymid >= y_min_f) & (ymid <= y_max_f) &
        (zmid >= z_min_f) & (zmid <= z_max_f)
    )

    n_fine_local = int(np.count_nonzero(fine_mask))
    n_total_local = int(num_cells_local)
    n_fine_global = comm.allreduce(n_fine_local, op=MPI.SUM)
    n_total_global = comm.allreduce(n_total_local, op=MPI.SUM)
    n_coarse_global = n_total_global - n_fine_global

    # ----------------------------
    # Write XDMF/H5 output
    # This creates:
    #   locally_refined_box.xdmf
    #   locally_refined_box.h5
    # ----------------------------
    xdmf_path = "locally_refined_box.xdmf"

    with XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(material_id)
        xdmf.write_function(target_h_nm)

        if cell_tags is not None:
            cell_tags.name = "cell_tags"
            try:
                xdmf.write_meshtags(cell_tags, mesh.geometry)
            except TypeError:
                xdmf.write_meshtags(cell_tags)

        if facet_tags is not None:
            facet_tags.name = "facet_tags"
            mesh.topology.create_connectivity(tdim - 1, tdim)
            try:
                xdmf.write_meshtags(facet_tags, mesh.geometry)
            except TypeError:
                xdmf.write_meshtags(facet_tags)

    if comm.rank == 0:
        print("Built locally refined box mesh")
        print(f"Domain size: {Lx*1e9:.1f} x {Ly*1e9:.1f} x {Lz*1e9:.1f} nm")
        print(f"Fine region center: ({cx*1e9:.1f}, {cy*1e9:.1f}, {cz*1e9:.1f}) nm")
        print(f"Fine region size: {fine_Lx*1e9:.1f} x {fine_Ly*1e9:.1f} x {fine_Lz*1e9:.1f} nm")
        print(f"Fine region bounds:")
        print(f"  x: {x_min_f*1e9:.1f} to {x_max_f*1e9:.1f} nm")
        print(f"  y: {y_min_f*1e9:.1f} to {y_max_f*1e9:.1f} nm")
        print(f"  z: {z_min_f*1e9:.1f} to {z_max_f*1e9:.1f} nm")
        print(f"h_fine = {h_fine*1e9:.2f} nm")
        print(f"h_coarse = {h_coarse*1e9:.2f} nm")
        print("")
        print("Cell marker counts:")
        print(f"  material_id = 1, fine/refined region: {n_fine_global}")
        print(f"  material_id = 2, coarse region:       {n_coarse_global}")
        print(f"  total cells:                          {n_total_global}")
        print("")
        print("Wrote: locally_refined_box.xdmf")
        print("Wrote: locally_refined_box.h5")
        print("")
        print("In ParaView:")
        print("  1. Open locally_refined_box.xdmf")
        print("  2. Apply the Clip filter or Slice filter through the center")
        print("  3. Set Representation to 'Surface With Edges'")
        print("  4. Color by 'material_id'")
        print("  5. material_id = 1 is the fine/refined region")
        print("  6. material_id = 2 is the coarse outer region")
        print("")
        print("Optional:")
        print("  Color by 'target_h_nm' instead.")
        print("  target_h_nm = 4 means fine target mesh size.")
        print("  target_h_nm = 20 means coarse target mesh size.")

    gmsh_api.finalize()


if __name__ == "__main__":
    main()
