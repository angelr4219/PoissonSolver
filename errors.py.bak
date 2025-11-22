from __future__ import annotations
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
from dolfinx.fem import petsc as fempetsc
import ufl

def interpolate_exact_into(V: fem.FunctionSpace, exact_fun):
    uE = fem.Function(V, name="u_exact")
    uE.interpolate(exact_fun)
    return uE

def global_errors(domain: mesh.Mesh, uh: fem.Function, uE: fem.Function):
    e  = uE - uh
    dx = ufl.dx(domain)
    L2_sq      = fem.assemble_scalar(fem.form(ufl.inner(e, e) * dx))
    H1_semi_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx))
    a = uh.x.array; b = uE.x.array
    Linf_dof_local = float(np.max(np.abs(a - b))) if a.size else 0.0
    comm = domain.comm
    L2_sq      = comm.allreduce(L2_sq, op=MPI.SUM)
    H1_semi_sq = comm.allreduce(H1_semi_sq, op=MPI.SUM)
    Linf_dof   = comm.allreduce(Linf_dof_local, op=MPI.MAX)
    return float(np.sqrt(L2_sq)), float(np.sqrt(H1_semi_sq)), float(Linf_dof)

def cellwise_error(domain: mesh.Mesh, uh: fem.Function, uE: fem.Function, qdeg: int = 4):
    e  = uE - uh
    dx = ufl.dx(domain, metadata={"quadrature_degree": qdeg})
    V0 = fem.functionspace(domain, ("DG", 0))
    w  = ufl.TestFunction(V0)
    f_L2 = ufl.inner(e, e) * w * dx
    f_H1 = ufl.inner(ufl.grad(e), ufl.grad(e)) * w * dx
    b_L2 = fempetsc.create_vector(fem.form(f_L2)); fempetsc.assemble_vector(b_L2, fem.form(f_L2))
    b_H1 = fempetsc.create_vector(fem.form(f_H1)); fempetsc.assemble_vector(b_H1, fem.form(f_H1))
    err2_int  = fem.Function(V0, name="cell_int_e2")
    grad2_int = fem.Function(V0, name="cell_int_grad_e2")
    err2_int.x.array[:]  = b_L2.getArray(readonly=True)
    grad2_int.x.array[:] = b_H1.getArray(readonly=True)
    l2_cell  = np.sqrt(err2_int.x.array.copy())
    h1s_cell = np.sqrt(grad2_int.x.array.copy())
    return l2_cell, h1s_cell, err2_int, grad2_int

def cell_centroids(domain: mesh.Mesh):
    tdim = domain.topology.dim
    num_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_local, dtype=np.int32)
    centroids = mesh.compute_midpoints(domain, tdim, cells)
    return cells, centroids

def report_errors(domain: mesh.Mesh, uh: fem.Function, exact_fun, out_prefix: str | None = None, qdeg: int = 4):
    V  = uh.function_space
    uE = interpolate_exact_into(V, exact_fun)
    L2, H1s, Linf_dof = global_errors(domain, uh, uE)
    l2_cell, h1s_cell, err2_int, grad2_int = cellwise_error(domain, uh, uE, qdeg=qdeg)
    local_cells, centroids = cell_centroids(domain)
    if l2_cell.size:
        i_local = int(np.argmax(l2_cell))
        max_cell_id_local = int(local_cells[i_local])
        max_cell_centroid = centroids[i_local]
        max_cell_val = float(l2_cell[i_local])
    else:
        max_cell_id_local, max_cell_centroid, max_cell_val = None, None, 0.0
    if out_prefix:
        with XDMFFile(domain.comm, f"{out_prefix}_err2_cellint.xdmf", "w") as xf:
            xf.write_mesh(domain); xf.write_function(err2_int)
        with XDMFFile(domain.comm, f"{out_prefix}_grad2_cellint.xdmf", "w") as xf:
            xf.write_mesh(domain); xf.write_function(grad2_int)
    return {
        "L2": L2,
        "H1_seminorm": H1s,
        "Linf_dof": Linf_dof,
        "l2_cell": l2_cell,
        "h1s_cell": h1s_cell,
        "max_cell": {
            "local_id": max_cell_id_local,
            "centroid": max_cell_centroid,
            "L2_cell_norm": max_cell_val,
        },
    }
