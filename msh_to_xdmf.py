#!/usr/bin/env python3
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import read_from_msh
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--msh", required=True)
    p.add_argument("--out", required=True, help="XDMF base name (no extension)")
    p.add_argument("--gdim", type=int, default=3)
    args = p.parse_args()

    domain, cell_tags, facet_tags = read_from_msh(args.msh, MPI.COMM_WORLD, gdim=args.gdim)

    # Mesh + cell tags
    with XDMFFile(domain.comm, f"{args.out}.xdmf", "w") as xf:
        xf.write_mesh(domain)
        if cell_tags is not None:
            xf.write_meshtags(cell_tags, domain.geometry)

    # Facet tags
    if facet_tags is not None:
        with XDMFFile(domain.comm, f"{args.out}_facets.xdmf", "w") as xf:
            xf.write_mesh(domain)
            xf.write_meshtags(facet_tags, domain.geometry)

if __name__ == "__main__":
    main()
