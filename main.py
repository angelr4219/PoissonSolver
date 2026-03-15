#!/usr/bin/env python3
from __future__ import annotations

import argparse

from poisson_solver.cases import topdisk3d
from poisson_solver.cases import charged_disk_box
from poisson_solver.cases import square_gate_stack
from poisson_solver.cases import rod_bore
from poisson_solver.cases import layered_disk_gate_box


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refactored Poisson/Laplace solver entry point."
    )
    subparsers = parser.add_subparsers(dest="case", required=True)

    topdisk3d.register_parser(subparsers)
    charged_disk_box.register_parser(subparsers)
    square_gate_stack.register_parser(subparsers)
    rod_bore.register_parser(subparsers)
    layered_disk_gate_box.register_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
