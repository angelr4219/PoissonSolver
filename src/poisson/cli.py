from __future__ import annotations
import argparse, sys, subprocess, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=["2d","3d"])
    ap.add_argument("--mpirun", type=int, default=1)
    ap.add_argument("--options_file", type=str, default="petsc.options")
    args, rest = ap.parse_known_args()

    script = "verify/dielectric_interface_2D.py" if args.demo=="2d" else "verify/image_charge_3D.py"
    cmd = []
    if args.mpirun > 1:
        cmd += ["mpirun", "-n", str(args.mpirun)]
    cmd += [sys.executable, script, "--options_file", args.options_file]
    cmd += rest
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
