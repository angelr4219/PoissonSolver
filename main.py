# main.py
from poisson_linear import poisson_linear
import argparse
from poisson_intro_solution import *
from poissonLinear2 import *
from DoS_2D import *

if __name__ == "__main__":
    uh = poisson_linear()
    values = uh.x.array
    if values.size:
        print(f"phi: min={values.min():.3e} V, max={values.max():.3e} V")
