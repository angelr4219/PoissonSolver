# main.py
from poisson_linear import poisson_linear

if __name__ == "__main__":
    uh = poisson_linear()
    values = uh.x.array
    if values.size:
        print(f"phi: min={values.min():.3e} V, max={values.max():.3e} V")
