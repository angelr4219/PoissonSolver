#!/usr/bin/env python3
import sys, numpy as np
import matplotlib
# Use a non-interactive backend so it works inside docker/servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def load_csv(path):
    x, u = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,1), unpack=True)
    order = np.argsort(x)
    return x[order], u[order]

def common_grid(x1, x2, n=2000):
    xmin = max(np.min(x1), np.min(x2))
    xmax = min(np.max(x1), np.max(x2))
    if xmin >= xmax:
        raise RuntimeError("No overlapping x-range between the two inputs.")
    return np.linspace(xmin, xmax, n)

def edge_safe_mask(x, xs_gates, a, band):
    mask = np.abs(x) <= 2*a
    for e in np.concatenate([xs_gates - a, xs_gates + a]):
        mask &= (np.abs(x - e) > band)
    return mask

def main(p1="results/phi_p4a_line.csv", p2="results/phi_p5a_line.csv"):
    # geometry (keep consistent with your runs)
    a_nm = 35.0
    a = a_nm * 1e-9
    xs_gates = np.array([-2*a, 0.0, 2*a])

    # load and align
    x1,u1 = load_csv(p1)
    x2,u2 = load_csv(p2)
    X = common_grid(x1, x2, n=min(4000, max(len(x1), len(x2))*2))
    U1 = np.interp(X, x1, u1)
    U2 = np.interp(X, x2, u2)
    dU = U2 - U1

    # edge-safe window (~2 * median Δx around gate edges)
    dx = np.median(np.diff(X)) if X.size>1 else a*0.01
    band = 2.0*abs(dx)
    mask = edge_safe_mask(X, xs_gates, a, band)
    if not np.any(mask):
        print("[WARN] edge-safe mask empty; using full overlap range.")
        mask = np.ones_like(X, dtype=bool)

    # metrics
    dU_mask = dU[mask]
    Linf = float(np.max(np.abs(dU_mask)))
    L2   = float(np.sqrt(np.mean(dU_mask**2)))
    print(f"Comparing:\n  A: {p1}\n  B: {p2}")
    print(f"Overlap points: {X.size}, edge-safe points: {int(np.sum(mask))}")
    print(f"Δφ = φ_B - φ_A  |  L∞ = {Linf:.4e} V,  L2 = {L2:.4e} V")

    # plot
    fig, ax = plt.subplots(figsize=(8,4.5), dpi=120)
    ax.plot(X, U1, label=f"A ({Path(p1).stem})")
    ax.plot(X, U2, label=f"B ({Path(p2).stem})", linestyle="--")
    ax.plot(X[mask], dU[mask], label="Δφ (B−A), edge-safe", linewidth=1.25)
    # mark gate edges
    for e in np.concatenate([xs_gates - a, xs_gates + a]):
        ax.axvline(e, alpha=0.2, linewidth=0.8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("Potential φ (V)")
    ax.set_title(f"Comparison of box sizes\nL∞(Δφ)={Linf:.3e} V,  L2(Δφ)={L2:.3e} V (edge-safe)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    # save
    Path("results").mkdir(exist_ok=True, parents=True)
    out_png = f"results/compare_{Path(p1).stem}_vs_{Path(p2).stem}.png"
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Saved plot: {out_png}")

    # try to show if a display exists (outside headless docker this may pop a window)
    try:
        import os
        if os.environ.get("DISPLAY"):
            plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
