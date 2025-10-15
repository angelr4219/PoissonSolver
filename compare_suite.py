#!/usr/bin/env python3
import re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# geometry (keep in sync with your solver)
a_nm = 35.0
a    = a_nm * 1e-9
xs_gates = np.array([-2*a, 0.0, 2*a])

def load_csv(path):
    x, u = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,1), unpack=True)
    order = np.argsort(x)
    return x[order], u[order]

def common_grid(x1, x2, n=2000):
    xmin = max(np.min(x1), np.min(x2))
    xmax = min(np.max(x1), np.max(x2))
    if xmin >= xmax:
        raise RuntimeError("No overlapping x-range")
    return np.linspace(xmin, xmax, n)

def edge_safe_mask(x, a, xs_gates, band):
    mask = np.abs(x) <= 2*a
    for e in np.concatenate([xs_gates - a, xs_gates + a]):
        mask &= (np.abs(x - e) > band)
    return mask

def p_from_name(name):
    # expects .../phi_p{int}a_line.csv  -> returns that int
    m = re.search(r"phi_p(\d+)a_line\.csv$", name)
    return int(m.group(1)) if m else None

def compare_two(pA, pB, outdir="results"):
    xA,uA = load_csv(pA)
    xB,uB = load_csv(pB)
    X = common_grid(xA, xB, n=min(4000, max(len(xA), len(xB))*2))
    UA = np.interp(X, xA, uA)
    UB = np.interp(X, xB, uB)
    dU = UB - UA

    # edge-safe window (~2 * median Δx)
    dx = np.median(np.diff(X)) if X.size>1 else a*0.01
    band = 2.0*abs(dx)
    mask = edge_safe_mask(X, a, xs_gates, band)
    if not np.any(mask):
        mask = np.ones_like(X, dtype=bool)

    dU_m = dU[mask]
    Linf = float(np.max(np.abs(dU_m)))
    L2   = float(np.sqrt(np.mean(dU_m**2)))

    # plot
    Path(outdir).mkdir(parents=True, exist_ok=True)
    stemA, stemB = Path(pA).stem, Path(pB).stem
    fig, ax = plt.subplots(figsize=(8,4.5), dpi=120)
    ax.plot(X, UA, label=f"{stemA}")
    ax.plot(X, UB, label=f"{stemB}", linestyle="--")
    ax.plot(X[mask], dU[mask], label=f"Δφ ({stemB} − {stemA})", linewidth=1.25)
    for e in np.concatenate([xs_gates - a, xs_gates + a]):
        ax.axvline(e, alpha=0.2, linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Potential φ (V) / Δφ (V)")
    ax.set_title(f"{stemA} vs {stemB}   |   L∞(Δφ)={Linf:.3e} V,  L2(Δφ)={L2:.3e} V")
    ax.grid(True, alpha=0.25); ax.legend()
    out_png = f"{outdir}/compare_{stemA}_vs_{stemB}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    # return metrics + curve for combined plot
    return {"A": stemA, "B": stemB, "Linf": Linf, "L2": L2, "X": X, "dU": dU, "mask": mask, "png": out_png}

def main():
    files = sorted(glob.glob("results/phi_p*a_line.csv"))
    if not files:
        print("No files found under results/*.csv"); return
    # choose reference = largest p
    ps   = [(p_from_name(f), f) for f in files if p_from_name(f) is not None]
    ps   = sorted(ps, key=lambda t: t[0])
    refp, refF = ps[-1]
    print(f"Reference: p={refp}a -> {refF}")

    # compare each smaller p to reference
    summary = []
    curves  = []
    for p, f in ps[:-1]:
        m = compare_two(f, refF)
        summary.append([p, refp, m["Linf"], m["L2"], m["png"]])
        curves.append((p, m["X"][m["mask"]], m["dU"][m["mask"]]))

    # write summary CSV
    out_csv = "results/compare_summary.csv"
    np.savetxt(out_csv, np.array(summary, dtype=object), fmt="%s", delimiter=",",
               header="p_compared, p_ref, Linf_V, L2_V, plot_path", comments="")
    print(f"Wrote: {out_csv}")

    # combined Δφ plot (all vs reference)
    if curves:
        fig, ax = plt.subplots(figsize=(8,4.5), dpi=120)
        for p, X, dU in curves:
            ax.plot(X, dU, label=f"Δφ p{p}a − p{refp}a", linewidth=1.1)
        for e in np.concatenate([xs_gates - a, xs_gates + a]):
            ax.axvline(e, alpha=0.2, linewidth=0.8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Δφ (V)")
        ax.set_title(f"Change vs reference box (p{refp}a)  |  edge-safe window")
        ax.grid(True, alpha=0.25); ax.legend()
        out_png = f"results/compare_all_vs_p{refp}a.png"
        fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
        print(f"Saved combined Δφ plot: {out_png}")

if __name__ == "__main__":
    main()
