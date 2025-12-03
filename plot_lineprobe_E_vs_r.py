import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to CSV from point_charge_box verification
csv_path = Path("results/point_charge_box/lineprobe_z.csv")

print(f"Loading CSV from: {csv_path}")
data = np.genfromtxt(csv_path, delimiter=",", names=True)

r      = data["r"]       # distance from charge
E_num  = data["E_num"]   # FEM |E|
E_ref  = data["E_ref"]   # analytic |E| ~ 1/r^2

# Mask out r ~ 0 and NaNs/zeros
mask = (r > 0) & np.isfinite(E_num) & (E_num != 0)

r_plot     = r[mask]
E_num_plot = np.abs(E_num[mask])
E_ref_plot = np.abs(E_ref[mask])

# ---- Fit slope on log–log (optional, for sanity) ----
log_r = np.log10(r_plot)
log_E = np.log10(E_num_plot)
slope, intercept = np.polyfit(log_r, log_E, 1)
print(f"Fitted slope for log10|E_num| vs log10 r: {slope:.3f} (expected ≈ -2)")

# ---- 1) Log–log plot: should look like 1/r^2 ----
plt.figure()
plt.loglog(r_plot, E_num_plot, label="|E| FEM")
plt.loglog(r_plot, E_ref_plot, "--", label="|E| analytic 1/r^2")

plt.xlabel("r (m)")
plt.ylabel("|E| (V/m)")
plt.title("|E| vs r (log–log)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.savefig("lineprobe_E_vs_r_loglog.png", dpi=200)
print("Saved log–log plot to lineprobe_E_vs_r_loglog.png")

# ---- 2) Linear plot just to see monotonic decrease ----
plt.figure()
plt.plot(r_plot * 1e9, E_num_plot, label="|E| FEM")
plt.xlabel("r (nm)")
plt.ylabel("|E| (V/m)")
plt.title("|E| vs r (linear)")
plt.grid(True)
plt.tight_layout()
plt.savefig("lineprobe_E_vs_r_linear.png", dpi=200)
print("Saved linear plot to lineprobe_E_vs_r_linear.png")

plt.show()
