import numpy as np
import matplotlib.pyplot as plt

# Adjust this path if needed
csv_path = "results/point_charge_box/lineprobe_z.csv"

# Load CSV, skipping the header row
data = np.genfromtxt(csv_path, delimiter=",", names=True)

r      = data["r"]       # distance from charge
E_num  = data["E_num"]   # FEM |E|
E_ref  = data["E_ref"]   # analytic |E| ~ 1/r^2

# Mask out r ~ 0 and NaNs
mask = (r > 0) & np.isfinite(E_num) & (E_num != 0)

r_plot     = r[mask]
E_num_plot = np.abs(E_num[mask])
E_ref_plot = np.abs(E_ref[mask])

# --- 1) Log–log plot to check power-law slope ---
plt.figure()
plt.loglog(r_plot, E_num_plot, label="|E| FEM")
plt.loglog(r_plot, E_ref_plot, "--", label="|E| analytic 1/r^2")

plt.xlabel("r (m)")
plt.ylabel("|E| (V/m)")
plt.title("|E| vs r (log–log)")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.show()

# --- 2) Optionally, linear plot just to see it decreasing ---
plt.figure()
plt.plot(r_plot * 1e9, E_num_plot, label="|E| FEM")
plt.xlabel("r (nm)")
plt.ylabel("|E| (V/m)")
plt.title("|E| vs r (linear)")
plt.grid(True)
plt.tight_layout()
plt.show()
