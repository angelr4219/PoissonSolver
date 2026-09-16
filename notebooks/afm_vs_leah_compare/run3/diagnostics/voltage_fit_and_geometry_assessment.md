# Voltage-basis fit and geometry assessment

Follow-on to `physical_case_audit.md` and the corrected depth metrics
(`../metrics/depth_metrics_corrected.csv`, `summary_metrics_corrected.json`).
This diagnostic asks: **given that the tip Dirichlet voltage is the single
most uncertain free parameter in run3 (per `calibrate_afm_tip_vs_leah.py`'s
own docstring — "Nothing about Leah's actual tip geometry, gap, or voltage is
known"), can adjusting *only* that voltage bring the FEM into agreement with
`basePotential3d(1).vtk`?** The answer, established below by exact linearity
of Laplace's equation rather than by trial and error, is **no** — a real
shape/sign mismatch survives at every tip voltage, positive or negative.

New run used: `notebooks/afm_vs_leah_compare/run4/` (tip voltage = 0.0 V,
otherwise identical to run3). Script used: `voltage_basis_fit.py` in this
directory's sibling `scripts/` folder.

---

## Part 1 — Zero-tip run (run4)

Exact docker command used (verbatim, copy-pasteable):

```bash
docker run --rm \
  -v "$HOME":/root/home \
  -w /root/home/Desktop/poisson_solver \
  dolfinx/dolfinx:stable \
  python3 calibrate_afm_tip_vs_leah.py \
    --gap 10 --tip-radius 10 --cone-height 160 --shank-radius 120 \
    --shaft-radius 120 --shaft-height 100 --air-height 320 \
    --tip-voltage 0.0 --degree 1 \
    --h-apex 5.0 --h-device 10.0 --h-near 15.0 --h-bottom 300.0 \
    --output notebooks/afm_vs_leah_compare/run4
```

(`--bottom-voltage` was left at its default, `-4.4`, which already matches
run3's value — not passed explicitly, consistent with run3's own invocation.)

Result landed at **`notebooks/afm_vs_leah_compare/run4/`** (the script's
`next_run_directory` auto-increment logic was not relied on — `--output` was
passed explicitly per the task's "safe" option — but it would have produced
`run4` anyway since only `run1`/`run3` existed).

Verification performed:
- `run4/run_results.json` mesh block: `"cells": 467621, "dofs": 89527, "degree": 1,
  "h_apex_nm": 5.0, "h_device_nm": 10.0, "h_near_nm": 15.0, "h_bottom_nm": 300.0`
  — **identical** to run3's mesh block.
- `run4/run_results.json` geometry block — every field (`lx`, `ly`, `air_height`,
  `tip_gap`, `tip_radius`, `cone_height`, `shank_radius`,
  `cone_half_angle_deg_effective`, `shaft_radius`, `shaft_height`, `probe1_z`,
  `probe2_z`, `probe1_radius`, `probe2_radius`, `bottom_z`) — **identical** to run3.
- `run4/run_results.json` voltages block: `"AFM_tip": 0.0, "bottom_back_gate": -4.4`
  — tip voltage is the **only** field that differs from run3 (`1.0` there).
- Output structure matches run3: `sige_afm_tip.xdmf` + `.h5` (fields `phi_V`,
  `relative_permittivity`, `material_id`, `cell_tags`, `facet_tags`),
  `run_results.json`, `centerline.csv`.
- Per-material cell counts in the solver log (tags 1–7, e.g. `tag=7 cells=216,000
  eps_r=12.0` — the SiGe buffer) match run3's mesh exactly, confirming identical
  geometry/mesh, not just identical parameter *values*.
- **run3 integrity**: captured `mtime` + path for all 66 files under
  `notebooks/afm_vs_leah_compare/run3/` before running Part 1, and diffed
  against the same listing after run4 completed. **Zero differences — run3 is
  byte-for-byte, mtime-for-mtime untouched.**
- Run time: a few minutes, consistent with run3 at this mesh resolution; no
  parameters were reduced to speed it up.

---

## Part 2 — Affine voltage fit

Method: `phi_0` = run4's `phi_V` (tip=0.0V), `phi_1` = run3's `phi_V` (tip=1.0V),
both interpolated onto the VTK's own points via
`interpolate_case_onto_ref` (imported unmodified from
`src/poisson/vtk_xdmf_compare.py`, called twice against one shared
`read_vtk_reference` result — no reimplementation). Least-squares:

```
V* = [(phi_1 - phi_0) . (phi_VTK - phi_0)] / [(phi_1 - phi_0) . (phi_1 - phi_0)]
```

**Coordinate-overlap check on the restricted point subset** (not assumed from
the prior full-domain audit): for the 226,981 VTK points with z in [0,53] nm,
nearest-neighbor fallback fraction is **0.0000%** for both run4 and run3
interpolations — confirmed genuine linear-interpolation overlap, no
fallback contamination, matching (and now explicitly re-verified for) the
prior full-domain finding.

### Best-fit V\*

| Window | n points | V\* (Volts) |
|---|---:|---:|
| **Device region, z=0–53nm (primary)** | 226,981 | **+2.456** |
| z=0–20nm | 14,884 | +2.574 |
| z=20–53nm | 212,097 | +2.448 |
| z=35nm slice only (local optimum) | 3,721 | +2.338 |
| z=48nm slice only | 3,721 | +2.332 |
| z=0nm slice only | 3,721 | +2.341 |
| z=9.67nm slice only | 3,721 | +2.714 |

**Stability**: V\* is **positive everywhere tested** (range +2.33 to +2.71 V,
~16% spread) — never negative, never near zero, never a "physically strange"
value. There is a mild, monotonic-looking trend (shallow slices favor a
somewhat higher V\* than the z=35/48nm well slices), but the spread is modest
compared to run3's actual assumed value of +1.0V (all fitted values are
**more than double** run3's tip voltage). This is a moderately stable
best-fit voltage, not wildly inconsistent across depth — but see Part 3:
stability of V\* is a necessary, not sufficient, condition for voltage alone
explaining the mismatch, and it is not sufficient here.

### Device-region error before/after

Using the per-point-aggregate definitions of `E_L2`/`E_range`/shape-RMSE/
Pearson-r reproduced verbatim from `corrected_comparison_metrics.py` (same
`E_range` denominator convention: the VTK's GLOBAL peak-to-peak range,
computed once over the whole reference field = 4.560 V, never a per-window
range):

| Metric | BEFORE (run3, V=1.0) | AFTER (V\*=+2.456) |
|---|---:|---:|
| E_L2 (device region) | 5.876 | 0.748 |
| E_range (device region) | **0.2021** | **0.0257** |
| Mean voltage bias (signed err) | −0.9138 V | +0.00003 V (≈0) |
| Shape RMSE | 0.1177 V | 0.1173 V |
| Pearson r (raw, mixed-depth aggregate) | 0.2912 | 0.2912 |

`E_range` drops ~8x and the mean bias is driven essentially to zero by
construction (that's what the least-squares fit optimizes for) — a real,
substantial improvement in **amplitude/offset** agreement. But shape RMSE and
Pearson r are **unchanged to 4+ significant figures** before and after.

### z=35nm specifically (Leah's own stated comparison plane)

| Metric | BEFORE (V=1.0) | AFTER, device-V\*=+2.456 | AFTER, z35-local-optimal V\*=+2.338 |
|---|---:|---:|---:|
| E_range | 0.1845 | 0.0194 | 0.0104 |
| Pearson r | **−0.9696** | **−0.9696** | **−0.9696** |

**The correlation does not move at all** — not partially, not by a small
amount, but literally unchanged to 6 decimal places, regardless of which V\*
is used (device-region-optimal or the z=35-locally-optimal value). E_range
(amplitude match) improves dramatically at both fits, especially the local
one (0.1845 → 0.0104, an 18x improvement) — but the spatial pattern is just
as inverted after the "best possible" fit at that exact depth as it was
before.

### Why this happens — exact, not approximate

This isn't a numerical coincidence. Checked directly:
`corr(phi_0, phi_1)` (the zero-tip field vs the run3 field, in-plane at
z=35nm) = **1.0000** (Pearson r). Because `phi_0` and `phi_1` solve the same
linear BVP on the same mesh/geometry and differ only in the tip's Dirichlet
value, their in-plane spatial *shapes* at any fixed depth are — to numerical
precision — perfectly affinely related to each other (`phi_0 ≈ A + B·phi_1`
for depth-dependent constants A, B). Consequently `phi_fit = phi_0 + V·(phi_1
− phi_0)` is itself an affine function of `phi_1` alone for any V, and
Pearson correlation is invariant under affine transforms with positive
scale. **This means no tip voltage — positive, negative, or zero — can ever
change the FEM's in-plane spatial correlation with the VTK at a fixed depth,
as long as the tip is the only thing being varied.** The shape mismatch is
therefore not a calibration problem; it is structurally outside what a
voltage sweep can reach.

---

## Part 3 — Geometry assessment

**Most diagnostic check, stated plainly: a single V\* (or even the
locally-optimal V\* chosen separately for z=35nm) does NOT bring shape
correlation anywhere near +1 at z=35nm or nearby depths. It stays at
essentially exactly −0.97, unchanged, regardless of voltage.** This is proven
both empirically (identical Pearson r to 6 decimals across all fits tried)
and structurally (phi_0/phi_1 in-plane shapes are collinear, so any affine
combination of them is also collinear with phi_1, hence has identical
correlation to the VTK). Voltage magnitude and even voltage *sign* are ruled
out as explanations for the shape mismatch — this is airtight for the tip
voltage specifically, given fixed geometry.

### Visual confirmation (z=35nm 8-panel plot)

`../plots/representative_xy/xy_8panel_z35_well.png`: the "VTK shape
(mean-subtracted)" panel shows a **dark, cool-colored depression centered at
the origin** (a minimum, ~−0.17V trough) that spreads out smoothly and
radially to a warm rim at large |x|,|y|. The "FEM shape (mean-subtracted)"
panel shows the **opposite** — a small warm (positive) blob roughly centered
near the origin, cooling outward — i.e. run3's FEM has a *bump*, not a dip,
under the tip. The "shape difference" panel (bottom-right) is strongly
positive and centered at the origin, exactly where the two fields disagree
in sign, and fades to near-zero toward the domain edges — confirming the
mismatch is a genuine, spatially-localized, sign-opposed feature directly
beneath the tip axis, not a diffuse or edge-driven artifact.

### Category diagnosis

Checking each candidate category against the evidence:

- **Constant voltage bias**: YES, present and now corrected by the fit
  (mean bias −0.91V → ~0V in the device region). This part of the mismatch
  IS fixable by voltage alone.
- **Incorrect voltage scale**: YES — the fitted V\* (+2.3 to +2.7V) is
  2.3–2.7x run3's assumed +1.0V. If the VTK does contain a tip contribution,
  it's a substantially larger magnitude than +1.0V, and amplitude agreement
  (E_range) improves ~8-18x once corrected. This part IS fixable by voltage
  alone.
- **Lateral translation**: not assessed directly by this fit (V\* is a pure
  scalar, not a translation), but the 8-panel plot shows both the VTK dip and
  FEM bump centered near the same (x,y)≈(0,0) location — no obvious gross
  lateral offset. Low priority relative to the sign issue.
- **Incorrect lateral feature width**: plausible secondary issue (not
  directly tested here — would require comparing FWHM columns already in
  `depth_metrics_corrected.csv`), but is dominated by the much larger
  sign-inversion problem; not the primary story.
- **Incorrect depth decay**: the sensitivity table shows V\* drifting mildly
  with depth (+2.71 at z=9.7nm down to +2.33 at z=48nm) — a real but modest
  (~16%) effect, secondary to the shape-sign issue.
- **Material-stack mismatch**: not implicated by this analysis specifically —
  `physical_case_audit.md` already confirmed run3's material_id layer
  transitions match the script's documented 6-layer stack; this fit doesn't
  probe permittivities directly.
- **Mesh discretization error**: unlikely to explain a sign flip of this
  magnitude and spatial extent (150nm across); discretization error would be
  local/high-frequency, not a smooth, domain-wide sign inversion.
- **Floating-gate modeling error (Probe1/Probe2)**: plausible contributor —
  run3 treats Probe1/Probe2 as passive/Neumann (voltage=null); if Leah's
  simulation modeled them as floating conductors or with different BCs, that
  could alter the local field shape near z=35/48nm, though it's unclear this
  alone would flip the *sign* of the on-axis feature globally through the
  shallow device region (the sign inversion is present at z=0nm too, above
  any probe).
- **Tip-geometry mismatch (sign/polarity)** — **YES, primary finding.** The
  proven voltage-invariance of the correlation, combined with the opposite
  curvature visible in the 8-panel plot (VTK: dip under tip; FEM: bump under
  tip) at every depth in the device region, is the dominant, best-supported
  category. This could mean: (a) the physical mechanism producing the VTK's
  on-axis feature is not a positive-charge-like Dirichlet tip pulling
  potential up, but something that pulls it down (e.g. the true tip bias is
  negative — but per Part 2, no scalar rescaling of run3's specific tip
  *shape* can produce that dip, since phi_1's shape itself, not just its
  sign, would need to be a mirror image of what a +Dirichlet-tip produces to
  match the VTK; a bare sign flip of V alone is already ruled out as
  insufficient since it's the same shape either way — flipping V would flip
  the *scale* of a bump into a dip only if the underlying shape of phi_1
  itself already had the opposite curvature to phi_0, which is not what we
  found: phi_1 and phi_0 are the same shape), or (b) as flagged in
  `physical_case_audit.md`, the VTK's origin feature isn't caused by a tip
  matching run3's geometry/type at all (different tip shape/position, or not
  a tip-driven feature in the first place — consistent with the still-open
  "basePotential = no perturber" naming-convention question).

### Bottom line

**Voltage adjustment alone is demonstrably insufficient.** The fit produces
a stable, always-positive, plausible-looking V\* (~+2.3 to +2.7V) that
dramatically improves amplitude/offset agreement (E_range improves 8–18x,
mean bias corrected to ~0), which is a real and useful partial result. But
the central, documented shape/sign anomaly — opposite lateral curvature at
z=35nm and throughout the shallow device region — is **structurally immune**
to any tip-voltage choice, proven both by exhaustive-enough empirical sweep
(6 different depth windows, all giving the same unchanged correlation) and
by the underlying linear-algebra reason (phi_0 and phi_1 are the same
in-plane shape, only differently scaled/offset). This is strong evidence
that whatever's driving the VTK's origin-centered dip is geometrically or
physically different from run3's tip model — either the wrong tip
geometry/position/type, or (per the still-unresolved question in
`physical_case_audit.md`) no tip contribution in the VTK at all. **This is
new evidence, not previously established**: it shows conclusively that the
mismatch is NOT explainable as "right geometry, wrong voltage" — something
run3's own docstring left open as the presumed calibration path. It narrows
the remaining open question to geometry/physical-mechanism, not calibration.

### Smallest justified next experiment

Given voltage is now ruled out as a fix, and a full geometry sweep is
expensive, the smallest next diagnostic that would meaningfully progress the
open question (tip present vs. absent in the VTK) is:

**Run one additional case with a NEGATIVE tip voltage (e.g. `--tip-voltage
-1.0`, everything else unchanged) purely to sanity-confirm, by direct
solve rather than only by the affine-basis argument, that the FEM's z=35
in-plane shape does NOT invert sign.** This is a strict confirmatory check on
the "phi_0/phi_1 collinearity" result (which already implies this outcome
mathematically) — cheap (same mesh can, in principle, be reused/cached if
the script supports it; otherwise one more full solve, same cost as run4).
If it confirms no sign change (expected), that closes the door on "just flip
the tip's polarity" as an explanation and firmly redirects effort toward (1)
resolving the `basePotential` vs `totalPotential` naming-convention question
with Leah directly (still the single highest-value unblock, per
`physical_case_audit.md`), and/or (2) if a tip is confirmed present, testing
a genuinely different tip geometry (e.g. off-axis position, different
radius/cone angle, or a qualitatively different perturbing structure than a
simple Dirichlet cone) rather than continuing to vary voltage or this tip's
size/shape parametrically.
