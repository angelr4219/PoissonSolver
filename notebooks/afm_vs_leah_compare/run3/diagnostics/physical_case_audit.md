# Physical case audit: does `basePotential3d(1).vtk` represent the same BVP as run3?

Read-only audit. All numbers below were independently reproduced by parsing files
directly (see "evidence" per section), not copied from prior summaries.

## Files actually checked

- `/Users/angelramirez/Downloads/basePotential3d(1).vtk` — parsed by hand (custom
  Python, no pyvista needed; legacy ASCII VTK is trivial to tokenize).
- `notebooks/afm_vs_leah_compare/run3/run_results.json`,
  `notebooks/afm_vs_leah_compare/run1/run_results.json`
- `notebooks/afm_vs_leah_compare/run3/sige_afm_tip.xdmf` + `.h5` (via `xml.etree` +
  `h5py`, same pattern as `src/poisson/vtk_xdmf_compare.py:read_xdmf_case`)
- `notebooks/afm_vs_leah_compare/run3/diagnostics/coordinate_overlap_audit.md`
  (pre-existing diagnostic from a prior session — cross-checked, not blindly trusted)
- `calibrate_afm_tip_vs_leah.py` (full docstring + material-id constants + CLI defaults)
- `compare_sige_afm_tip_vs_leah.py` (docstring, CLI)
- `VALIDATION_LADDER_NEXT_STEPS.md`
- `/Users/angelramirez/Desktop/masqe_comparison_bundle.zip` → extracted:
  `compare_basepotential_vs_masqe.py`, `fit_bg_flux.py`,
  `GlobalDevice/basePotential3d.vtk`, `GlobalDevice/baseSurfacePotential.dat`,
  `GlobalDevice/baseBottomPotential.dat`, `GlobalDevice/extendedBasePot3d.vtk`
- `/Users/angelramirez/Downloads/Hoffman2_masqegrp_Info*.txt` (MaSQE cluster-access
  instructions only — no physics content)
- Broad repo grep for `leah|masqe|basepotential` across all tracked files, and a
  `find` over `~/Downloads` for anything named `*potential*`, `*masqe*`, `*leah*`
  (found several **other, differently-sized** `basePotential3d.vtk` files — see
  "unresolved unknowns")
- No PDF, email, `.pptx`, or note describing Leah's tip voltage/geometry was found
  anywhere on Desktop or Downloads.

---

## Confirmed facts

**VTK grid/geometry** (parsed directly from `basePotential3d(1).vtk`):
- `DIMENSIONS 61 61 171`, x ∈ [-150,150] step 5nm, y ∈ [-150,150] step 5nm — matches run3's `lx=ly=300.0`.
- z ∈ [0, 2053] nm, non-uniform. Fine 0.5nm spacing spans **z=[25,58]nm only**; below that, only coarse points at z=0, 2, 9.67, 17.33, 25. This finely-resolved band brackets run3's declared interfaces at 30/40/43/53nm but **does not** finely resolve run3's shallow 0→2nm Si-cap interface.
- **run3's own material_id field** (read via h5py off `sige_afm_tip.h5`, cell centroids binned by z) transitions at z≈0, 2, 30(31.25 centroid), 40, 43(44.25 centroid), 53(55 centroid) → exactly matches the 6-layer stack documented in `calibrate_afm_tip_vs_leah.py`'s docstring (Si 0-2 / SiGe 2-30 / Si 30-40 / SiGe 40-43 / Si 43-53 / SiGe buffer 53-2053). **The FEM mesh's actual material layout matches the script's stated geometry.**

**VTK bottom face (z=2053)**: every one of 3721 points = exactly **-4.400000000000008 V** (min=max=mean). Confirmed independently.

**VTK top face (z=0)**: NOT flat. Center (x=0,y=0) = **exactly -1.235000 V**. Falls off smoothly and radially-symmetrically: -0.593V at r=15nm, -0.249V at r=30nm, -0.163V at r=45nm, down to ≈-0.060V by r≥120nm, -0.0483V at the corner (150,150). This is a sharp, localized, radially-symmetric feature centered at the origin — not a uniform plane, not noise.

**VTK z=35nm and z=48nm slices** (run3's Probe1/Probe2 depths): smooth, continuous, radially symmetric, NO step/discontinuity between small radius and periphery (e.g. z=35: -0.302V at r=0 down to -0.253V at r=30nm, gradual). No evidence of a hard Dirichlet disk boundary at either depth in the VTK data.

**run3 parameters** (`run_results.json`): AFM tip Dirichlet = **+1.0 V**, bottom Dirichlet = **-4.4 V** (chosen specifically to match this VTK, per script docstring — see below), Probe1/Probe2 = `null` voltage (natural/Neumann, not Dirichlet), eps_si=11.7, eps_sige=12.0, eps_air=1.0, lx=ly=300nm, tip_gap=10nm, tip_radius=10nm, cone_height=160nm. run1 used a different (coarser, larger-gap) tip geometry but the **same** +1.0V tip / -4.4V bottom convention — an earlier, less-refined attempt at the same target.

**`calibrate_afm_tip_vs_leah.py`'s own docstring states explicitly**: "*Nothing about Leah's actual tip geometry, gap, or voltage is known — only the effect it left in the VTK (a footprint on z=0 peaking at -1.235V at the center, decaying to -0.06V by r=150nm, on a -4.4V bottom).*" This is the prior author's own admission that tip voltage/geometry/gap are **all unconstrained free parameters being swept**, not independently known values from Leah.

**A distinct, older `basePotential3d.vtk`** exists in `masqe_comparison_bundle.zip/GlobalDevice/` (different file: 500×500nm domain, different byte size) whose companion `baseSurfacePotential.dat` is **uniformly -11.29V across the entire surface** (no localized feature at all), and whose top/bottom BCs are documented in `compare_basepotential_vs_masqe.py`'s docstring as "top=-1V, bottom=-12V" — i.e. **flat plate values, no gate/disk/tip active**. `VALIDATION_LADDER_NEXT_STEPS.md` (same repo, same MaSQE collaboration, disk-gate benchmark) independently and explicitly states: *"DOLFINx no-gate (σ only) vs MaSQE `basePotential3d.vtk`"* = **"background vs background"**, and that a **separate** file `totalPotential3d.vtk` (never received — listed under "What Is Still Needed") is what would contain the gate-active perturbation. This is a real, internally-consistent, previously-established naming convention in this exact collaboration: **`basePotential` = no perturbing element active; `totalPotential` = perturber included.**

---

## Reasonable inferences

- The `-4.4V` bottom-BC match between run3 and the VTK is **not** independent confirmation of a shared BVP — `calibrate_afm_tip_vs_leah.py`'s docstring says -4.4V was *chosen* specifically because it's this VTK's own bottom value. So "the bottom matches" is true by construction, not evidence the two cases otherwise agree.
- Units: the VTK's range [-4.4, 0.16] is consistent with Volts under this convention, but this is inferred only from the numeric coincidence with our own chosen bottom voltage — no explicit unit label exists in the VTK header or any note.
- The sharp, radially-symmetric -1.235V peak at exactly (x=0,y=0) is geometrically consistent with *something* localized sitting on-axis above the sample — an AFM tip is a plausible candidate given its shape, but a fixed non-swept device feature (dot, patch charge, permanently-present near-field probe) that MaSQE's own team would still classify under "base" is equally plausible and cannot be ruled out from the field data alone.
- The absence of any Dirichlet-like discontinuity at z=35/48nm is weak evidence that whatever produced the top peak is *not* being reinforced by an actively-biased structure at those two probe depths in the VTK — mildly consistent with run3's own passive/Neumann treatment there, but not proof either.

## Unresolved unknowns

- **Whether `basePotential3d(1).vtk` includes an AFM tip's contribution at all.** No documentation of Leah's tip voltage, geometry, position, or gap was found anywhere searched.
- eps_r assumptions in Leah's simulation — undocumented.
- Whether Probe1/Probe2 (z=35/48) were modeled by Leah as floating conductors, grounded, or simply absent (not even present as a distinct region) — no discontinuity was found, but that's also consistent with them not existing in her mesh at all.
- Why at least 4 differently-sized files are all named `basePotential3d.vtk`/`basePotential3d(1).vtk` across Downloads and the zip bundle (`~/Downloads/basePotential3d.vtk` 6.65MB, `~/Downloads/basePotential3d.vtk 2` 6.65MB, `~/Downloads/basePotential3d(1).vtk` 14.6MB — the one used here — and `GlobalDevice/basePotential3d.vtk` 26.1MB in the zip). These are evidently different simulation runs/geometries (different domain sizes, different bottom voltages: -12V vs -4.4V) reused under the same filename over time. This makes it unsafe to assume the "basePotential = no-perturber" convention established for the 500×500nm/-12V file mechanically carries over to this 300×300nm/-4.4V file — but it is the only documented precedent for what this MaSQE team means by "basePotential," and it points the wrong direction for run3's current assumption.

## Verdict

**Genuinely indeterminate, with a serious unresolved contradiction that should be treated as the blocking issue before any further calibration.**

The single most important open question: **does `basePotential3d(1).vtk` include the AFM tip's contribution?** This audit could not confirm it either way from the data or any available documentation. Two pieces of evidence pull in opposite directions:
1. The field itself has a sharp, localized, tip-shaped feature at the origin — suggestive of a tip being present.
2. This exact MaSQE collaboration has an established, explicitly documented naming convention elsewhere in this same repo (`VALIDATION_LADDER_NEXT_STEPS.md`, `compare_basepotential_vs_masqe.py`) where `basePotential` specifically means "background, no perturbing element" and a separate `totalPotential` file is needed for the perturbed case. `calibrate_afm_tip_vs_leah.py` was written *assuming* the opposite (that the peak IS the tip) without documented confirmation from Leah.

If (2) is the correct reading, no amount of tip voltage/geometry/gap calibration against this file can ever succeed, because the file may contain zero tip contribution — the entire ~1V shallow-depth discrepancy could simply be "run3 has a tip and the VTK doesn't." **Before further geometry tuning, this needs to be resolved with Leah directly** (does `basePotential3d(1).vtk` include the tip, and if so what tip voltage/geometry did she use) — no file on this machine settles it.
