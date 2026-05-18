# Validation Ladder — Next Steps

## Comparison Mode Statement

| Stage | What we compare | Mode | Status |
|-------|----------------|------|--------|
| Stage 1 | DOLFINx no-gate (σ only) vs MaSQE `basePotential3d.vtk` | **background vs background** | ✅ Done — <0.003% outside QW |
| Stage 2 | DOLFINx Δφ (gate_active − background) vs MaSQE Δφ (totalPotential − basePotential) | **delta vs delta** | ⏳ Ready — needs `totalPotential3d.vtk` |

"background + gate vs background" is **not** the right framing. The gate adds a perturbation; the meaningful comparison is always the perturbation (delta) against MaSQE's perturbation, not the total field.

---

## Case Family Definitions

### Family 1 — True Blank
**Physical meaning:** Zero applied voltage everywhere. No gate, no bottom contact bias, no surface charge. This is the reference for "what does the FEM solution look like with no physics at all." Used to verify the solver itself (should give φ ≈ 0 everywhere for a uniform dielectric stack).

| File basename | Geometry | h (nm) |
|--------------|----------|--------|
| `blank_exact_tet_h6` | Exact device box (Lx=700, Ly=700, Lz=305 nm) | 6 |
| `blank_larger_tet_h6` | Larger box (Lx=1000, Ly=1000, Lz=400 nm) | 6 |

**BCs:** `--skip_disk_bc`, V_gate not set, V_bottom=0 (or floating — no bottom BC).
**Delta pairing:** pos1_p12 uses blank as background (no +12 V biased bg exists).

---

### Family 2 — Biased Background
**Physical meaning:** Bottom contact at V_bottom = −12 V, no gate, no surface charge. This is the electrostatic environment *without* the disk gate active. It is the correct background for the signed (−1 V gate) delta comparison.

| File basename | Geometry | h (nm) |
|--------------|----------|--------|
| `biased_bg_exact_tet_h6` | Exact device box | 6 |
| `biased_bg_larger_tet_h6` | Larger box | 6 |

**BCs:** `--skip_disk_bc`, V_bottom = −12 V, no sigma.
**Delta pairing:** neg1_neg12 − biased_bg.

---

### Family 3 — Positive Gate-Active
**Physical meaning:** Gate at V_gate = +1 V, bottom at V_bottom = +12 V. Tests the positive-bias regime. No corresponding biased background exists at +12 V, so the delta is taken against blank (Family 1).

| File basename | Geometry | h (nm) |
|--------------|----------|--------|
| `pos1_p12_exact_tet_h6` | Exact device box | 6 |
| `pos1_p12_larger_tet_h6` | Larger box | 6 |

**BCs:** Gate active (no `--skip_disk_bc`), V_gate = +1.0 V, V_bottom = +12.0 V.
**Delta pairing:** pos1_p12 − blank.

---

### Family 4 — Signed Gate-Active (Current Benchmark)
**Physical meaning:** Gate at V_gate = −1 V, bottom at V_bottom = −12 V. This matches the MaSQE `totalPotential3d.vtk` configuration. The signed convention is important: the gate depletes the 2DEG.

| File basename | Geometry | h (nm) |
|--------------|----------|--------|
| `neg1_neg12_exact_tet_h6` | Exact device box | 6 |
| `neg1_neg12_larger_tet_h6` | Larger box | 6 |

**BCs:** Gate active, V_gate = −1.0 V, V_bottom = −12.0 V.
**Delta pairing:** neg1_neg12 − biased_bg.

---

### Family 5 — Delta Cases
**Physical meaning:** Δφ = φ_gate_active − φ_background. This is the gate perturbation — the additional potential induced by turning on the gate on top of the background bias. This is what should be compared against MaSQE's `totalPotential − basePotential`.

| Delta | Gate case | Background | Pairing rationale |
|-------|-----------|------------|-------------------|
| `delta_neg` | neg1_neg12 | biased_bg | Same V_bottom = −12 V; isolated gate effect |
| `delta_pos` | pos1_p12 | blank | No +12 V biased bg exists; blank is the reference |

Scripts: `analysis/compare_delta_cases.py`, `analysis/compare_gate_vs_background.py`

---

### Family 6 — Point-Charge Benchmark
**Physical meaning:** A Gaussian charge blob in free space/dielectric, compared against the analytic 1/r Coulomb solution (corrected for ε). This is a **separate physics validation** — it tests the volumetric Poisson solve (ρ ≠ 0 term), not the gate/surface-charge BCs. It does not connect to the disk-gate geometry.

Script: `analysis/pointcharge_benchmark.py`
Output dir: `outputs/point_charge_sanity/`

---

## File Naming Convention

```
results/<basename>/<basename>.xdmf   ← FEM solution (DOLFINx XDMF/H5)
results/<basename>/<basename>.h5
results/<basename>/<basename>_config.json
results/<basename>/<basename>_summary.txt
```

`<basename>` encodes: `{gate_sign}_{V_bottom_sign}_{mesh_type}_h{h_nm}`, e.g.:
- `blank_exact_tet_h6`
- `biased_bg_exact_tet_h6`
- `neg1_neg12_exact_tet_h6`

---

## QW Band-Offset Zone

z = 50–55 nm is the sSi quantum well. MaSQE applies band-edge offsets here that shift the electrostatic potential by ~10–20 mV locally. The pure Poisson solver (DOLFINx) does not include these offsets. Expect 1–2% mismatch in this zone — this is **not a Poisson bug**. All comparison scripts flag this zone in orange.

---

## What Is Still Needed

1. **`totalPotential3d.vtk`** from Leah — the gate-active MaSQE output.
   - Once available: drop into `MASQUE-Comparison/` and run `./run_trackB.sh`
   - Script will auto-detect it and run the Stage 2 delta comparison.

2. **Stage 2 comparison command** (once file is available):
   ```bash
   docker run --rm -it -v "$PWD":/app -w /app dolfinx/dolfinx:nightly \
     sh -lc 'pip install -q h5py scipy 2>/dev/null; \
             /dolfinx-env/bin/python3 -u analysis/compare_gate_vs_background.py \
               --xdmf-gate results/neg1_neg12_exact_tet_h6/neg1_neg12_exact_tet_h6.xdmf \
               --xdmf-bg   results/biased_bg_exact_tet_h6/biased_bg_exact_tet_h6.xdmf \
               --label-gate "neg1 gate" --label-bg "biased_bg" \
               --outdir outputs/gate_vs_bg_neg'
   ```

3. **MaSQE delta extraction** — needed in `compare_two_states.py`:
   ```
   delta_masque = totalPotential3d.vtk − basePotential3d.vtk
   delta_dolfinx = neg1_neg12_exact_tet_h6 − biased_bg_exact_tet_h6
   ```
   Compare these two deltas on the same interpolated grid.

---

## Run Order

```bash
# Stage 1 — already done, but to re-run:
./run_blank_cases.sh
./run_biased_background_cases.sh

# Stage 2 gate-active solves:
./run_neg1_neg12_cases.sh    # signed convention (matches MaSQE)
./run_pos1_cases.sh          # positive convention

# Analysis — background comparison:
docker run ... analysis/compare_background_cases.py \
  --xdmf-a results/blank_exact_tet_h6/blank_exact_tet_h6.xdmf \
  --xdmf-b results/biased_bg_exact_tet_h6/biased_bg_exact_tet_h6.xdmf \
  --label-a "Blank" --label-b "Biased BG" \
  --outdir outputs/background_compare

# Analysis — delta comparison (gate perturbation):
docker run ... analysis/compare_gate_vs_background.py \
  --xdmf-gate results/neg1_neg12_exact_tet_h6/neg1_neg12_exact_tet_h6.xdmf \
  --xdmf-bg   results/biased_bg_exact_tet_h6/biased_bg_exact_tet_h6.xdmf \
  --outdir outputs/gate_vs_bg_neg

# Point-charge sanity (independent):
./run_point_charge_sanity.sh
```
