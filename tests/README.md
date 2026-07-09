# Verification Test Suite

All tests run inside Docker via `./run_dolfinx.sh`. Each prints a results table and optionally writes XDMF/H5 output for ParaView inspection. Run everything in order with the bash runner, or pick individual tests.

---

## Run commands

```bash
# ── Tier 0: infrastructure (no Docker needed) ──────────────────────────────
python3 tests/test_mesh_feasibility.py                      # always run first

# ── Tier 1: core verification ladder (must all pass before MaSQE compare) ──
./run_dolfinx.sh tests/test_tag_integrity.py            --write-xdmf
./run_dolfinx.sh tests/test_h_convergence_manufactured.py   --quick --write-xdmf --save-csv
./run_dolfinx.sh tests/test_p_convergence_manufactured.py   --quick --write-xdmf --save-csv
./run_dolfinx.sh tests/test_bc_comparison_point_charge.py   --quick --write-xdmf --save-csv

# ── Tier 2: periodic/FFT and gate geometry ─────────────────────────────────
./run_dolfinx.sh tests/test_bc_comparison_neutral_periodic.py --quick --write-xdmf --save-csv
./run_dolfinx.sh tests/test_fft_vs_fem.py               --quick --write-xdmf --save-csv
./run_dolfinx.sh tests/test_square_gate_h_convergence.py    --quick --write-xdmf --save-csv

# ── Tier 3: refinement economy and device quantities ──────────────────────
./run_dolfinx.sh tests/test_four_square_refinement_box.py   --quick --write-xdmf --save-csv
./run_dolfinx.sh tests/test_probe_stability.py              --quick --write-xdmf --save-csv

# ── Run everything at once ────────────────────────────────────────────────
bash tests/run_all_verification.sh              # --quick (default)
bash tests/run_all_verification.sh --full       # expensive, run on workstation
```

---

## What each test proves

| Test | What it proves | Quick? | XDMF output |
|------|---------------|--------|-------------|
| `test_mesh_feasibility` | Uniform 3D mesh cost — rejects insane h values before wasting time | instant | no |
| `test_tag_integrity` | gmsh tags for gates/walls/volume are present and correctly sized | fast | yes |
| `test_h_convergence_manufactured` | P1 FEM gives rate≈2.0 in L2 on a smooth problem (FEM math is correct) | ~1 min | yes |
| `test_p_convergence_manufactured` | p=1→2 gives ≥10× error drop (spectral convergence on smooth problem) | ~1 min | yes |
| `test_bc_comparison_point_charge` | Zero-wall BC stays flat ~1e-2V; Coulomb BC converges with h | ~3 min | yes |
| `test_bc_comparison_neutral_periodic` | Periodic FEM and FFT agree for charge-neutral (dipole) source | ~3 min | yes |
| `test_fft_vs_fem` | FFT floors at ~1.4e-3V (smearing); FEM-CoulombBC converges past it | ~5 min | yes |
| `test_square_gate_h_convergence` | FEM converges vs analytic Davies half-space formula for rectangular gate | ~5 min | yes |
| `test_four_square_refinement_box` | RefinementBox cases D/E match fine-uniform B accuracy with fewer DOFs | ~10 min | yes |
| `test_probe_stability` | Device quantities (dot minima, barriers, gate crosstalk) are converged | ~10 min | yes + CSV slices |

---

## Output structure

Each test writes to its own subdirectory by default:

```
tests/test_h_convergence_manufactured/output/
    h_conv_n16_p1.xdmf   + .h5    ← solution field
    h_conv_n32_p1.xdmf   + .h5
    h_convergence_results.csv

tests/test_four_square_refinement_box/output/
    four_gate_A_h20nm.xdmf + .h5
    four_gate_B_h10nm.xdmf + .h5
    four_gate_D_h20_fine10nm.xdmf + .h5
    four_gate_results.csv

tests/test_probe_stability/output/
    probe_A_h20nm.xdmf + .h5
    z40nm_slice_A.csv               ← scatter: x,y,phi at QW plane
    probe_stability_results.csv
    probe_stability_quantities.csv
...
```

When run via `run_all_verification.sh`, all outputs go to:
```
results/verification_<TIMESTAMP>/
    summary.txt              ← PASS/FAIL for each test
    test_h_convergence.log
    h_convergence/           ← XDMF + CSV for that test
    ...
```

---

## Feasibility summary (device box 300×300×150 nm)

| h (nm) | P1 DOFs | Status |
|--------|---------|--------|
| 20 | ~2,300 | OK — laptop |
| 10 | ~15,000 | OK — laptop |
| 5 | ~115,000 | OK — laptop |
| 2 | ~1.7M | Aggressive — workstation |
| 1 | ~13.7M | Reject for uniform mesh |
| 0.5 | ~109M | Reject |
| ≤0.1 | >10B | Reject |

**Use refinement boxes** to get h=1–2nm accuracy near gates without uniform fine meshing.

---

## Must-pass before MaSQE comparison

1. `test_tag_integrity` — PASS (all gate tags present)
2. `test_h_convergence_manufactured` — PASS (rate≈2.0)
3. `test_bc_comparison_point_charge` — Coulomb BC error decreasing with h
4. `test_fft_vs_fem` — FEM-CoulombBC error at fine h < error at coarse h

If any of these fail, do **not** proceed to MaSQE delta comparison. Fix the solver first.

---

## Diagnostic only (do not need to pass for MaSQE)

- `test_p_convergence_manufactured` — informational; plateau at p≥2 on fine mesh is expected
- `test_bc_comparison_neutral_periodic` — periodic/FFT agreement, only relevant if using periodic BC
- `test_square_gate_h_convergence` — useful for gate-geometry accuracy but edge singularities limit rate
- `test_probe_stability` — device quantities; use to confirm mesh is converged before reporting physics
