# Synthetic validation of `corrected_comparison_metrics.py`

Test module: `notebooks/afm_vs_leah_compare/run3/scripts/test_corrected_metrics.py`
(run with `python3 test_corrected_metrics.py`; exits 0 iff all checks pass).

Neither `corrected_comparison_metrics.py` nor `src/poisson/vtk_xdmf_compare.py` was
modified. All synthetic cases use hand-built numpy point/value arrays and call
`compute_slice_row` (per-slice metrics) and `interpolate_case_onto_ref` /
`FieldData` (interpolation) directly. Every test states the analytically-expected
value *before* comparing to the measured value. Tolerance used throughout:
`atol`/`rtol` ~1e-9 for exact-formula checks (identity, constant offset, linear
scaling — these have closed-form exact answers), and looser tolerances (~5%, or
a few grid-spacings) only where the underlying method is itself a discrete
approximation (FWHM half-max crossing via linear interpolation on a finite grid).

## Result: 13/13 checks PASS. No bugs found in the synthetic-test scope.

| # | Test | Expected | Measured | Verdict |
|---|------|----------|----------|---------|
| 1 | Identical fields (`sin(x)cos(y)`, real spatial variation) | signed_err_mean=0, rmse=0, E_L2=0, E_range=0, masked_relerr_mean=0, shape_rmse=0, pearson_r=1.0 | all exactly 0, pearson_r=1.000000 | PASS |
| 1b | Degenerate constant field (both fields = 3.14159 everywhere) | pearson_r=NaN (zero variance, guarded by `_safe_pearsonr`), shape_rmse=0 | pearson_r=nan, shape_rmse=0 | PASS |
| 2 | Constant offset C=0.5V, spatially-varying ref (`3+sin(x)cos(y)`) | signed_err_mean=C=0.5 exactly, rmse=\|C\|=0.5 exactly, E_range=rmse/global_range=0.250628, **shape_rmse=0** (offset must not leak into shape metric), pearson_r=1.0 | signed_err_mean=0.500000, rmse=0.500000, E_range=0.250628, shape_rmse=1.3e-16, pearson_r=1.000000000 | PASS |
| 3 | Multiplicative scaling, k=2.0 and k=-1.0 | E_L2=\|k-1\| exactly (1.0 and 2.0), rmse=\|k-1\|·rms(ref), pearson_r=+1.0 (k=2.0) / **-1.0** (k=-1.0) exactly | E_L2=1.000000000 / 2.000000000, pearson_r=+1.000000000 / -1.000000000 | PASS (confirms a measured -1.0 correlation in the real analysis genuinely means inverted shape, not a bug) |
| 4 | Lateral shift dx=1.5, dy=1.0 (Gaussian bump) | peak_loc_dx=1.5, peak_loc_dy=1.0, peak_loc_dist=1.802776 exactly (grid-aligned); shape_rmse>0 nontrivial; pearson_r degrades below 0.99 | peak_loc_dx=1.500000, peak_loc_dy=1.000000, peak_loc_dist=1.802776, shape_rmse=0.364781, pearson_r=0.368 | PASS |
| 5a | Reflection of a *symmetric* radial bump (x→-x is a no-op for it) | fields identical, max\|diff\|=0, shape_rmse=0, pearson_r=1.0 | all exact | PASS |
| 5b | Reflection of an *asymmetric* bump (peak at x0=+1) | peak_loc_dx=-2·x0=-2.0, peak_loc_dist=2.0, real shape mismatch (shape_rmse>0, pearson_r<0.9) | peak_loc_dx=-2.000000, peak_loc_dist=2.000000, shape_rmse=0.389, pearson_r=0.282 | PASS |
| 6 | Partial domain overlap (case covers x≤1.0 of a x∈[-5,5] ref grid, 3 z-layers) | n_fallback=120 (hand-counted: 4 of 10 unique x-grid-values > 1.0, ×3 z-layers ×10 y-values = 120/300), frac=0.4000 | n_fallback=120, frac=0.4000, fallback_mask array identical to hand-built ground-truth mask | PASS |
| 7 | Ref crosses zero (linear ramp `vtk=x`), constant abs error eps=0.01 | n_masked_excluded=41 (hand-count of \|x\|<0.044 on the 41×41 grid), masked_relerr_max ≤ eps/threshold=0.2273 (bounded); independent unmasked pct_err=100·eps/\|ref\| diverges (1%, 10%, 100%, 1e4%, 1e6% as \|ref\|→1,0.1,...,1e-6) | n_masked_excluded=41, masked_relerr_max=0.0400 (well within bound); unmasked pct_err confirmed monotonically unbounded | PASS |
| 8 | FWHM width difference, Gaussian σ_vtk=1.0 vs σ_fem=2.0 | Feature **is** implemented (`_fwhm_along_line` + `fwhm_vtk_nm`/`fwhm_fem_nm`/`fwhm_diff_nm` columns). Analytic FWHM=2.3548·σ → 2.3548 and 4.7096, diff=+2.3548 | fwhm_vtk_nm=2.3041, fwhm_fem_nm=4.4454, diff=+2.1414 (within ~5%/grid-spacing tolerance of the discrete half-max-crossing estimator) | PASS |
| CSV | Independent spot-check, z_requested=10nm (z_actual=9.6667nm) | 17 columns recomputed from scratch via `read_vtk_reference`/`read_xdmf_case`/`interpolate_case_onto_ref`/`nearest_z_mask` + raw numpy only (no `ccm` helpers reused) | All 17 columns match CSV to 1e-5 rtol (e.g. E_L2=7.91373, E_range=0.23989, pearson_r=-0.906864) | PASS |
| CSV | Independent spot-check, z_requested=100nm (z_actual=97.9000nm) | same as above | All 17 columns match CSV to 1e-5 rtol (e.g. E_L2=13.5316, E_range=0.227654, pearson_r=-0.994281) | PASS |

## Summary

All 13 synthetic checks and both independent CSV spot-checks (z=10nm, z=100nm —
distinct from the prior agent's z=0/z=35/z≈1993nm checks) pass. No bugs were
found in `corrected_comparison_metrics.py` or the primitives it reuses from
`vtk_xdmf_compare.py` within the scope of these tests:

- Signed vs. absolute error are correctly separated everywhere.
- The mean-subtraction in the shape metrics (`shape_rmse`, `shape_rel_l2`,
  `pearson_r`) genuinely removes constant voltage offsets and is scale/sign
  sensitive in exactly the way needed to trust the real analysis's measured
  negative correlations as genuine shape inversion, not a computation error.
- `nearest_fallback_frac` / `fallback_mask` from `interpolate_case_onto_ref`
  correctly and exactly identify out-of-hull (extrapolated) points.
- The `MASK_THRESHOLD_V=0.044` masking correctly excludes near-zero-crossing
  points and prevents the well-known percent-error blow-up from contaminating
  the masked statistics; the module correctly does *not* expose an unmasked
  percent-error column at all (only the masked `masked_relerr_*` fields),
  consistent with its documented philosophy that `E_L2`/`E_range` are primary.
- The FWHM-based lateral-width-difference feature is implemented and reports
  the correct sign and roughly correct magnitude (discrete half-max-crossing
  estimation, not exact, but sane).

This gives strong independent confidence that the `E_L2`, `E_range`, `rmse`,
`shape_rmse`, and `pearson_r` numbers already reported for the real run3-vs-VTK
comparison (`depth_metrics_corrected.csv`, `summary_metrics_corrected.json`)
are computed correctly by this module.
