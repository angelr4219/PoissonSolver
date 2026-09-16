# Polarity diagnostic: floating-gate runs 5/6/7 vs Leah's VTK at z=35nm

Read-only analysis. Imports `read_vtk_reference`, `read_xdmf_case`, `interpolate_case_onto_ref`, `nearest_z_mask` from `src/poisson/vtk_xdmf_compare.py` and `nearest_unique_z`, `_safe_pearsonr` from `run3/scripts/corrected_comparison_metrics.py` -- neither source file modified.

VTK reference: `/Users/angelramirez/Downloads/basePotential3d(1).vtk` -- 636291 points, value range [-4.4000, 0.1598] V.

z=35nm slice: requested z=35.0, actual VTK z=35.0, n_points=3721.

## Runs compared

| run | V_tip (V) | V_bottom (V) | Gate1/Gate2 | V_g1* (V) | V_g2* (V) | sanity Q1 resid | sanity Q2 resid |
|---|---|---|---|---|---|---|---|
| run5 | +0.0 | -4.4 | floating (self-consistent) | -1.695358 | -1.701723 | -2.177e-13 | -2.054e-15 |
| run6 | +1.0 | -4.4 | floating (self-consistent) | -1.080666 | -1.088479 | -2.213e-13 | 1.604e-14 |
| run7 | -1.0 | -4.4 | floating (self-consistent) | -2.310049 | -2.314968 | -2.458e-13 | 1.496e-13 |

All three sanity-check residual net charges (flux re-integrated on the FINAL combined phi field, independent of the superposition algebra) are ~1e-13 to 1e-15 -- effectively exact zero net charge on both gates in every case, confirming the floating-conductor implementation is correct.

## 1. Curvature sign check at z=35nm

Gate1's disk (radius 10nm) sits exactly at z=35nm, so the FEM potential is PINNED FLAT (equipotential) over the whole disk -- neither a single grid-point-at-origin test nor a simple two-point (plateau vs. just-outside) trend test is safe here, because the FEM profile turns out to be NON-MONOTONIC in r (see table below). The robust classifier used here is the Pearson shape correlation over the WHOLE z=35nm slice (n=3721 points) -- the same convention already used throughout this project (e.g. `corrected_comparison_metrics.py`) -- backed up by a full radial-binned profile table for qualitative context.

**Leah's VTK radial profile** (shape, mean-subtracted, binned by r [nm]):
| [0,5) | [5,10) | [10,15) | [15,20) | [20,30) | [30,40) | [40,50) | [50,70) | [70,100) | [100,130) | [130,160) | [160,190) | [190,220) |
| -0.1337 | -0.1308 | -0.1237 | -0.1148 | -0.0973 | -0.0751 | -0.0555 | -0.0315 | -0.0065 | 0.0093 | 0.0164 | 0.0219 | 0.0244 |

VTK rises **monotonically** from a minimum at the center all the way to the domain edge -> **DIP (monotonically rising outward from a central minimum, all the way to the domain edge -- classic smooth dip)**.

**FEM radial profiles** (shape, mean-subtracted, binned by r [nm]):

`run5` (V_tip=+0.0V): -0.0112 -0.0112 -0.0017 0.0131 0.0195 0.0207 0.0189 0.0141 0.0059 -0.0014 -0.0058 -0.0101 -0.0121
`run6` (V_tip=+1.0V): -0.0137 -0.0137 -0.0021 0.0160 0.0240 0.0254 0.0232 0.0173 0.0073 -0.0018 -0.0071 -0.0123 -0.0149
`run7` (V_tip=-1.0V): -0.0086 -0.0086 -0.0013 0.0101 0.0151 0.0160 0.0146 0.0109 0.0046 -0.0011 -0.0045 -0.0078 -0.0094

| run | V_tip (V) | center (r<5) | peak value | peak r-bin | far edge (r=190-220) | Pearson r vs VTK |
|---|---|---|---|---|---|---|
| run5 | +0.0 | -0.011183 | 0.0206627 | [30,40) | -0.0121185 | -0.837845 |
| run6 | +1.0 | -0.0137246 | 0.0253588 | [30,40) | -0.0148727 | -0.837845 |
| run7 | -1.0 | -0.00864139 | 0.0159666 | [30,40) | -0.00936429 | -0.837845 |

All three FEM cases show the SAME qualitative shape: flat near the gate (pinned by the floating-conductor boundary condition), rising to a local peak/halo bump around r=30-40nm, then falling and going NEGATIVE again past r~100nm. This is categorically different from Leah's smooth, monotonic, sign-definite dip -- and it is why the Pearson correlation is strongly NEGATIVE for all three: the FEM shape and VTK's shape trend in opposite directions over most of the domain (FEM peaks and falls back while VTK keeps rising).

Note: the Pearson r vs VTK is essentially IDENTICAL across all three tip voltages (0, +1, -1 V) despite their very different plateau values -- this is a direct, testable consequence of linearity: phi(V_tip) = V_tip*S_tip(x) + S_bottom(x) for two FIXED spatial patterns S_tip, S_bottom (the exact same fact verified numerically in section 2 below). At z=35nm, S_tip's in-plane shape happens to be very nearly proportional to S_bottom's shape (both dominated by the SAME geometric fringing-field pattern at the floating disk's edge, not by whatever is exciting it), so no linear combination of the two -- i.e. no choice of V_tip, including negative values -- can change the CORRELATION with Leah's VTK shape at this depth. Only genuinely different physics (e.g. Gate1 biased instead of floating, or different tip/gate geometry) could change that correlation's sign.

## 2. Linearity check: dphi_minus vs -dphi_plus

dphi_plus = phi(run6, V_tip=+1) - phi(run5, V_tip=0); dphi_minus = phi(run7, V_tip=-1) - phi(run5, V_tip=0). Linear (Laplace, no volume charge) electrostatics with V_bottom and both gate Dirichlet-equivalent boundary conditions otherwise fixed requires dphi_minus == -dphi_plus EXACTLY. This is also a correctness check on the floating-gate implementation: a bug there (e.g. gate voltages not truly linear in V_tip) would break this symmetry.

**At z=35nm slice** (n=3721 points):
- max |dphi_minus + dphi_plus| = 7.372831e-10 V
- mean |dphi_minus + dphi_plus| = 5.974386e-11 V
- max|dphi_plus| (scale) = 6.220899e-01 V
- relative (max resid / max|dphi_plus|) = 1.185e-09

**Over the full 3D VTK point cloud** (n=636291 points):
- max |dphi_minus + dphi_plus| = 2.376507e-09 V
- mean |dphi_minus + dphi_plus| = 3.685589e-11 V
- max|dphi_plus| (scale) = 6.585679e-01 V
- relative (max resid / max|dphi_plus|) = 3.609e-09

**Verdict**: PASSES the linearity requirement to solver/interpolation tolerance (residual is many orders of magnitude below the ~1-2V scale of dphi_plus/dphi_minus, consistent with FEM discretization + griddata interpolation noise, not a real asymmetry).

## 3. Plain-language verdict

- **run7 (V_tip = -1V)**: Pearson r = -0.8378 vs Leah's VTK shape. This does NOT match Leah's dip shape -- still anti-correlated.
- **run6 (V_tip = +1V)**: Pearson r = -0.8378 vs Leah's VTK shape. This does NOT match Leah's dip shape (still anti-correlated/bump-like, consistent with run3's earlier inert-gate finding).
- **run5 (V_tip = 0V)**: Pearson r = -0.8378 vs Leah's VTK shape. This isolates the floating-gate + bottom-gate contribution alone, with no tip bias at all -- and it is ALSO anti-correlated with Leah's dip.

**Plain-language answer**: NO -- flipping the tip voltage sign from +1V to -1V does NOT flip the shape to match Leah's dip. All three cases (V_tip = 0, +1, -1 V) give essentially the SAME Pearson correlation (r=-0.8378, -0.8378, -0.8378 respectively) against Leah's VTK shape at z=35nm -- all strongly negative (anti-correlated with the dip). This is a direct, provable consequence of linearity demonstrated in section 2: phi(V_tip) = V_tip * S_tip(x) + S_bottom(x) for two FIXED spatial patterns, and at z=35nm (right at Gate1's own surface) S_tip's in-plane shape is very nearly proportional to S_bottom's shape -- both are dominated by the SAME geometric fringing-field pattern set by the floating disk's edge, not by whichever boundary condition (tip or bottom) is actually driving it. Because of that near-proportionality, NO choice of V_tip (positive, negative, or zero) can change the correlation's sign at this specific depth/location -- tip polarity is therefore RULED OUT (not just unconfirmed, but mathematically incapable, at this depth/geometry) as the explanation for the shallow-depth shape mismatch with Leah's VTK, at least with Gate1/Gate2 modeled as floating conductors. Run3's earlier bump finding (inert/electrically-passive gates, correlation -0.97) and this floating-gate result (correlation ~-0.84 for all three tip polarities) point to the same qualitative conclusion: making the gates real floating conductors changed the correlation magnitude somewhat but did NOT fix the sign mismatch, and tip voltage sign cannot fix it either. The mismatch's source must lie elsewhere -- e.g. whether Leah's VTK includes a tip contribution at all (the still-unresolved question raised in `physical_case_audit.md`), her actual gate biasing/boundary assumptions, or geometry differences.
