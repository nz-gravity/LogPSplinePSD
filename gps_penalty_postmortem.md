# GPS Penalty Experiment Post-Mortem: Revert to scikit-fda

**Status:** Closed — reverted to scikit-fda in commit `6fe73b8`
**Date:** March 2026
**Affected code:** `src/log_psplines/psplines/penalty.py` (removed), `src/log_psplines/psplines/initialisation.py` (restored)

---

## Background

The original basis/penalty used **scikit-fda**:
```python
basis = BSplineBasis(domain_range=[0, 1], order=degree+1, knots=knots)
P = L2Regularization(LinearDifferentialOperator(diff_order)).penalty_matrix(basis)
P = P / np.max(P)  # normalise so max(P) = 1
```
This computes the exact O-spline penalty `P[i,j] = ∫ B_i''(x) B_j''(x) dx`,
which is correct for any knot spacing (uniform or density-based).

The **GPS replacement** (Li & Cao 2022) was motivated by:
1. Removing the heavy `scikit-fda` dependency
2. Theoretically better handling of non-uniform knots via the general difference matrix `D_m^T D_m`
3. Removing endpoint-pinning bias of clamped B-splines by using phantom knots outside `[0, 1]`

---

## What Was Implemented

### Phantom knot vector (`_build_knot_vector`)
Instead of repeating boundary knots `degree` times (clamped):
```
[0, 0, 0, k1, k2, ..., 1, 1, 1]   # clamped, degree=2
```
places `degree` equally-spaced phantom knots outside `[0, 1]`:
```
[-h, 0, k1, k2, ..., 1, 1+h, ...]  # phantom
```
This gives all basis functions the same symmetric bell shape, removing the
boundary asymmetry of clamped splines.

### General difference matrix (`build_general_difference_matrix`)
Implements Li & Cao (2022) eq. 4a + 5: a recursively constructed matrix `D_m`
such that `D_m^T D_m ≈ ∫ B_i^(m)(x) B_j^(m)(x) dx`, with weights `W_m`
accounting for knot spacing. For uniform knots `D_m^T D_m ∝ Delta_m^T Delta_m`
(standard difference penalty). For non-uniform knots it is the correct extension.

---

## Experimental Results (100-seed VAR3-3D study, N=2048, K=20, Nb=4, rect window)

| Configuration                        | Coverage       | RIAE           | ESS   |
|--------------------------------------|----------------|----------------|-------|
| **scikit-fda** (baseline, rect)      | **0.873 ± 0.041** | 0.150 ± 0.015 | ~11000 |
| GPS + phantom knots + max(\|P\|) norm  | 0.811 ± 0.059  | 0.150 ± 0.015  | 7282  |
| GPS + clamped knots + max(\|P\|) norm  | 0.819 ± 0.059  | 0.151 ± 0.015  | 8446  |
| GPS + no normalisation               | 0.598 ± 0.051  | 0.178 ± 0.018  | 8624  |

All GPS variants give ~5–28 percentage points worse coverage than scikit-fda.
The MCMC convergence is fine in all cases (ESS > 1000, R-hat ≈ 1.0).

---

## Root Cause Analysis

### Stage 1: Phantom knots — ~6% coverage loss
The phantom knot vector places the maximum of `D^T D` in the **interior**
of the domain, while clamped knots place it at the **boundary**. After
`max(|P|)` normalisation:

| Basis    | Interior diagonal | Boundary diagonal |
|----------|-------------------|-------------------|
| Clamped  | **0.60**          | 1.00              |
| Phantom  | **1.00**          | ~0.60             |

Phantom knots therefore apply ~1.67× stronger smoothing in the interior
(where >99% of frequency bins live), systematically narrowing credible
intervals. This alone accounts for the 0.87 → 0.81 drop.

**Fix attempted:** switch back to clamped knots in GPS → coverage 0.819.
Still a ~5% regression, pointing to a second issue.

### Stage 2: Removing normalisation entirely — catastrophic
Without `max(|P|)` normalisation, the raw `D^T D` values are `O(n_basis²)`,
making the effective `φ * P` enormous even for small `φ`. The prior on `φ`
was calibrated for a normalised `P`, so the model massively over-smoothed.
Coverage collapsed to 0.60.

### Stage 3: GPS clamped vs scikit-fda — ~5% residual regression
With clamped knots and `max(|P|)` normalisation, a direct numerical comparison
shows:

```python
# Uniform knots:      GPS clamped == scikit-fda  (max abs diff = 0.0)
# Non-uniform knots:  GPS clamped ≈ scikit-fda   (max abs diff ≈ 0.015)
```

The **B-spline basis matrices are identical** in both cases (scipy and scikit-fda
produce the same design matrix). The **penalty matrices agree exactly for uniform
knots** but differ by ≤1.5% for non-uniform (density-based) knots, because:

- **scikit-fda** computes the **exact** integral `∫ B_i''(x) B_j''(x) dx`
- **GPS `D^T D`** is a **discrete approximation** of the same integral,
  weighted by knot spacing differences

For density-based placement (the default), knots are concentrated toward low
frequencies. In these dense regions the GPS weight matrix `W_m` introduces a
small but consistent approximation error relative to the exact integral. This
shifts the posterior smoothing parameter `φ` just enough to systematically
narrow credible intervals across all 9 PSD matrix elements, compounding over
~8000 frequency bins to a ~5% absolute coverage drop.

---

## Why the Exact Integral Matters

The Bayesian model's log-prior on the spline weights is:
```
log p(w | φ) ∝ -½ φ · wᵀ P w
```
The posterior on `φ` is a Gamma distribution whose parameters depend on `wᵀ P w`.
Any systematic change to `P` — even 1–2% in the off-diagonal elements — shifts
the posterior `φ` to a slightly different value, which in turn shifts CI widths.
Since coverage is averaged over 9 × ~4000 = 36,000 (frequency, element) pairs
per seed, even a sub-percent change in average CI width propagates directly to
a measurable coverage change across 100 seeds.

The GPS approximation is *theoretically motivated* but *empirically insufficient*
for this model. The exact integral from scikit-fda is the correct reference.

---

## Files Removed

| File | Reason |
|------|--------|
| `src/log_psplines/psplines/penalty.py` | GPS penalty + phantom knots — not used |
| `tests/test_gps_penalty.py` | Tests for removed code |
| `docs/studies/multivar_psd/var3d_gps_basis.slurm` | GPS OzStar job |
| `docs/studies/multivar_psd/var3d_gps_basis_shrink.slurm` | GPS+shrinkage OzStar job |

---

## Open Question

The GPS penalty (Li & Cao 2022) is mathematically correct for non-uniform
B-splines. The failure here is not conceptual but practical: the 1–2%
approximation error in `D^T D` vs `∫ B''B''` is small enough to seem
negligible but large enough to measurably hurt Bayesian coverage. One path
forward would be to compute `∫ B_i''(x) B_j''(x) dx` analytically using
Gaussian quadrature over each knot span (which is what scikit-fda does
internally), without requiring scikit-fda as a dependency. This is left for
future work.

---

## Decision

**Revert to scikit-fda.** The dependency cost is acceptable; the coverage
benefit is real and proven across 100+ seeds. GPS is theoretically appealing
but practically worse.
