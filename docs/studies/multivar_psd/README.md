
# Coverage: What It Actually Measures

After auditing the code in `_utils.py`, the reported coverage is average
pointwise 90% CI coverage:

- For each frequency and each `(i, j)` PSD matrix element, check whether the
  true value falls inside the 5th-95th posterior percentile.
- Then average over all elements and all seeds.

For a well-calibrated 90% CI this should be about 0.90. Global coverage (all
frequencies simultaneously) was never computed and would be effectively 0 given
about 72,000 elements, so it is not a useful summary.

## Full Results Table

| Setting | N | K | Coarse | Shrinkage | RIAE | Coverage | CIW diag | ESS |
| --- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |
| small (100 runs avg) | ~2048 | 20 | OFF | No | 0.192 | 0.671 | 0.210 | ~13k |
| small | ~2048 | 20 | ON | No | 0.160 | 0.651 | 0.192 | ~7k |
| small | ~2048 | 20 | ON | Yes | 0.137 | 0.736 | 0.155 | ~2.8k |
| large (100 runs avg) | 16384 | 4 | OFF | No | 0.056 | 0.474 | 0.066 | - |
| large | 16384 | 10 | OFF | No | 0.132 | 0.604 | 0.067 | ~17k |
| large | 16384 | 10 | ON | No | 0.133 | 0.589 | 0.069 | ~19k |
| large | 16384 | 20 | ON | Yes | 0.109 | 0.734 | 0.059 | ~831 |
| large | 16384 | 50 | ON | No | 0.131 | 0.675 | 0.092 | ~14k |
| large | 16384 | 50 | ON | Yes | 0.112 | 0.741 | 0.059 | ~1.9k |

## What Each Experiment Revealed

- Coarse-graining (`NH=4`) makes almost no difference to RIAE or coverage at
  large `N`. It reduces the effective number of frequencies by 4x and is about
  15% faster, so it is worth using as a computational saving, but it does not
  change the statistical properties.
- More knots (`K=10` to `K=50` at large `N`) widen the CI intervals (`0.067` to
  `0.092`) and improve coverage (`0.589` to `0.675`) by giving the posterior
  more freedom. Coverage still does not reach 0.90. This helps, has diminishing
  returns, and costs runtime.
- Shrinkage prior (centering `log delta^2` on the design PSD) consistently
  improves RIAE across all settings. But it also narrows CI widths further, so
  the coverage improvement (to about `0.74`) comes entirely from the median
  being closer to truth, not from wider bands. This uses privileged knowledge
  of truth and solves the wrong problem for coverage.
- ESS is the practical cost signal. Without shrinkage, ESS is large (about
  7k-19k) and sampling is efficient. With shrinkage, ESS collapses (about
  800-2000) because the design PSD prior adds curvature that NUTS struggles
  with at large `N`.

## Root Cause of Undercoverage

Two separable effects:

1. Wishart likelihood over-concentration: With `N=16384` frequency bins, each
   contributing one Wishart observation, the posterior concentrates at rate
   `N`. The Whittle approximation treats bins as independent, which is
   asymptotically correct for the mean but overestimates curvature. The
   posterior is too narrow regardless of prior choice. This is a documented
   finite-`N` property of Whittle-based Bayesian inference (Contreras-Cristan et
   al. 2006, Kirch et al. 2019).
2. Block 3 Cholesky collapse: When channels 1 and 2 together explain channel 3
   well, the residual variance `delta^2_3k -> 0` is preferred by the likelihood
   at some frequencies. The prior on `log delta^2_3` has no floor, so the
   posterior collapses there and those CI widths become pathologically narrow.
   This is documented in the appendix.

Effect (1) dominates at large `N`. Effect (2) is channel-ordering specific and
more pronounced in highly correlated settings such as LISA AET channels.

## What Can Actually Fix It

| Fix | Helps RIAE | Helps Coverage | Cost |
| --- | :---: | :---: | --- |
| More knots | Slightly | Marginally | Runtime |
| Shrinkage prior | Yes | Only via better median | ESS collapse, needs truth |
| Stronger penalty (smaller tau) | No | Marginally | Bias |
| Debiased Whittle likelihood | Yes | Yes, directly | Different paper |
| Prior lower bound on `delta^2_3` | No | Locally (block 3) | Minor |

The principled fix is the debiased Whittle likelihood (Sykulski et al. 2019),
which corrects finite-`N` curvature. Everything else is a patch. The honest
position for the paper is that undercoverage at large `N` is an intrinsic
limitation of the Whittle approximation, point estimates are trustworthy, and
fixing calibration is future work.

## Known Limitations: Credible Interval Coverage

The method produces well-calibrated **point estimates** (RIAE decreases
monotonically with sample size) but exhibits **systematic undercoverage** of
posterior credible intervals at large sample sizes.

In simulation studies with a 3-channel VAR(2) process at `N=16384`, the nominal
90% pointwise credible bands achieve only about 60-74% empirical coverage,
despite accurate median estimates (RIAE about 0.11-0.13).

### Why This Happens

The Whittle likelihood treats periodogram matrices at each frequency bin as
independent complex Wishart observations. With thousands of frequency bins, the
likelihood curvature concentrates the posterior far faster than the true
posterior uncertainty shrinks. The result is credible intervals that are too
narrow, a known finite-`N` property of Whittle-based Bayesian inference,
independent of prior choice or spline basis size.

Concretely, across our experiments:

- Increasing knots (`K=10` to `K=50`) widens intervals and improves coverage,
  but does not close the gap to 90%.
- Adding a shrinkage prior toward a design PSD improves point estimation
  (RIAE) but further narrows intervals, worsening rather than fixing the
  coverage gap.
- The undercoverage is intrinsic to the Whittle/Wishart approximation, not a
  bug in the implementation.

### Implications

Posterior medians and RIAE should be trusted. Credible interval widths should
be interpreted as approximate rather than exact probability statements,
particularly at large `N`. A principled fix would require replacing the Whittle
likelihood with a bias-corrected version (e.g. the debiased Whittle likelihood
of Sykulski et al. 2019), which is left for future work.

## GPS Penalty Experiment — Abandoned (March 2026)

A replacement for `scikit-fda` using `scipy.BSpline` + phantom knots + the
Li & Cao (2022) `D_m^T D_m` general difference penalty was trialled and
**reverted** after a 100-seed study showed consistent coverage regression.
See `gps_penalty_postmortem.md` (repo root) for the full post-mortem.

Summary of results (N=2048, Nb=4, K=20, rect window, 99–100 seeds):

| Configuration                          | Coverage       | RIAE           |
|----------------------------------------|----------------|----------------|
| **scikit-fda baseline (rect)**         | **0.873 ± 0.041** | 0.150 ± 0.015 |
| GPS + phantom knots                    | 0.811 ± 0.059  | 0.150 ± 0.015  |
| GPS + clamped knots                    | 0.819 ± 0.059  | 0.151 ± 0.015  |

Root cause: GPS `D^T D` is a ≤1.5% approximation of scikit-fda's exact integral
`∫ B_i''B_j'' dx` for non-uniform (density-based) knots. This is sufficient to
shift posterior φ and systematically narrow credible intervals. Reverted in
commit `6fe73b8`.
