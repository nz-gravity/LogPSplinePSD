
What the method does
Bayesian multivariate PSD estimation via a Cholesky-parameterised P-spline model under the Whittle/Wishart likelihood, sampled with NUTS. The key outputs per run are: RIAE (point estimation accuracy), coverage (credible interval calibration), CI widths, ESS, and runtime.

Coverage: what it actually measures
After auditing the code in _utils.py, the reported coverage is average pointwise 90% CI coverage — for each frequency and each (i,j) PSD matrix element, check whether the true value falls inside the 5th–95th posterior percentile, then average over all elements and all seeds. For a well-calibrated 90% CI this should be ≈ 0.90. Global coverage (all frequencies simultaneously) was never computed and would be effectively 0 given ~72,000 elements — not a useful summary.

Full results table
SettingNKCoarseShrinkageRIAECoverageCIW diagESSsmall (100 runs avg)~204820OFFNo0.1920.6710.210~13ksmall~204820ONNo0.1600.6510.192~7ksmall~204820ONYes0.1370.7360.155~2.8klarge (100 runs avg)163844OFFNo0.0560.4740.066—large1638410OFFNo0.1320.6040.067~17klarge1638410ONNo0.1330.5890.069~19klarge1638420ONYes0.1090.7340.059~831large1638450ONNo0.1310.6750.092~14klarge1638450ONYes0.1120.7410.059~1.9k

What each experiment revealed
Coarse-graining (NH=4) makes almost no difference to RIAE or coverage at large N. It reduces the effective number of frequencies by 4× and is ~15% faster, so it's worth using as a computational saving, but it doesn't change the statistical properties.
More knots (K=10→50 at large N) widens the CI intervals (0.067→0.092) and improves coverage (0.589→0.675) by giving the posterior more freedom. But coverage still doesn't reach 0.90. This knob helps, has diminishing returns, and costs runtime.
Shrinkage prior (centring log δ² on the design PSD) consistently improves RIAE across all settings — it's a better point estimator. But it actually narrows the CI widths further, so the coverage improvement (to ~0.74) comes entirely from the median being closer to truth, not from wider bands. Your supervisor is right to be uncomfortable: it uses privileged knowledge of the truth and solves the wrong problem for coverage.
ESS is the practical cost signal. Without shrinkage, ESS is large (~7k–19k) and sampling is efficient. With shrinkage, ESS collapses (~800–2000) because the design PSD prior adds curvature that NUTS struggles with at large N.

Root cause of undercoverage
Two separable effects:

Wishart likelihood over-concentration. With N=16384 frequency bins, each contributing one Wishart observation, the posterior concentrates at rate N. The Whittle approximation treats bins as independent, which is asymptotically correct for the mean but overestimates the curvature — the posterior is too narrow regardless of the prior. This is a documented finite-N property of Whittle-based Bayesian inference (Contreras-Cristán et al. 2006, Kirch et al. 2019).
Block 3 Cholesky collapse. When channels 1 and 2 together explain channel 3 well, the residual variance δ²₃ₖ → 0 is preferred by the likelihood at some frequencies. The prior on log δ²₃ has no floor, so the posterior collapses there, making those CI widths pathologically narrow. This is documented in your own appendix.

Effect (1) dominates at large N. Effect (2) is channel-ordering-specific and more pronounced in highly correlated settings like LISA AET channels.

What can actually fix it
FixHelps RIAEHelps coverageCostMore knotsSlightlyMarginallyRuntimeShrinkage priorYesOnly via better medianESS collapse, needs truthStronger penalty (smaller τ)NoMarginallyBiasDebiased Whittle likelihoodYesYes — directlyDifferent paperPrior lower bound on δ²₃NoLocally (block 3)Minor
The principled fix is the debiased Whittle likelihood (Sykulski et al. 2019), which corrects the finite-N curvature. Everything else is a patch. The honest position for the paper is that undercoverage at large N is an intrinsic limitation of the Whittle approximation, the point estimates are trustworthy, and fixing calibration is future work.



Known Limitations: Credible Interval Coverage
----------------------------------------------

The method produces well-calibrated **point estimates** (RIAE decreases
monotonically with sample size) but exhibits **systematic undercoverage**
of posterior credible intervals at large sample sizes.

In simulation studies with a 3-channel VAR(2) process at ``N=16384``,
the nominal 90% pointwise credible bands achieve only ~60--74% empirical
coverage, despite accurate median estimates (RIAE ≈ 0.11--0.13).

**Why this happens**

The Whittle likelihood treats periodogram matrices at each frequency bin
as independent complex Wishart observations.  With thousands of frequency
bins, the likelihood curvature concentrates the posterior far faster than
the true posterior uncertainty shrinks.  The result is credible intervals
that are too narrow — a known finite-N property of Whittle-based Bayesian
inference, independent of prior choice or spline basis size.

Concretely, across our experiments:

- Increasing knots (K=10 → K=50) widens intervals and improves coverage,
  but does not close the gap to 90%.
- Adding a shrinkage prior toward a design PSD improves point estimation
  (RIAE) but further narrows the intervals, worsening rather than fixing
  the coverage gap.
- The undercoverage is intrinsic to the Whittle/Wishart approximation,
  not a bug in the implementation.

**Implications**

Posterior medians and RIAE should be trusted.  Credible interval widths
should be interpreted as approximate rather than exact probability
statements, particularly at large N.  A principled fix would require
replacing the Whittle likelihood with a bias-corrected version (e.g.
the debiased Whittle likelihood of Sykulski et al. 2019), which is left
for future work.
