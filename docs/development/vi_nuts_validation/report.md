# VI versus NUTS validation: stages A–C

Run date: 2026-10-02 (Pacific/Auckland). These are small-basis development experiments, not repeated-simulation calibration.

## What the experiment resolves

**Reference error:** accepted screens for exact_time_varying, raw_slow_m32; unresolved references for exact_stationary, raw_ar1_m8. Discrepancies against unresolved references remain descriptive.

The additional AR(1) tuning repair at acceptance .999/depth 12 has status failed_reference and 2 divergences. It is promoted only if all reference screens pass.

**Optimization error:** on the exact time-varying diagonal guide, seed 7101, fixed-seed negative ELBO changed from -845.87 at 5000 steps to -830.62 at 15000 (final evaluation MCSE 0.64). Increasing steps or adding covariance parameters does not establish optimizer convergence. The checkpoint table includes roughness scales and D losses as well as spectral features.

**Family/geometry restrictions:** richer covariance guides improve some spectral/MMD comparisons, but remaining optimization instability prevents isolating irreducible family error. Smooth raw-series spectral SD ratios are diag=1.40, lowrank:10=1.26, mvn=1.09. The measured spectrum bands are not uniformly too narrow.

For the smooth raw-series diagonal guide at seed 7101/15000 steps, the median original-coefficient marginal variance ratio is 0.640, while three selected log-spectrum variance ratios are 1.95, 2.38, 1.59. The saved diagonal/cross-covariance decomposition measures covariance cancellation; it does not attribute all differences solely to the guide family. The exact Gaussian anticorrelation example is separately tested analytically.

**Importance proposals:** 12/12 target/family combinations have at least one native k ≥.7 across optimization/diagnostic seeds. Joint weight ESS and repeated-seed stability are reported separately. No reweighted posterior is claimed.

**Model adequacy:** exact independent Gamma controls separate transform approximation from algorithm discrepancies. Raw-series transform dependence and predictive adequacy remain untested here. No whitening or calibration result is inferred from posterior comparisons.

A bounded follow-on should stabilize optimization on these same targets before estimating a guide-family accuracy limit. Unresolved reference geometry requires its own validated repair. New parameterization experiments were not executed.

## Reference health and scope

| Target | Reference status | Divergences | Max R-hat | Min bulk/tail ESS | Min BFMI | Depth hits | Max mean MCSE/SD |
|---|---|---:|---:|---:|---:|---:|---:|
| exact_stationary | failed_reference | 0 | 1.00205 | 1038/1916 | 0.242 | 1 | 0.032 |
| exact_time_varying | accepted_screen | 0 | 1.00324 | 1345/1143 | 0.812 | 0 | 0.032 |
| raw_ar1_m8 | failed_reference | 1 | 1.00111 | 4553/7229 | 0.831 | 0 | 0.015 |
| raw_slow_m32 | accepted_screen | 0 | 1.00218 | 2376/2834 | 0.848 | 0 | 0.022 |

Reference screening uses four independent initializations, rank-normalized split R-hat <1.01, bulk/tail ESS ≥400, no divergences/depth saturation, BFMI ≥0.3 and mean MCSE ≤0.05 posterior SD. Latent sites and declared functionals are checked together. These are screens, not convergence proofs. A failed screen remains a limitation in every associated comparison. Initial references and repairs are separate artifacts.

## Task-level posterior comparisons

At 15,000 actual optimization steps. Values aggregate three optimization seeds. Spectral features are declared log variances, integrated band powers and time contrasts; roughness scales and D distributions remain available per feature in comparison.json.

| Target | Guide | Max spectral mean error / NUTS SD | Median SD ratio | Median 95% width ratio | Max event probability difference | Functional MMD² / NUTS baseline | Latent MMD² / NUTS baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| exact_stationary | diag | 0.977 | 1.16 | 1.17 | unavailable | 0.268/0.00573 | 0.331/0.00428 |
| exact_stationary | lowrank:10 | 0.5 | 1.07 | 1.06 | unavailable | 0.162/0.00573 | 0.18/0.00428 |
| exact_stationary | mvn | 0.361 | 1.07 | 1.08 | unavailable | 0.143/0.00573 | 0.157/0.00428 |
| exact_time_varying | diag | 2.64 | 1.24 | 1.24 | 0.000732 | 0.303/0.000559 | 0.0956/0.000526 |
| exact_time_varying | lowrank:10 | 2.09 | 1.09 | 1.1 | 0.000732 | 0.0894/0.000559 | 0.0264/0.000526 |
| exact_time_varying | mvn | 2.54 | 1.07 | 1.07 | 0.000732 | 0.291/0.000559 | 0.0939/0.000526 |
| raw_ar1_m8 | diag | 1.47 | 1.31 | 1.26 | 0.418 | 0.0915/0.000669 | 0.027/0.000408 |
| raw_ar1_m8 | lowrank:10 | 1.42 | 1.12 | 1.11 | 0.409 | 0.0655/0.000669 | 0.0128/0.000408 |
| raw_ar1_m8 | mvn | 2.56 | 1.13 | 1.1 | 0.311 | 0.0827/0.000669 | 0.0194/0.000408 |
| raw_slow_m32 | diag | 1.71 | 1.4 | 1.41 | 0.00559 | 0.13/0.0015 | 0.0669/0.00082 |
| raw_slow_m32 | lowrank:10 | 1.47 | 1.26 | 1.25 | 0.0115 | 0.0345/0.0015 | 0.017/0.00082 |
| raw_slow_m32 | mvn | 1.63 | 1.09 | 1.11 | 0.0161 | 0.0228/0.0015 | 0.0162/0.00082 |

### Physical field errors

| Target | Guide | Max area-weighted RMS mean error / reference SD | Worst-region mean error / reference SD |
|---|---|---:|---:|
| exact_stationary | diag | 1.17 | 1.7 |
| exact_stationary | lowrank:10 | 0.446 | 0.753 |
| exact_stationary | mvn | 0.367 | 0.556 |
| exact_time_varying | diag | 2.01 | 3.09 |
| exact_time_varying | lowrank:10 | 1.58 | 2.5 |
| exact_time_varying | mvn | 1.85 | 3.17 |
| raw_ar1_m8 | diag | 1.22 | 2.88 |
| raw_ar1_m8 | lowrank:10 | 1.24 | 2.12 |
| raw_ar1_m8 | mvn | 1.4 | 2.65 |
| raw_slow_m32 | diag | 1.14 | 2.49 |
| raw_slow_m32 | lowrank:10 | 0.954 | 1.72 |
| raw_slow_m32 | mvn | 1.12 | 2.36 |

Field values use normalized physical trapezoid quadrature, all posterior draws and bounded frequency chunks. For tensor targets the declared region is time [.1,.9] × frequency [.08,.42]; field artifacts retain exact coordinates and worst-region locations. Matched MMD subset counts are 512; the actual bulk ESS of untransformed reference subset features is separately saved in fields.json and may be much smaller. The 9-dimensional stationary lowrank:10 guide uses a factor rank exceeding its dimension, so it does not impose a low-rank covariance restriction there.

Univariate Wasserstein distances, 50/90/95% interval ratios, event MCSE and correlations are saved for every feature/seed/checkpoint. Event differences must be judged against their function-specific MCSE; constant indicators are explicitly unresolved tails, not zero-MCSE certificates. Field analyses reconstruct all joint draws in frequency chunks on a declared physical grid, with normalized quadrature RMS and worst-region errors (fields.json). No spatial cells are treated as independent datasets.

MMD uses the same regularized reference whitening for both methods, trained on a disjoint first-quarter reference subset. Bandwidths are sqrt(feature dimension) times [0.5,1,2]; estimator is unbiased and excludes within-sample diagonals. Negative baseline estimates are preserved. All comparisons use 512 matched draws and report descriptive NUTS-versus-NUTS and VI-versus-VI baselines. Reference draws remain autocorrelated; there are no IID permutation p-values or universal MMD cutoffs.

## Joint importance proposal diagnostics

| Target | Guide | Native k range | Adapter k range | Min raw ESS fraction | Min smoothed ESS fraction | Median fit workflow seconds |
|---|---|---:|---:|---:|---:|---:|
| exact_stationary | diag | 0.811–1.12 | 0.837–1 | 0.00373 | 0.00756 | 7.46 |
| exact_stationary | lowrank:10 | 0.672–1.09 | 0.672–1.13 | 0.00137 | 0.00429 | 10.8 |
| exact_stationary | mvn | 0.709–1.17 | 0.682–1.02 | 0.000259 | 0.0058 | 9.24 |
| exact_time_varying | diag | 0.755–2.25 | 0.819–2.32 | 0.000278 | 0.000353 | 7.3 |
| exact_time_varying | lowrank:10 | 0.582–1.59 | 0.706–1.57 | 0.000835 | 0.000826 | 11.4 |
| exact_time_varying | mvn | 1.13–2.27 | 1.34–2.46 | 0.000256 | 0.000409 | 9.18 |
| raw_ar1_m8 | diag | 0.851–1.42 | 1–1.77 | 0.000287 | 0.000439 | 7.37 |
| raw_ar1_m8 | lowrank:10 | 0.69–1.45 | 0.712–1.51 | 0.000841 | 0.00131 | 11.9 |
| raw_ar1_m8 | mvn | 0.957–1.67 | 1.02–1.65 | 0.000335 | 0.000501 | 10.3 |
| raw_slow_m32 | diag | 1.15–1.6 | 1.08–1.74 | 0.000335 | 0.000472 | 7.25 |
| raw_slow_m32 | lowrank:10 | 0.832–1.24 | 0.744–1.29 | 0.000361 | 0.00116 | 12 |
| raw_slow_m32 | mvn | 0.895–1.73 | 0.734–1.83 | 0.000455 | 0.000434 | 9.64 |

Each diagnostic uses 4096 joint particles ×3 independent diagnostic seeds, in chunks of 128. NumPyro native PSIS and the packed-density ArviZ weight adapter use different draws, so finite-sample k values need not match. Both densities use the same unconstrained measure; NumPyro potential_energy includes all prior/likelihood factors and one support Jacobian. No densities over PSD pixels and no KDE substitute for the learned guide. Weight ESS is not MCMC ESS. No posterior reweighting was performed.

ArviZ Stats 1.0.0 negates its array input internally. A tested lower-tail identity selects and records the input sign; the adapter explicitly normalizes log weights. Exact matching densities are labeled constant_weights, with full ESS and undefined tail k, even when the native GPD routine returns a nonfinite value.

## Optimization audit

| Target | Guide | Max mean change 5000→15000 / NUTS SD | Max probability change 5000→15000 |
|---|---|---:|---:|
| exact_stationary | diag | 0.319 | unavailable |
| exact_stationary | lowrank:10 | 0.798 | unavailable |
| exact_stationary | mvn | 0.495 | unavailable |
| exact_time_varying | diag | 4.34 | 0.000732 |
| exact_time_varying | lowrank:10 | 2.22 | 0.000732 |
| exact_time_varying | mvn | 3.76 | 0.000488 |
| raw_ar1_m8 | diag | 2.73 | 0.537 |
| raw_ar1_m8 | lowrank:10 | 2.56 | 0.69 |
| raw_ar1_m8 | mvn | 2.31 | 0.432 |
| raw_slow_m32 | diag | 3.28 | 0.0227 |
| raw_slow_m32 | lowrank:10 | 2.29 | 0.00879 |
| raw_slow_m32 | mvn | 2.43 | 0.0168 |

All guide families use optimization seeds 7101/7102/7103 and checkpoints at 1000, 5000 and 15000 actual steps. Early stopping was disabled. Objectives use fixed evaluation seeds 8201–8204 and 32 particles; independent evaluations use diagnostic seeds 8101–8103. The legacy relative-loss stopping test is demonstrably sensitive to additive data constants. An opt-in paired objective/noise and location-stability rule is provided, while diagnostic collection alone preserves the existing stopping behavior. Flat objective values do not certify posterior accuracy.

## Costs, artifacts and unexecuted work

Executed 4 target comparisons and 36 VI fits, with three saved checkpoint comparisons per fit. Retained NUTS jobs: 7 (four initial references, two first repairs and one additional AR(1) repair in this run). Each target directory records the exact completed jobs and source SHA, dirty-content hash, versions, CPU device/backend, float64 parameter dtypes, model fingerprint and timings. Original failed references and execution failures are retained. NumPyro 0.22.0 was installed in an isolated worktree environment; the main checkout and its environment were preserved.

Fit workflow timings include initialization/compilation, optimization, checkpoint evaluations, guide draws and density diagnostics. Additional checkpoint reconstruction/comparison and offline field/report costs are saved separately. The historical remaining_optimization_seconds phase also includes checkpoint objective evaluations; it is not pure optimizer kernel time. Guide draw throughput is not used as accurate posterior throughput.

Exact-model controls generate powers from the intended spline surface using Gamma(nu/2, scale=2V). Raw-process comparisons reuse archived paired observations and the exact 8×6 cubic basis at fixed m=8 or m=32, with all priors, likelihoods and normalizations held fixed. No inference results are reused as four-chain references. Raw transform dependence/model adequacy remain unresolved.

Stage D predictive whitening/PIT/held-out adequacy and stage E SBC/repeated coverage were not run. The coefficient-whitening primitive is covered by analytic contract tests, but power observations cannot recover original coefficient signs/phases. Optimizer-state resumption, flow checkpoint reconstruction and large-basis recovery are not implemented or validated. The saved checkpoint is guide-only. No adaptive window selection, parameterization experiment, new flow or remote/OzSTAR execution occurred.

Bulk artifact directory: `/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-nuts-validation-v2`. Reproduction: `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/run.py --previous <prior-artifact-directory> --out <new-directory>`. Reports/field analysis: `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/summarize.py --out <artifact-directory> --fields`.

Official APIs checked against installed versions: [NumPyro PSIS](https://num.pyro.ai/en/stable/utilities.html#psis-diagnostic), [NumPyro 0.22.0 release](https://github.com/pyro-ppl/numpyro/releases/tag/0.22.0), [ArviZ array PSIS](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.base.array_stats.psislw.html).

## Verification

All 129 repository tests passed with float64 and NumPyro 0.22.0, including the five slow scientific regression tests. After the final stationary target-identity fix, all 23 stationary and blocked-diagnostic checks passed again. Ruff and whitespace checks passed on edited files. Five PSIS input-convention and Gaussian/reference contracts also passed against ArviZ Stats 1.3.3 in a temporary overlay; the benchmark used installed ArviZ Stats 1.0.0. Logs are saved alongside verification.json.

Two execution failures are retained separately: an initial tail-ESS API call before chain persistence (no chains saved from that trial), and NumPy scalar fingerprint serialization before the AR(1) fit. Both were repaired and rerun. References are now saved before diagnostic evaluation. Tests do not certify posterior accuracy, raw-process coverage or model adequacy.
