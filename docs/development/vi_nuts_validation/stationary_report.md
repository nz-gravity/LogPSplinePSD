# Stationary AR(4): VI versus NUTS, stages 0–3

No tested recipe completes the full declared validation milestone. At 40k, full covariance reproduces the fixed-smoothing PSD and band-power uncertainty across all three seeds, but the two-interval stability gate is not met. Inferred smoothing remains unexecuted.

This report describes executed development work on data seed 60101, not the proposed protocol or a confirmation study. `audit.md`, `protocol.json`, `dry_run_manifest.json`, numerical `frozen_target/arrays.npz` and the append-only attempt ledger precede the fits. The native frequency-only centered model, likelihood, data, prior and basis are identical between algorithms within each target. Fixed and hierarchical physical target IDs are distinct.

| Target / guide | Reference | Primary mean error / SD; SD ratios, all seeds | Stability | Functional MMD² / NUTS baseline | Native / packed k; packed smoothed ESS fraction | VI workflow seconds per seed | Limitation |
|---|---|---|---|---|---|---|---|
| fixed / diag | accepted (repair_1) | max 0.0578; 0.931–1.39 | outside_screen; point pass False | 0.01–0.0121 / 0.000372–0.000372 | 0.907–1.17 / 0.966–1.04; 0.0038–0.00598 | 15.5–15.7 | 0/3 seeds pass final primary point screens; status counts {'within_screen': 62, 'outside_screen': 9, 'mc_precision_limited': 1, 'unavailable': 0} |
| fixed / mvn | accepted (repair_1) | max 0.0426; 0.988–1.02 | outside_screen; point pass False | 0.000141–0.000427 / 0.000372–0.000372 | 0.119–0.243 / 0.154–0.359; 0.921–0.93 | 16.1–16.9 | 3/3 seeds pass final primary point screens; status counts {'within_screen': 72, 'outside_screen': 0, 'mc_precision_limited': 0, 'unavailable': 0} |
| hierarchical / diag | unexecuted | unavailable | unexecuted | unavailable | unavailable | unavailable | unexecuted_prior_stage_gate |
| hierarchical / mvn | unexecuted | unavailable | unexecuted | unavailable | unavailable | unavailable | unexecuted_prior_stage_gate |

Primary features are nine log PSD values plus three log band powers, and physical/log roughness when inferred. Raw bands are secondary. All final comparisons use 16,384 fresh full joint guide draws. Checkpoints use 4,096 common-key draws; their precision limits are retained. `comparison.csv` records every frozen scalar, checkpoint, seed, coefficients, 50/90/95% width ratio, Wasserstein and combined MC uncertainty. Mean and SD screen statuses use a two-MCSE uncertainty interval; borderline estimates remain `mc_precision_limited`. Reference MCSE estimates account for chain autocorrelation.

## References, attempts and exact scope

- fixed/initial: execution_failed; four planned chains; 0 retained draws per chain; 0.72 s. Error retained: `TypeError("init_strategy.<locals>.strategy() missing 1 required positional argument: 'site'")`.
- fixed/repair_1: accepted_screen; four planned chains; 4000 retained draws per chain; 24.21 s. Divergences 0, cap hits 0, maximum rank split R-hat 1.00132, minimum bulk/tail ESS 27671/11244, minimum BFMI 0.979, primary mean MCSE/SD max 0.00604.

Completed VI jobs: 6. Executed seeds: fixed/diag/7101, fixed/diag/7102, fixed/diag/7103, fixed/mvn/7101, fixed/mvn/7102, fixed/mvn/7103. Every completed fit used 40,000 actual updates and eight full-data training particles. Objective evaluation uses 256 particles×8 fixed keys and three separate keys; PSIS uses 4,096 joint draws×3 diagnostic seeds. Numeric guide checkpoints exist at 5k, 10k, 20k, 30k, 35k and 40k, saved before expensive analysis. These contain guide parameters, not optimizer state.

The fixed initial attempt failed before sampling because the experimental independent-initialization callback was passed as a function rather than the partial expected by NumPyro. That failure and traceback are retained. A deterministic initialization test verifies the corrected callback and four distinct starts. The single authorized fixed-target repair used its already frozen 2k warmup/4k draws per chain, acceptance 0.99, depth 10 and dense mass; no observations or target components changed. `repair_reason.json` was written before launch. This exhausts the fixed-target reference repair budget. The original planned settings were 1k/2k, acceptance 0.95. No original posterior is claimed.

The first controller reached the end of the six fixed fits before its offline summary function was available and failed at report generation. Its log is retained separately; subsequent summarization reads saved draws only and does not refit or replace inference attempts.

No hierarchical reference or fit ran when the fixed stability gate failed. Seeds 60102–60106 were neither generated nor inspected. No parameterisation sweep, time-varying investigation, new likelihood, flows, 80k tail, optimizer-state continuation, reweighting or extra diagnostic precision collection ran. The routine suite includes existing small stationary and time-varying regression fits, as required when changing shared inference.
The runner also made one separately keyed 16,384-draw packed-guide collection per fit (seed 8401) for nonredundant parameter MMD; only the first 512+512 samples enter its comparison/baseline. This collection is part of the executed diagnostic recipe, rather than a precision extension or an optimizer run. Posterior features use the distinct fresh draws from fit_vi; checkpoint collections use seed 8301. These actual draw counts and seeds are recorded in executed_draw_design.json alongside the original pre-fit protocol.

## Posterior agreement and stability

- fixed/diag: dense-field standardized log-mean RMS 0.0146–0.018; worst-location mean discrepancy 0.0436–0.0443; dense SD range 0.908–1.12. Physical quadrature is normalized over the frozen 257-node [.02,.48] grid. The full fit domain is shown separately in the PSD overlay.
  Complete coefficient means: max discrepancy 0.0251 reference SD; SD ratios 0.592–0.916. Primary final point screen passes 0/3 seeds. Geometry/correlation matrices and compact nonredundant latent MMD are retained per seed.
  Across all six late interval comparisons and three final seed pairs, worst mean drift 0.1439 reference SD and worst SD drift 5.20%. All paired negative-ELBO changes, their MCSE and independent-key objective estimates are saved; a flat objective alone does not certify posterior stability.
- fixed/mvn: dense-field standardized log-mean RMS 0.0129–0.0156; worst-location mean discrepancy 0.0332–0.0381; dense SD range 0.968–1.02. Physical quadrature is normalized over the frozen 257-node [.02,.48] grid. The full fit domain is shown separately in the PSD overlay.
  Complete coefficient means: max discrepancy 0.0294 reference SD; SD ratios 0.972–1.02. Primary final point screen passes 3/3 seeds. Geometry/correlation matrices and compact nonredundant latent MMD are retained per seed.
  Across all six late interval comparisons and three final seed pairs, worst mean drift 0.1138 reference SD and worst SD drift 5.27%. All paired negative-ELBO changes, their MCSE and independent-key objective estimates are saved; a flat objective alone does not certify posterior stability.

The hierarchy gate requires one fit passing both 30k→35k and 35k→40k point screens in the declared primary physical features and every coefficient, without resolved outside-screen drift. The same thresholds were retained after seeing the results. All seed pairs and both late intervals are assessed separately; end-point seed agreement does not erase earlier movement.
- diag, 30k→35k: max mean drift across seeds 0.124–0.144; max relative SD drift 0.0417–0.052; statuses ['outside_screen', 'outside_screen', 'outside_screen'].
- diag, 35k→40k: max mean drift across seeds 0.0498–0.0593; max relative SD drift 0.0178–0.0253; statuses ['within_screen', 'within_screen', 'within_screen'].
- mvn, 30k→35k: max mean drift across seeds 0.109–0.114; max relative SD drift 0.0364–0.0527; statuses ['outside_screen', 'outside_screen', 'outside_screen'].
- mvn, 35k→40k: max mean drift across seeds 0.0418–0.0653; max relative SD drift 0.0201–0.0248; statuses ['within_screen', 'within_screen', 'within_screen'].

At 40k the full-covariance Gaussian has close selected-functional and complete coefficient agreement with the diagnosed conditional reference. Diagonal VI can match pointwise spectral uncertainty while missing integrated band uncertainty and coefficient covariance. Its measured discrepancy is specific to this guide, coordinates and recipe; it is not a proof of the irreducibly best diagonal approximation. Fixed-smoothing agreement does not establish inferred-smoothing agreement or explain a hierarchical funnel.

## Joint proposal diagnostics and interpretation

Native NumPyro k and the tested packed-density adapter use the complete prior/factor/model density, guide density and exactly one support Jacobian in common unconstrained coordinates. Fixed sigma is excluded from packing. All raw/smoothed ESS values, ESS fractions, maximum weights, native/adapter status and nonfinite cases remain in complete.json/guide metadata. A k≥0.7 warns about importance sampling; it is not the primary mean/SD screen, and low k does not prove equality. No weights were used to alter these posterior summaries.
Native NumPyro splits each diagnostic seed into per-particle guide traces; the packed adapter samples one vectorized posterior collection. They therefore use different realized particles even for the same seed, so their finite-sample k values need not coincide. Both repeated ranges are shown, with adapter weights/ESS labelled separately; no discrepancy is hidden by averaging k.

Functional and nonredundant latent MMD use 512 matched draws, whitening trained on a disjoint reference subset and the frozen three bandwidths. VI–VI and NUTS–NUTS baselines are retained, including negative unbiased values. These are descriptive comparisons with autocorrelated reference draws, without IID p-values, universal thresholds or an equality claim.

## Cost, verification and reproduction

NumPyro 0.22.0; JAX 0.9.1; x64 True; backend cpu. Repository SHA 63b5c7a10d8bd4a3dfed8f1df05b02112321fd1b; dirty-source hashes and installed versions recorded for each job. No dependency update was required. Frozen native basis and penalty are float64; reference/guide parameter dtypes are recorded.

Inherited schedule: `{'name': 'warmup_cosine', 'type': 'warmup_cosine', 'peak_lr': 0.01, 'initial_lr': 0.001, 'warmup_steps': 1000, 'end_lr': 0.0001}` from `/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-optimization-exact-tv/optimization.toml` with its saved hash. Adam default beta/epsilon choices and clipping norm 1 are unchanged. Initialization/first compiled update, optimization, checkpoint objective/PSIS diagnostics, reconstruction and I/O timings are retained in each fit; workflow seconds in the table include diagnostic work and cannot imply speed superiority. Peak fit memory MiB 880–1.01e+03; current artifact bytes 145,835,738. Offline field/report timing is separate.

Runnable commands from the audited worktree:

```bash
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary.py prepare
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary.py all
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary.py summarize
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python -m pytest -q
```

`all` preserves completed jobs and failed attempts and applies the frozen repair/gate budget; it never resumes optimizer state. `summarize` is offline. The complete routine suite passed 139 tests with 29 warnings in 70.89 seconds, including marked slow regressions. Artifact verification also passed: all six saved guides reproduce the stored log joint and log guide exactly on checked saved points; all 36 parameter checkpoints round-trip; 1,890 comparison rows contain MC uncertainty. Tests, retained initial failures, and visual review are recorded in verification.json and logs. Figures: PSD median/90% intervals with truth only as context, standardized mean difference, SD ratio, explicit unexecuted roughness panel, and late/seed stability. Mean/SD curve shading represents two pointwise MCSE, not simultaneous confidence bands. Native complete reconstruction is checked against exp(Bc); scalar likelihood, conditional gradients, fixed packing, AR stationarity/sign/normalization, independent starts and persistence-before-analysis failure are tested.

## Next supported experiment, proposed only

The supported next experiment is a bounded refinement of full-covariance optimization on this same fixed-smoothing target, using the accepted reference and all three optimization seeds. Predeclare a short low-rate tail beyond 40k and two new late checkpoints; preserve the inherited schedule prefix and rerun it because guide checkpoints do not store optimizer state. Check the same coefficient/primary mean and SD drifts and paired objectives. Do not infer an intrinsic guide-shape error from the earlier movement. Proceed to the matched hierarchical target only after that stability milestone, without demanding diagonal VI pass.

This study stops here. It does not establish whitening, coverage, SBC, model adequacy, robustness across AR records, a speed advantage, inferred smoothing, or TV validation. The truth overlay is descriptive, not calibration.

The conditional NUTS coefficient geometry shows adjacent correlations from -0.681 to -0.384; coefficient 15–16 correlation is -0.624. Diagonal guides cannot retain those correlations. The recorded band SD inflation and reduced coefficient SDs are consistent with that missing covariance, while the full-covariance fit closely reproduces them. This is a measured covariance diagnostic, not a proof of an optimizer-independent family optimum.

## Figures

![Full-domain PSD](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-ar4/figures/psd.png)

![Dense log-PSD mean discrepancy](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-ar4/figures/mean_error.png)

![Dense log-PSD SD ratios](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-ar4/figures/sd_ratio.png)

![Roughness stage status](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-ar4/figures/roughness.png)

![Checkpoint and seed stability](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-ar4/figures/stability.png)
