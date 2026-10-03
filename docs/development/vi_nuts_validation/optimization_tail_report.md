# Fixed exact time-varying target: optimization follow-up

Run date: 2026-10-02 (Pacific/Auckland). This experiment asks whether the unchanged diagonal guide can be fitted repeatably and stably before attributing remaining error to its shape.

Completed 3/3 planned fits. Settings within all predeclared late-drift and seed-repeatability screens: **none**. These are development screens, not convergence proofs or posterior validation.

## Frozen target and reference

Observations, counts, physical coordinates, 8×6 cubic basis, eigenbasis prior, roughness-scale priors, likelihood, noncentered parameterization and float64 precision are unchanged. Inputs were loaded from the previous exact control; no data generation or new NUTS inference occurred. The previously accepted four-chain reference is copied with SHA256 identities, and its accepted screen remains conditional evidence rather than a convergence proof.

Target fingerprint: `1e7a3be31733307e19296b3a4230829b16065b39f9ac7c405db465e46ae23bdb`. Rebuilt log joint agreed with 128 archived unconstrained density evaluations to maximum absolute difference 0. Numerical basis/penalty arrays match exactly. The extension reproduces the selected initial experiment's 40000-step learned parameters with maximum differences [0.0, 0.0, 0.0]. This extension was selected after inspecting the initial 18-fit experiment and remains exploratory. Input/reference file hashes are in frozen_contract.json.

## Optimization protocol

One fixed diagonal guide; Adam and global gradient clipping at 1 are unchanged. Schedules: 1000-step warmup from 0.001 to 0.01, then cosine decay to 0.0001 at step 40000, followed by a constant 0.0001 tail through 80000. Training particles: [8]; seeds: [7101, 7102, 7103]. Early stopping is disabled. Checkpoints: [1000, 5000, 15000, 25000, 30000, 35000, 40000, 50000, 60000, 70000, 80000] actual updates. Equal update budgets have different particle work; timings are reported separately.

Evaluation particles are independent of training particles: 256 per evaluation ×8 fixed keys, with independent evaluations on three other keys. This reduces objective noise compared with the previous 32×4 audit. Checkpoint comparisons use 4096 full joint guide draws and the same draw key, feature definitions, event thresholds, MMD transforms and bandwidths as the earlier study. Common random numbers reduce comparison noise without changing the fitted guide. All quantities and key counts are saved.

Optimization screens require both 60k→70k and 70k→80k intervals in all seeds, plus all three final seed pairs: all 10 feature means and all 50 unconstrained latent means change by ≤0.1 reference SD; their SD changes are ≤5%; declared event probability changes are ≤0.01; paired negative-ELBO changes are ≤0.2 plus twice their paired MCSE. Interval-width changes are measured separately. Reference uncertainty affects the SD scale; it is not counted as another independent replicate.

## Stability and repeatability

| Schedule | Training particles | Max late mean drift / ref SD | Max late relative SD drift | Max latent mean drift / ref SD | Max seed mean spread / ref SD | Max seed relative SD spread | Late pairs within screens | Seed pairs within screens |
|---|---:|---:|---:|---:|---:|---:|---|---|
| warmup_cosine_tail | 8 | 0.118 | 0.00914 | 0.204 | 0.104 | 0.00848 | 1/6 | 2/3 |

All maxima include roughness scales, contrasts and D distributions, not only plotted spectrum points. Detailed paired objective changes/MCSE, feature identities, latent SD drift, interval-width drift, probabilities and failed flags are in optimization_summary.json. Constant indicators carry an unresolved-tail status and cannot validate tail probabilities.

## Remaining NUTS discrepancies and importance proposals

At 80,000 steps, ranges cover the three optimization seeds. Agreement with NUTS and usefulness for importance sampling remain separate from optimizer stability.

| Schedule | Particles | Negative ELBO range | Max spectral mean error / ref SD | Median spectral SD ratio range | Max all-feature mean error / ref SD | Functional MMD² range / NUTS baseline | Native k range | Minimum raw/smoothed ESS fraction | Median fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup_cosine_tail | 8 | -847.984–-847.969 | 0.1 | 1.18–1.19 | 0.147 | 0.0063–0.00649/0.000559 | 0.485–0.942 | 0.00451/0.0156 | 15.8 |

The unchanged MMD calculation uses a disjoint reference subset to train one whitening transform for both methods, three predeclared RBF bandwidths and 512 matched draws. NUTS-versus-NUTS/VI-versus-VI baselines are descriptive; there are no IID permutation p-values. Reference autocorrelation and MCSE remain available in the frozen reference record. Repeated final PSIS uses 4096 independent guide particles ×3 seeds, separately from plotting and objective evaluations; weight ESS is not MCMC ESS. No reweighted posterior was produced.

## Interpretation and limitations

A decaying learning rate can make consecutive checkpoints move little while leaving the optimizer in different seed-dependent locations. A flat objective, a stable last checkpoint or an improved mean therefore does not answer repeatability alone. The independent objective evaluations and all seed pairs are retained to expose this distinction.

This experiment varies two optimization controls jointly in a small factorial design. It does not isolate irreducible diagonal-family error unless a setting becomes stable and repeatable. No new parameterization, full-covariance fit, flow, transform change, predictive adequacy test or calibration study was run. Development seeds were reused deliberately for matched diagnosis; success here would still need untouched-seed confirmation before a validated protocol is claimed.

Artifacts: `/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-optimization-exact-tv-tail`. Total sum of per-job workflow times: 70.5 seconds; medians in the table cover fit initialization/compilation, optimization, checkpoint evaluations, guide draws and final diagnostics. Additional reconstruction/comparison work is saved per checkpoint. Process startup and freezing/report costs are separate. Full joint packed checkpoint draws, final constrained posterior draws and numerical learned guide parameters are retained.

Reproduce with `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/optimize.py --reference <saved-exact-target-directory> --out <new-directory>`. Analyze saved fits with `PYTHONPATH=src JAX_ENABLE_X64=true python examples/vi_nuts_validation/optimization_report.py --out <artifact-directory>`.

## Verification

All 134 repository tests passed with float64 and NumPyro 0.22.0, including five slow scientific regression tests and the scheduled/multi-particle analytic Gaussian contract. Ruff and whitespace checks passed on the edited files. No inference job failed. The initial preflight batch-axis error was repaired before inference; its log is retained in the primary artifact directory. Implementation checks do not validate the fitted posterior.
