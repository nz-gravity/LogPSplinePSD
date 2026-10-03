# Stationary full-covariance continuation: stages 0–2

Fixed stability milestone: **True**. Hierarchy exploratory unlock: **True**. These are distinct gates.

Hierarchical posterior agreement: **not established**. Optimization stability and seed repeatability are reported independently below. Matched-accuracy efficiency study justified: **False**; it has not been launched.

All three hierarchical fits pass late stability and final seed-pair checks, and the spectral/band/coefficient screens pass. Physical roughness SD ratios are 0.797–0.805; log-roughness SD ratios are 0.821–0.829. The two-MCSE upper bounds remain below 0.9: the smoothing uncertainty discrepancy is resolved, while measured late drift and seed differences remain within their screens. This does not establish an optimal Gaussian guide or prove a funnel. Low k does not rescue that posterior mismatch.

Cost: the six VI workflows took 24.69–25.31 s each, including fit, final joint diagnostics, persistence and offline posterior comparisons. The new hierarchical four-chain reference took 10.85 s through persistence and health/feature analysis. Fresh-process inclusive times and the full timing boundaries are given below. These are diagnostic workflow costs; they do not establish a matched-accuracy speed advantage. All three seeds remain visible. The read-only source reference and numerical observations/basis/penalty are hashed in continuation_protocol.json.

| Target / seed | Reference health / precision | Final primary agreement | Coefficients | Late stability | Final seed pairs | Joint proposal | Updates | Workflow seconds |
|---|---|---|---|---|---|---|---|---|
| fixed / 7101 | accepted_screen / within_screen | within_screen | within_screen | within_screen | within_screen | within_screen | 60000 | 24.69 |
| fixed / 7102 | accepted_screen / within_screen | within_screen | within_screen | within_screen | within_screen | within_screen | 60000 | 24.93 |
| fixed / 7103 | accepted_screen / within_screen | within_screen | within_screen | within_screen | within_screen | within_screen | 60000 | 25.31 |
| hierarchical / 7101 | accepted_screen / within_screen | outside_screen | within_screen | within_screen | within_screen | within_screen | 60000 | 24.84 |
| hierarchical / 7102 | accepted_screen / within_screen | outside_screen | within_screen | within_screen | within_screen | within_screen | 60000 | 25.16 |
| hierarchical / 7103 | accepted_screen / within_screen | outside_screen | within_screen | within_screen | within_screen | within_screen | 60000 | 25.30 |

The following table keeps the descriptive MMD baselines and three-seed PSIS ranges beside each outcome. Joint-proposal status uses the inherited warning screen; it does not imply posterior equality.

| Target / seed | Functional MMD² VI/NUTS | NUTS/NUTS | VI/VI | Latent MMD² | Native k range | Packed k range | Packed weight ESS fraction |
|---|---|---|---|---|---|---|---|
| fixed / 7101 | 0.000339104 | 0.000372146 | 0.000423152 | 0.000355199 | 0.163–0.269 | 0.268–0.332 | 0.926–0.93 |
| fixed / 7102 | -0.000244453 | 0.000372146 | -0.000219608 | 4.0432e-05 | 0.174–0.3 | 0.185–0.314 | 0.924–0.931 |
| fixed / 7103 | -0.000289147 | 0.000372146 | -0.000217013 | -7.41439e-05 | 0.146–0.413 | 0.163–0.329 | 0.928–0.932 |
| hierarchical / 7101 | -0.000237701 | 0.000688067 | -0.000292379 | -0.000205611 | 0.448–0.634 | 0.442–0.637 | 0.396–0.551 |
| hierarchical / 7102 | 0.000105171 | 0.000688067 | -0.000179536 | 3.52393e-05 | 0.511–0.556 | 0.446–0.687 | 0.364–0.542 |
| hierarchical / 7103 | 6.38858e-05 | 0.000688067 | -0.000134115 | 4.6181e-05 | 0.51–0.627 | 0.402–0.631 | 0.393–0.547 |

## Frozen target and provenance

The development record is the archived seed-60101 stationary AR(4), 4,096 samples at dt=1, coefficients [0.9, −0.8, 0.7, −0.6], innovation variance 0.4, stationary initialization and no realization-specific variance normalization. The native full-record rectangular Fourier observations contain 2,047 interior frequencies; DC and Nyquist are excluded. The fitted model has 30 centered frequency-only P-spline coefficients with the saved numerical second-derivative penalty, including its 1e-6 ridge and weak null directions. The conditional scale is the native HalfNormal(1.28) prior median, 0.8633468802509846; restoring the hierarchy samples that same scale prior. Observations, likelihood, basis, penalty and all other priors are identical within each VI/NUTS comparison.

Source artifacts: [/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-ar4](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-ar4). Physical target and coordinate IDs, all source hashes, schedule configuration and predeclared budgets are in [continuation_protocol.json](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/continuation_protocol.json). The fixed target and sampled-scale target have distinct IDs. The current density and coefficient gradients agree exactly when the hierarchy is conditioned at the archived scale, with full scale-dependent normalization retained.

Execution used the existing NumPyro 0.22.0, JAX/jaxlib 0.9.1, Optax 0.2.7 environment, CPU float64, four sequential NUTS chains and fresh inference state for each fit. The repository revision is 63b5c7a10d8bd4a3dfed8f1df05b02112321fd1b; dirty-source hashes, platform and cache/thread environment are recorded in provenance.json. Dependencies were not upgraded. No diagonal VI, data generation, parameterisation change, flow, new likelihood, benchmark or confirmation job ran.

## Executed optimization and endpoints

The original schedule function is evaluated with its original 40,000-update horizon (warmup 0.001→0.01 over 1k, cosine to 0.0001 by 40k). The tail is constant 1e-4 at optimizer indices 40000..59999. Each fit starts from the same data-based physical pilot and native guide scale 0.1, with fresh Adam/SVI state and the archived seed; the 40k state stays live through the tail. There is no optimizer resume from a guide checkpoint and no stretched cosine. Checkpointing/objective draws do not advance training RNG.

Fixed replay comparison occurs at actual update 40k before any tail updates. Parameter discrepancy, exact equality and numeric arrays are saved per seed. The old failed 30k→35k intervals remain in the source stability_fixed.json. New intervals are 40k→50k and 50k→60k (10k updates each); no unmeasured drift-per-update comparison is made with the old 5k intervals.

## Fixed posterior agreement, stability and repeatability

- Seed 7101: replay/packing prerequisite within_screen. 40k primary status within_screen, max mean error 0.0328 reference SD, SD ratios 0.988–1.01. 60k primary status within_screen, max mean error 0.0243, SD ratios 0.985–1.01.
  40k→50k: within_screen; worst mean drift 0.0339 reference SD, worst SD drift 1.12%, paired objective change -0.0044 ± 0.0122 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  50k→60k: within_screen; worst mean drift 0.0525 reference SD, worst SD drift 0.97%, paired objective change -0.0032 ± 0.0096 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  Native k 0.163–0.269; packed k 0.268–0.332; packed smoothed weight ESS fraction 0.926–0.93. Functional MMD² 0.000339104, NUTS–NUTS 0.000372146, VI–VI 0.000423152; latent MMD² 0.000355199. No MMD cutoff or IID p-value.
- Seed 7102: replay/packing prerequisite within_screen. 40k primary status within_screen, max mean error 0.0380 reference SD, SD ratios 0.991–1.01. 60k primary status within_screen, max mean error 0.0516, SD ratios 0.992–1.01.
  40k→50k: within_screen; worst mean drift 0.0684 reference SD, worst SD drift 1.02%, paired objective change 0.0011 ± 0.0063 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  50k→60k: within_screen; worst mean drift 0.0304 reference SD, worst SD drift 1.26%, paired objective change -0.0013 ± 0.0076 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  Native k 0.174–0.3; packed k 0.185–0.314; packed smoothed weight ESS fraction 0.924–0.931. Functional MMD² -0.000244453, NUTS–NUTS 0.000372146, VI–VI -0.000219608; latent MMD² 4.0432e-05. No MMD cutoff or IID p-value.
- Seed 7103: replay/packing prerequisite within_screen. 40k primary status within_screen, max mean error 0.0426 reference SD, SD ratios 0.992–1.02. 60k primary status within_screen, max mean error 0.0319, SD ratios 0.991–1.02.
  40k→50k: within_screen; worst mean drift 0.0396 reference SD, worst SD drift 0.94%, paired objective change 0.0011 ± 0.0067 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  50k→60k: within_screen; worst mean drift 0.0174 reference SD, worst SD drift 1.37%, paired objective change -0.0047 ± 0.0060 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  Native k 0.146–0.413; packed k 0.163–0.329; packed smoothed weight ESS fraction 0.928–0.932. Functional MMD² -0.000289147, NUTS–NUTS 0.000372146, VI–VI -0.000217013; latent MMD² -7.41439e-05. No MMD cutoff or IID p-value.

Final seed repeatability: within_screen; exploratory passing seeds [7101, 7102, 7103]; all-seed recipe True.

- Seeds 7101/7102: within_screen; worst mean drift 0.0454, SD drift 0.99%.
- Seeds 7101/7103: within_screen; worst mean drift 0.0277, SD drift 1.35%.
- Seeds 7102/7103: within_screen; worst mean drift 0.0503, SD drift 1.14%.

## Hierarchical posterior agreement, stability and repeatability

- Seed 7101: replay/packing prerequisite within_screen. 40k primary status outside_screen, max mean error 0.0588 reference SD, SD ratios 0.792–1.02. 60k primary status outside_screen, max mean error 0.0422, SD ratios 0.798–1.02.
  40k→50k: within_screen; worst mean drift 0.0482 reference SD, worst SD drift 1.08%, paired objective change -0.0082 ± 0.0109 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  50k→60k: within_screen; worst mean drift 0.0470 reference SD, worst SD drift 1.52%, paired objective change -0.0008 ± 0.0081 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  Native k 0.448–0.634; packed k 0.442–0.637; packed smoothed weight ESS fraction 0.396–0.551. Functional MMD² -0.000237701, NUTS–NUTS 0.000688067, VI–VI -0.000292379; latent MMD² -0.000205611. No MMD cutoff or IID p-value.
  log_sigma: standardized mean error -0.0141 ± 0.0284, SD ratio 0.8208 ± 0.0176; mean within_screen, SD outside_screen.
  sigma: standardized mean error -0.0422 ± 0.0286, SD ratio 0.7979 ± 0.0208; mean within_screen, SD outside_screen.
- Seed 7102: replay/packing prerequisite within_screen. 40k primary status outside_screen, max mean error 0.0313 reference SD, SD ratios 0.805–1.02. 60k primary status outside_screen, max mean error 0.0599, SD ratios 0.805–1.02.
  40k→50k: within_screen; worst mean drift 0.0631 reference SD, worst SD drift 1.64%, paired objective change 0.0027 ± 0.0062 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  50k→60k: within_screen; worst mean drift 0.0546 reference SD, worst SD drift 1.16%, paired objective change -0.0005 ± 0.0070 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  Native k 0.511–0.556; packed k 0.446–0.687; packed smoothed weight ESS fraction 0.364–0.542. Functional MMD² 0.000105171, NUTS–NUTS 0.000688067, VI–VI -0.000179536; latent MMD² 3.52393e-05. No MMD cutoff or IID p-value.
  log_sigma: standardized mean error -0.0336 ± 0.0285, SD ratio 0.8286 ± 0.0178; mean within_screen, SD outside_screen.
  sigma: standardized mean error -0.0599 ± 0.0287, SD ratio 0.8055 ± 0.0210; mean within_screen, SD outside_screen.
- Seed 7103: replay/packing prerequisite within_screen. 40k primary status outside_screen, max mean error 0.0551 reference SD, SD ratios 0.796–1.02. 60k primary status outside_screen, max mean error 0.0545, SD ratios 0.797–1.02.
  40k→50k: within_screen; worst mean drift 0.0376 reference SD, worst SD drift 1.59%, paired objective change -0.0017 ± 0.0087 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  50k→60k: within_screen; worst mean drift 0.0276 reference SD, worst SD drift 1.10%, paired objective change -0.0012 ± 0.0074 (two MCSE). Base estimates remain alongside any one-off precision refinement.
  Native k 0.51–0.627; packed k 0.402–0.631; packed smoothed weight ESS fraction 0.393–0.547. Functional MMD² 6.38858e-05, NUTS–NUTS 0.000688067, VI–VI -0.000134115; latent MMD² 4.6181e-05. No MMD cutoff or IID p-value.
  log_sigma: standardized mean error -0.0270 ± 0.0284, SD ratio 0.8215 ± 0.0176; mean within_screen, SD outside_screen.
  sigma: standardized mean error -0.0545 ± 0.0286, SD ratio 0.7966 ± 0.0208; mean within_screen, SD outside_screen.

Final seed repeatability: within_screen; exploratory passing seeds []; all-seed recipe False.

- Seeds 7101/7102: within_screen; worst mean drift 0.0470, SD drift 1.26%.
- Seeds 7101/7103: within_screen; worst mean drift 0.0291, SD drift 1.08%.
- Seeds 7102/7103: within_screen; worst mean drift 0.0501, SD drift 1.48%.


## References and retained failures

No new fixed NUTS job ran. Its accepted four-chain repair_1 posterior is reused; the old repair budget remains exhausted. Hierarchical initial/repair settings came from the already resolved protocol rather than being invented from the continuation plan: {'warmup': 1000, 'draws': 2000, 'target_accept': 0.95, 'max_tree_depth': 10} / {'warmup': 2000, 'draws': 4000, 'target_accept': 0.99, 'max_tree_depth': 10}. Four independent data-based starts use the tested corrected NumPyro initialization API, fresh mass/warmup state and a distinct reference seed. Physical IDs differ while observations/basis/penalty/prior roles are preserved.

Executed inference jobs: ['fixed_tail_7101', 'fixed_tail_7102', 'fixed_tail_7103', 'hierarchical_reference_initial', 'hierarchical_vi_7101', 'hierarchical_vi_7102', 'hierarchical_vi_7103']. No hierarchical repair was required in this execution; zero diagnostic precision refinements were needed. Every seed completed exactly 60,000 actual updates. The original inherited checkpoints plus 50k/60k are retained; losses and parameter checkpoints are saved before expensive summaries. The initial schedule test compared compiled rates with uncompiled rates, producing tiny rounding differences; corrected equivalent compiled comparisons and the real bitwise guide replays passed. The initial offline plot adapter failed on JSON lists, and its traceback is retained in logs/controller_initial.log. Converting saved numeric lists to arrays repaired reporting without repeating inference. Tests/logs preserve both initial failures.

| Reference / attempt | Role | Divergences / cap hits | Max R-hat | Min bulk / tail ESS | Min BFMI | Max primary mean MCSE / SD |
|---|---|---|---|---|---|---|
| fixed / repair_1 | reused accepted reference | 0 / 0 | 1.00132 | 27671 / 11244 | 0.979 | 0.0060 |
| hierarchical / initial | accepted_screen | 0 / 0 | 1.00151 | 6263 / 5767 | 0.899 | 0.0129 |

The new hierarchical reference completed its initial 4 × (1,000 warmup + 2,000 retained) workflow; the frozen repair was not needed. Both references have adequate precision for the declared comparisons. Their target-specific posterior draws are never interchanged.


## Diagnostic definitions and cost

Primary and secondary coordinates, physical per-draw band quadrature, dense 257-node grid and all complete coefficient draws are inherited. Endpoint accuracy uses two-MCSE classifications with |mean error|≤0.1 reference SD and SD ratios [.9,1.1]. Late comparisons call the same inherited function with its mean denominator, symmetric relative-SD denominator, paired MCSE and objective 0.2+2MCSE screen. Missing or precision-limited results do not pass a gate. At most one saved-guide precision refinement per comparison increases checkpoint collections to 16,384 or final scalar collections to 32,768; base results stay retained. Reference autocorrelation enters mean/SD/quantile MCSE. This is validation against NUTS, not an online stop for new data.

Final scalar/field summaries use 16,384 fresh joint draws; checkpoints use 4,096 common-key draws. Final PSIS uses 4,096×3 seeds, and objective evaluation uses 256×8 fixed keys plus three independent keys. Native PSIS splits keys into per-particle traces; packed PSIS uses vectorized posterior draws, so their realized particles and finite-sample k differ. Entire priors/factors and one support Jacobian are included; fixed sigma is excluded and hierarchical sigma is packed with its positive transform. MMD uses 512 matched draws and reference-only disjoint whitening, with descriptive baselines and negative unbiased values preserved. No reweighting.

The hierarchical 40k endpoint is a secondary 4,096-draw checkpoint result; the 60k endpoint is the primary final result. Fixed 40k accuracy reuses the exact archived 16,384-draw endpoint. Coefficient agreement labels refer to marginal mean/SD screens. Full coefficient correlations, covariance discrepancies, raw/log band powers, 50/90/95% interval-width ratios, standardized Wasserstein distances and dense-field diagnostics are retained in each complete.json and field_dense.json, alongside every feature's MC uncertainty.

Each fresh subprocess logs process-inclusive time (imports/recovery/compilation-containing fit/checks/output); fit manifests record initialization, first compiled chunk, remaining optimization loop including checkpoints, posterior sampling and final diagnostics. Guide/loss/posterior I/O and offline feature/MMD/field analysis are timed separately. Checkpoint timestamps separate reaching 40k, 50k and 60k. Timers synchronize through host copies/blocking conversions. The hierarchical NUTS timing includes warmup/sampling, compilation and native reconstruction, with chain persistence and health/feature analysis separated. These are research workflow costs under the recorded cache environment, not controlled cold or compile-reused latency comparisons. No ratio or speed advantage is inferred from historical workflow times.

| Target / seed | Shared fit including final diagnostics (s) | Parameter/posterior/loss persistence (s) | Offline features/MMD/fields (s) | Inclusive fit workflow (s) | Fresh process inclusive (s) |
|---|---|---|---|---|---|
| fixed / 7101 | 15.42 | 0.36 | 8.91 | 24.69 | 26.99 |
| fixed / 7102 | 15.77 | 0.20 | 8.95 | 24.93 | 27.67 |
| fixed / 7103 | 15.95 | 0.22 | 9.15 | 25.31 | 28.07 |
| hierarchical / 7101 | 16.55 | 0.33 | 7.96 | 24.84 | 27.34 |
| hierarchical / 7102 | 16.78 | 0.46 | 7.93 | 25.16 | 27.99 |
| hierarchical / 7103 | 16.79 | 0.38 | 8.13 | 25.30 | 27.71 |

The hierarchical reference used 10.33 s for compilation-containing warmup/sampling/native reconstruction, 0.26 s for chain persistence and 0.27 s for health/feature analysis (10.85 s workflow; 13.65 s fresh-process inclusive). Gate-analysis costs are additional research validation costs recorded in the attempt ledger; the per-fit figures do not include the full test suite, report rendering or every offline artifact check. No fixed-reference runtime is newly measured.

Gate-analysis and precision-refinement costs are recorded separately in attempts.jsonl/refinement JSON; timing.json preserves the full boundary breakdown. Tests and saved-artifact verification are recorded in verification.json. No test count or posterior validity is inferred from the earlier report.

## Implementation and verification

The continuation adapter reuses the stationary model, optimizer, reconstruction and diagnostics. The original stationary plot entry point gained configurable interval labels; shared inference defaults are unchanged. Five continuation contract tests cover exact schedule/replay behavior, RNG-independent checkpoint observation, fresh-state timing/data dependence, distinct gates and stopping. Existing density/gradient, latent-packing and scientific regressions remain in the verification scope.

The complete applicable routine and slow-regression suite passed: 144 tests, 29 warnings, no skips (80.25 s). The corrected focused contract run passed nine tests. Ruff and whitespace checks passed. Saved-artifact verification checked six 60k fits, every checkpoint round trip, bitwise fixed-prefix equality at all six archived checkpoints for all three seeds, 16,384 complete joint draws per final guide, and all 2,484 comparison rows. Recomputed packed target/guide densities and conditional coefficient gradients had zero maximum discrepancy. All six figures were inspected. Initial test/plot failures and historical scientific failures remain retained; passing software tests are separate from posterior validation.

Inspect [verification.json](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/verification.json), [comparison.csv](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/comparison.csv), [timing.json](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/timing.json) and [gate_decisions.json](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/gate_decisions.json) for numerical evidence. The source hashes remained unchanged throughout execution.

## Decision and stop

The evidence does not yet justify the matched-accuracy efficiency study. Resolve the reported prerequisite or hierarchical posterior discrepancy before speed tuning. No benchmark or reserved-data job has run.

The next supported experiment is a bounded diagnostic of coefficient–roughness geometry using the saved NUTS and guide draws: compare log sigma with the native quadratic roughness c'Pc, conditional spread and skewness. This would distinguish missing nonlinear dependence or marginal shape from the linear covariance already represented by the stable full-covariance guide. A later verified coordinate comparison may be warranted by that evidence, but has not been run or assumed to fix the discrepancy. More optimizer steps and speed tuning are not justified by this stable but underdispersed smoothing posterior alone.

Reproduction from the audited worktree:

```bash
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary_next.py audit-next
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary_next.py continue
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/stationary_next.py summarize-next
```

Completed or failed attempt directories are retained without optimizer resumption or replacement. Sources are checked against their initial hashes. Seeds 60102–60106 remain untouched. Truth overlays are context; these results establish neither calibration/coverage/SBC nor model adequacy, new-record robustness or time-varying performance.

## Figures

PSD shading shows posterior 90% intervals. Mean-discrepancy and SD-ratio shading, and band-summary error bars, show two Monte Carlo standard errors for the comparison estimates; they are not posterior credible intervals or simultaneous bands. The roughness panels show marginal posterior distributions, and stability panels show the declared drift screens and MC uncertainty.

![psd](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/figures/psd.png)

![mean_error](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/figures/mean_error.png)

![sd_ratio](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/figures/sd_ratio.png)

![bands](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/figures/bands.png)

![roughness](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/figures/roughness.png)

![stability](/Users/avi/.codex/worktrees/18a1/LogPSplinePSD/runs/vi-stationary-next/figures/stability.png)
