# Moving-periodogram investigation: stages 0–3 review

**Decision: do not implement adaptive window selection yet.** The truth horizon varies locally, but this first fixed-window landscape does not establish a robust local reconstruction gain from adaptation. Moderate m=32 is competitive across controls and LS2. Fast mixed evolution and the narrow stochastic resonance remain strongly limited by the frozen spline basis; VI uncertainty also differs materially from NUTS. No posterior horizon selection, adaptive refit, or LISA run was executed.

## Scope, reuse, and changes

Audited checkout `7b317130787c9d8d3b29f5933de4acbde3556a54`; preserved the pre-existing `.github/workflows/pypi.yml` change. Reused the exact scattered moving periodogram, PowerData, `prepare_power_model`, existing eigenbasis tensor prior, generic `fit_vi`, coefficient reconstruction and PSDResult. Added scalar `method='vi'` dispatch, PLS initialization for built-in guides, bounded-row scattered PLS normal equations, measured VI timings, persisted VI diagnostics, and pure stationarity diagnostics. NUTS remains the default. Parametric VI is explicitly rejected rather than silently running NUTS. No parallel likelihood or spline prior. During execution an external commit advanced HEAD to `63b5c7a` and changed only the release workflow; per-run manifests retain the actual SHA and dirty state. No commits were created by this investigation, and concurrent staging/docs-ignore edits were left intact.

Normalization and equation-to-code mapping: [audit.md](audit.md). Native powers are coefficient variance: P=2I, counts=2, E[I_white]=variance/(2*pi); one-sided interior PSD/Hz=4*pi*dt*S. Times remain (zero-based centre+1)/n; LS2 digital truth is divided by 2*pi and evaluated at u−1/n. Window sample-count duration and centre span are distinct and saved.

## Actually executed

- 48 fixed-window diagonal-VI fits: 6 raw-series families × 2 paired seeds × 4 windows, n=4096, 5000 optimizer steps maximum, 256 posterior draws, fixed cubic 16×10 coefficient basis.
- 6 separate larger-basis fits, seed3101 at m=16/64 for mixed/LS2/resonance; cubic 24×14 coefficients.
- 29 calibration fits on n=1024: 6 original two-chain NUTS targets, 12 diagonal VI fits across two inference seeds, 6 lowrank:10 VI fits, plus one recorded NUTS repair and four separately recorded 15000-step VI budget checks. Calibration uses the same model but a smaller 8×6 basis; this does not establish large-basis NUTS parity.
- 12 exact Gaussian transform diagnostics at n=512 with 4000 independent raw Gaussian replicates each: white, AR(1), and mixed modulation; two fixed windows, a predetermined guarded piecewise transform, and naive overlapping-spectrogram negative control.
- Deterministic truth loss/horizon curves for all 6 families at 3 tolerances; raw/prefix policies and censoring saved. These use known truth only. Reserved final seeds 9101–9110 were not opened.
- Failure count among inference jobs: 0. Failed jobs are retained in manifests, not excluded silently.

## Tests and execution failures

Final routine regression: **132 passed, 5 slow tests deselected**, 25.41 s. Includes small univariate and multivariate public fits, positivity, Hermitian/PD/coherence checks, ArviZ and rendering. Targeted contracts: **43 passed, 1 slow test deselected**, 12.04 s. Initial targeted run had 9 failures (24 passed): scalar ndarray return annotations and a config negative-test exception expectation under the runtime type hook; repaired and rerun. Ruff checks passed for all edited Python files; both staged and working-tree diff whitespace checks passed. The graph refresh command and result are recorded in verification.json. No full slow WDM recovery rerun was performed.

## Fixed-operator approximation audit

White means passed the predeclared five-MC-standard-error tolerance; the physical white PSD integral at dt=.25 equals 1. All marginal mean checks passed against exact finite-window expectations. Power variance uses |C|²+|P|², not C_ii² alone. Block-MC standard errors for variance and the highest-correlation pair are saved. The largest variance discrepancy was 5.61 block standard errors for a naive-spectrogram negative control; variance/covariance errors are descriptive and were not predeclared CI pass/fail gates. Marginal moments matching their exact Gaussian values does not make the independence/local-spectrum likelihood exact.

| Family | Operator | Maximum power correlation | RMS relative mean leakage/bias | Maximum noncircularity | Cross-epoch maximum |
|---|---|---:|---:|---:|---:|
| white | fixed_m8 | 0.0336 | 0.0000 | 0.0000 | 0.0336 |
| white | fixed_m32 | 0.0080 | 0.0000 | 0.0000 | 0.0080 |
| white | guarded_piecewise | 0.0336 | 0.0000 | 0.0000 | 0.0000 |
| white | naive_spectrogram | 0.3819 | 0.0000 | 0.0000 | 0.3819 |
| ar1 | fixed_m8 | 0.0455 | 0.1384 | 0.1596 | 0.0455 |
| ar1 | fixed_m32 | 0.0091 | 0.0364 | 0.0438 | 0.0091 |
| ar1 | guarded_piecewise | 0.0455 | 0.1060 | 0.1596 | 0.0000 |
| ar1 | naive_spectrogram | 0.3004 | 0.1384 | 0.1596 | 0.3004 |
| mixed | fixed_m8 | 0.2954 | 0.2417 | 0.4588 | 0.0440 |
| mixed | fixed_m32 | 0.1717 | 0.5596 | 0.4243 | 0.0083 |
| mixed | guarded_piecewise | 0.1717 | 0.5702 | 0.4243 | 0.0000 |
| mixed | naive_spectrogram | 0.5482 | 0.2520 | 0.4635 | 0.2909 |

Guarded epoch supports were disjoint. Cross-epoch covariance was zero to shown precision, including coloured noise whose underlying AR correlations persist but are very small across guards. Guarding does not remove within-epoch dependence or mean bias. The fast mixed n=512 diagnostic deliberately has more rapid sample-scale evolution than the n=4096 sweep; its numerical biases are not estimates for that larger sweep. Stationary AR leakage exists while D=0, so a temporal horizon cannot certify leakage control. Fixed-operator identities do not prove a data-selected transform distribution.

## Fixed-window risk landscape

Posterior mean spectrum; normalized forward Whittle risk on the same interior band/grid and exact common retained-centre mask. Bounds are computed from complete cycles, including trailing truncation. Full native-support, quiet/slow/fast/change/boundary risks, pointwise 90% coverage/width, edge risk, feature location/width and retention are saved per run. Coverage here is area-weighted pointwise coverage on TWO realizations, not a repeated-sampling calibration result. Small differences cannot be judged significant from two seeds.

| Family | m=8 | m=16 (fixed n^(1/3) rule) | m=32 | m=64 | Descriptive best mean |
|---|---:|---:|---:|---:|---:|
| white | 0.00164 | 0.00152 | 0.00117 | 0.00127 | 32 |
| ar1 | 0.00695 | 0.00516 | 0.00219 | 0.00197 | 64 |
| mixed | 0.06203 | 0.05332 | 0.05182 | 0.05217 | 32 |
| ls2 | 0.01249 | 0.01114 | 0.01036 | 0.01305 | 32 |
| resonance | 0.47184 | 0.30108 | 0.23203 | 0.21726 | 64 |
| abrupt | 0.02929 | 0.02041 | 0.01809 | 0.01828 | 32 |

Best fixed here uses development truth and is an optimistic descriptive reference; it is not a deployable tuned baseline tested on final seeds. No local risk envelope is treated as an adaptive posterior. Native/support differences remain inspectable in metrics.json.

| Family | Frozen-basis log-spectrum projection RMSE in band |
|---|---:|
| white | 0.00000 |
| ar1 | 0.00085 |
| mixed | 0.29462 |
| ls2 | 0.04521 |
| resonance | 0.48030 |
| abrupt | 0.11044 |

The larger-basis representation-only check gives mixed .2860, LS2 .0090, resonance .3236 log RMSE (primary .2946/.0452/.4803). The planned larger basis substantially improves LS2 representability but still cannot resolve the mixed fast oscillations or narrow resonance. Calibration mixed projection RMSE is .3389. These are basis limitations, so the stress cases cannot isolate window smearing. Abrupt covariance change is deliberate smooth-model misspecification.

| Larger-basis sensitivity | Primary matched risk | Larger matched risk |
|---|---:|---:|
| ls2 m=16 seed=3101 | 0.01050 | 0.01229 |
| ls2 m=64 seed=3101 | 0.01212 | 0.01311 |
| mixed m=16 seed=3101 | 0.05412 | 0.05848 |
| mixed m=64 seed=3101 | 0.05245 | 0.05483 |
| resonance m=16 seed=3101 | 0.29045 | 0.23900 |
| resonance m=64 seed=3101 | 0.21462 | 0.14949 |

## VI versus NUTS

| Family/window/guide/init/budget | RMS log posterior-mean difference | Median log-interval width VI/NUTS | Max absolute D pass-probability difference |
|---|---:|---:|---:|
| ar1 m=8 diag init=7101 steps=5000 | 0.0798 | 1.380 | 0.000 |
| ar1 m=8 diag init=7102 steps=5000 | 0.1016 | 1.169 | 0.000 |
| ar1 m=8 lowrank:10 init=7101 steps=5000 | 0.0618 | 1.157 | 0.000 |
| ar1 m=32 diag init=7101 steps=5000 | 0.0619 | 1.405 | 0.000 |
| ar1 m=32 diag init=7102 steps=5000 | 0.0957 | 1.158 | 0.000 |
| ar1 m=32 lowrank:10 init=7101 steps=5000 | 0.0984 | 1.212 | 0.004 |
| slow m=8 diag init=7101 steps=5000 | 0.1068 | 1.385 | 0.068 |
| slow m=8 diag init=7102 steps=5000 | 0.1093 | 1.336 | 0.179 |
| slow m=8 lowrank:10 init=7101 steps=5000 | 0.1086 | 1.275 | 0.121 |
| slow m=32 diag init=7101 steps=5000 | 0.1012 | 1.414 | 0.116 |
| slow m=32 diag init=7102 steps=5000 | 0.1250 | 1.370 | 0.209 |
| slow m=32 diag init=7101 steps=15000 | 0.0878 | 1.338 | 0.020 |
| slow m=32 lowrank:10 init=7101 steps=5000 | 0.1034 | 1.243 | 0.073 |
| slow m=32 lowrank:10 init=7101 steps=15000 | 0.0876 | 1.219 | 0.143 |
| mixed m=8 diag init=7101 steps=5000 | 0.0977 | 1.325 | 0.028 |
| mixed m=8 diag init=7102 steps=5000 | 0.1178 | 1.269 | 0.005 |
| mixed m=8 lowrank:10 init=7101 steps=5000 | 0.0544 | 1.187 | 0.056 |
| mixed m=32 diag init=7101 steps=5000 | 0.0892 | 1.403 | 0.079 |
| mixed m=32 diag init=7102 steps=5000 | 0.1141 | 1.345 | 0.028 |
| mixed m=32 diag init=7101 steps=15000 | 0.0787 | 1.335 | 0.017 |
| mixed m=32 lowrank:10 init=7101 steps=5000 | 0.0831 | 1.211 | 0.079 |
| mixed m=32 lowrank:10 init=7101 steps=15000 | 0.0915 | 1.165 | 0.048 |

D probabilities above are diagnostics on declared fixed complete windows at centres .12/.38/.68/.9, half-widths 8/32 and epsilon .005/.02/.08; no posterior horizon or schedule was selected. Full joint surface draws enter D. Monte Carlo granularity is 1/256 for VI; NUTS uses 2000 draws with ESS diagnostics below. Differences near thresholds need more draws/decision calibration before automation. No posterior variance multiplier was introduced.

| NUTS target | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Depth hits |
|---|---:|---:|---:|---:|---:|
| cal_ar1_m32_nuts_none_init7101 | 0 | 1.01 | 494.0 | 1035.0 | 0 |
| cal_ar1_m8_nuts_none_init7101 | 0 | 1.0 | 678.0 | 816.0 | 0 |
| cal_mixed_m32_nuts_none_init7101 | 1 | 1.01 | 645.0 | 471.0 | 0 |
| cal_mixed_m32_nuts_repair | 0 | 1.0 | 526.0 | 715.0 | 0 |
| cal_mixed_m8_nuts_none_init7101 | 0 | 1.01 | 495.0 | 800.0 | 0 |
| cal_slow_m32_nuts_none_init7101 | 0 | 1.01 | 656.0 | 773.0 | 0 |
| cal_slow_m8_nuts_none_init7101 | 0 | 1.01 | 534.0 | 687.0 | 0 |

Longer-budget checks reduced the slow/mixed diagonal-guide probability disagreements at m=32 to .020/.017, but median interval widths remained 1.338/1.335 times NUTS. Lowrank remained 1.219/1.165 times NUTS, with slow-case probability disagreement .143 after 15000 steps. Optimizer budget and guide shape both remain relevant; more steps did not make them interchangeable. The original mixed m=32 NUTS target had one divergence. It is preserved and a separate repair increased target acceptance to .99 and warmup to 1000. Comparisons use the repair when available. Step size was not requested from NumPyro and is unavailable, not zero. Healthy NUTS is a reference for this approximate likelihood, not a cure for transform dependence or basis misspecification. VI loss traces and independent initializations are retained; 5000 steps is a measured development budget, not proof of optimizer convergence. Raw ELBO is never used to rank different windows.

## Computational cost

Sum of measured job wall times: **345.6 s** (5.8 min), excluding process startup/import overhead and transform/reporting commands. Separate sweep and calibration dispatcher times are in execution.json; they overlapped, so their sum is not elapsed session time.
VI fit wall seconds, median (range): **3.04 (range 2.54–4.88)**. NUTS: **10.17 (range 9.55–12.18)**.
Isolated-process peak RSS MiB, median (range): **656.97 (range 523.50–1197.56)**; includes Python imports/JAX/runtime, not just posterior arrays.
VI initialization seconds: 1.87 (range 1.54–2.48); first chunk including compilation: 0.36 (range 0.31–0.85); remaining optimization: 0.37 (range 0.10–0.85); posterior draws: 0.20 (range 0.18–0.99).
Launch/import-inclusive original dispatcher times were 286.75 s for the 54-job sweep and 178.90 s for 24 calibration jobs; these ran concurrently. Additional repair and four 15000-step jobs are separately recorded. Initialization includes tracing/compilation. Pure compiler time is not independently measured. Preprocessing, model preparation, reconstruction and total times are saved per run; first-chunk time includes 100 optimizer steps. Posterior previews store 32 draws but all 256/2000 posterior draws enter summaries. Result round trips preserve native units, exact paired observations, posterior coefficients, model bases and VI losses/timings. Existing complex-valued h5netcdf storage emits a nonstandard-NetCDF warning; round trips through PSDResult/h5netcdf were verified, but generic NetCDF interoperability was not.

## Review gate

Truth D distinguishes short admissible windows during fast evolution from right-censored long windows in quiet regions. That supports a horizon diagnostic as a research question. It does not establish an estimation gain: the first landscape and basis sensitivities leave the hardest regimes confounded, and VI/NUTS widths/probabilities are insufficiently interchangeable for automatic selection. Stage 4 is **not justified as an automatic adaptive selector** by these results. Before approving it, resolve representation capacity and guide convergence/uncertainty on fixed-window targets, then expand paired development seeds and check approximation at the actual n. Keep the moderate fixed-window comparator. No adaptive refitting or LISA is warranted yet.

## Reproduction and artifacts

```bash
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode transforms
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode sweep
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode calibration
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode job --job '{"id":"cal_mixed_m32_nuts_repair","family":"mixed","data_seed":3101,"m":32,"method":"nuts","inference_seed":7101,"calibration":true,"nuts_warmup":1000,"nuts_target_accept":0.99}'
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode job --job '{"id": "cal_slow_m32_vi_diag_long15000", "family": "slow", "data_seed": 3101, "m": 32, "method": "vi", "inference_seed": 7101, "calibration": true, "guide": "diag", "vi_steps": 15000}'
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode job --job '{"id": "cal_slow_m32_vi_lowrank10_long15000", "family": "slow", "data_seed": 3101, "m": 32, "method": "vi", "inference_seed": 7101, "calibration": true, "guide": "lowrank:10", "vi_steps": 15000}'
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode job --job '{"id": "cal_mixed_m32_vi_diag_long15000", "family": "mixed", "data_seed": 3101, "m": 32, "method": "vi", "inference_seed": 7101, "calibration": true, "guide": "diag", "vi_steps": 15000}'
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/run_investigation.py --mode job --job '{"id": "cal_mixed_m32_vi_lowrank10_long15000", "family": "mixed", "data_seed": 3101, "m": 32, "method": "vi", "inference_seed": 7101, "calibration": true, "guide": "lowrank:10", "vi_steps": 15000}'
JAX_ENABLE_X64=true .venv/bin/python examples/adaptive_dynamic_whittle/summarize.py
JAX_ENABLE_X64=true .venv/bin/python -m pytest -m 'not slow' -q
```

All jobs/configs/data hashes, repo SHA/dirty state, versions/backend/dtype, inference settings, supports, timings, diagnostics, seed-specific metrics, loss distributions and structured failure states are in `runs/dynamic-whittle-stages-0-3/`. Bulk observations/posteriors remain ignored by Git. Source/config/report are retained. Commands preserve completed inference jobs; use a new --output directory for a fresh run. Transform summaries overwrite only their deterministic diagnostics when rerun.

![Fixed-window landscape](../../../../runs/dynamic-whittle-stages-0-3/figures/fixed_window_landscape.png)

![VI/NUTS uncertainty comparison](../../../../runs/dynamic-whittle-stages-0-3/figures/vi_nuts_comparison.png)

Prior methods: Tang's dynamic-Whittle construction supplies the observation representation; [van Delft and Eichler](https://arxiv.org/abs/1512.00825) already adapt local smoothing neighbourhoods iteratively. The proposed research distinction is choosing the observation-window representation from a posterior horizon with the existing log-P-spline prior; no novelty theorem or comprehensive novelty claim is established. Published Tang DOI access failed; the audit uses the specified v1 preprint.
