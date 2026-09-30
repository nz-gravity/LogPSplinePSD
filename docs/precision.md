# Univariate float32 precision investigation

Float32 is promising for small, normalized likelihood calculations, but this
study does **not** establish posterior equivalence for stationary P-splines.
The float64 reference has poor sampler diagnostics, and float32 target errors
increase when the smoothing scale approaches zero. Keep explicit float64
references for further investigation. The package's existing defaults are
unchanged.

This first investigation covers an analytic constant spectrum and the current
stationary univariate model. It uses the attached investigation note as a
study proposal, with the user's narrower univariate scope. Time-varying
univariate, multivariate, LISA applications, evidence calculations and
accelerators were not investigated. Existing multivariate and time-varying
suite tests were run for regression coverage only.

The checkout was clean at revision
`834842d8ef07d3664948e6a49346dbca106ea713`. Experiments ran on macOS arm64, one
JAX CPU device, Python 3.12.12, NumPy 2.4.3, JAX/jaxlib 0.9.1,
NumPyro 0.20.0, SciPy 1.17.1 and arviz-stats 1.0.0. Manifests record the
revision, dirty paths, source hashes, settings, spline data hashes and sampler
seeds. The constant target uses fixed sufficient statistics (summed power 12,
count 8); early constant-run file hashes refer to an unused shared spline
fixture. New runs hash a separate constant-statistic fixture.
Sampling workers select X64 once, at process startup. Matmul precision is
`highest`; device and chain method match across policies.

The actual default path is not uniformly float64. Without an environment
override, JAX starts with X64 disabled; importing the package preserves that
setting. Host normalization uses float64. With X64 enabled, basis construction
and penalties are float64, while `prepare_model()` explicitly casts observations
and bases to float32. Default latent sites and target calculations then use
float64 with already-rounded observations and bases. Thus simply enabling X64
does not produce the unrounded float64 reference used here. Reconstruction
uses host float64/complex128.

The only production change makes prior constants inherit the penalty dtype and
empty channel-zero arrays inherit the evaluated spectrum dtype. Previously
those arrays promoted an explicitly supplied float32 target when X64 was
enabled. No prior, likelihood, ridge, normalization, default or public precision
option was changed. Workers now audit latent sites, gradients, energy,
adaptation step sizes and mass matrices; all floating sampler state has the
requested dtype. Compiled target IR contains only `f32` or only `f64`,
respectively. Integer counts remain integers until likelihood arithmetic.

Both policies use the same float64 preparation followed by an explicit inference
cast. The coefficient fixture has 96 frequencies, eight spline weights, four
proper complex observations per frequency, duration and ENBW equal to one,
`eta=1`, and the existing centered HalfNormal smoothing hierarchy. Its smooth
truth is exactly representable in the fitted basis. The coefficient factors
compress summed power without replacing the observation count. FFT/WDM
preparation is not part of this paired inference pilot.

The analytic case uses the production stationary scalar likelihood with a
**test-only** inverse-gamma prior: eight proper complex observations, summed
power 12, prior IG(4,3), hence posterior IG(12,15). In log-spectrum coordinates
its target is `-12*v - 15*exp(-v) + constant`, including the Jacobian. Tests
compare gradients and local density differences against that expression, and
compare the real-component power convention with matched counts of two.
Production spline targets are checked against a separate NumPy expression for
the likelihood, existing prior and smoothing-scale Jacobian. Exact binary
parameter points separate arithmetic from rounding; saved pilot points also
measure parameter rounding. Returned scalars are converted to host float64
before subtracting local density differences.

Before the pilot, numerical gates were fixed at 0.01 nats for local target
errors, 5e-4 absolute gradient error, 5e-6 gradient error scaled by
`max(1, abs(reference_gradient))`, and 1e-6 relative PSD error. The gradient
gates are conservative engineering limits for normalized small targets;
near-zero gradients use the absolute gate. They are not universal guarantees.
The larger deterministic fixture has 4096 frequencies and 26 weights.

| Numerical probe | Float32 local error, nats | Largest absolute gradient error | Outcome |
| --- | ---: | ---: | --- |
| Ordinary fixture | 1.55e-5 | 7.82e-6 | Within gates |
| Larger grid | 2.43e-3 | 3.90e-4 | Within gates |
| Extreme smoothing, log sigma = -10 | 0.167 | 4.63 | Outside gates |
| Artificial count = 16,777,217 | 240.45 | 39.30 | Outside gates |

Ordinary reconstruction error is 6.03e-8 relative. Finite masked cells contribute
zero value and gradient; constant-covariance pooling preserves the likelihood.
Physical-scale probes confirm that normalization before casting/squaring avoids
underflow: raw float32 powers at amplitude 1e-30 underflow. These are numerical
unit checks, not validation of a LISA pipeline. The large-count probe is
artificial sufficient-statistic scaling, not a simulation of that many
independent observations. Float32 converts its count to 16,777,216.

A follow-up diagnostic retains the original float64 penalty and calculates
priors and likelihood accumulation in float64, while spectrum fields and
residual powers are float32. Its local errors are 4.96e-7 nats ordinarily,
6.09e-6 on the larger grid, 4.98e-7 under extreme smoothing, and 1.17 at high
counts. It helps substantially but does not rescue high counts. At saved pilot
points, its local errors are below 1.7e-5 nats, but gradient errors still exceed
the declared gates for four datasets. Gradients retain float32 coordinates;
kinetic energy and adaptation for this mixed calculation were not tested.
This is a deterministic diagnostic, not a certified mixed-precision sampler.
Widening a previously rounded penalty alone cannot recover its lost precision.

The pilot was fixed at five datasets (seeds 2718–2722), four sequential chains,
500 warmup and 1000 retained draws per chain, dense mass adaptation, acceptance
0.9 and maximum tree depth 10. Each policy uses independent recorded RNG
streams, independently adapts, and starts from the same dispersed float64
points cast to its dtype. A float64 repeat uses the first dataset. Summaries
were fixed at log PSD at three frequencies, integrated power over the
normalized frequency interval, and sigma. Reconstruction and rank-normalized
split Rhat, bulk/tail ESS, mean MCSE and SD MCSE use host float64.

| Data seed | Divergences, fp64 / fp32 | Largest Rhat, fp64 / fp32 | Conclusion |
| --- | ---: | ---: | --- |
| 2718 | 202 / 108 | 1.068 / 1.146 | Inconclusive diagnostics |
| 2719 | 62 / 20 | 1.070 / 1.041 | Inconclusive diagnostics |
| 2720 | 164 / 476 | 1.063 / 1.599 | Inconclusive diagnostics |
| 2721 | 39 / 162 | 1.036 / 1.036 | Inconclusive diagnostics |
| 2722 | 139 / 1114 | 1.295 / 1.685 | Inconclusive diagnostics |

The independent float64 repeat has 59 divergences and largest Rhat 1.082.
The bounded tuning check on seed 2718 used 1000 warmup, 2000 draws per chain
and acceptance 0.99; fp64/fp32 still have 23/22 divergences, 1221/1734 trajectory
limit hits, and largest Rhat 1.068/1.232. It was not followed by further reruns
seeking a favorable result. Timing from this check is not used because tests
were running concurrently.

These runs show a shared difficulty near sigma=0; interpreting that as centered
hierarchical geometry is a hypothesis, not proof of its sole cause. Saved
float64 pilot points with sigma as small as 5.24e-4 expose additional float32
arithmetic and penalty-rounding sensitivity: four datasets exceed at least one
numerical gate, with local density error up to 0.0833 nats. These are valid
parameter probes from poorly mixed runs, not validated posterior-region
coverage. Divergence counts alone cannot assign the difference to precision.

Equivalence margins were fixed at a mean shift below 0.1 reference posterior
SD and an SD ratio in [0.95,1.05]. Approximate simultaneous intervals use
`k = NormalQuantile(1 - 0.05/(2*M))`, with 64 mean/SD comparisons in the pilot,
including the repeat. A mean interval must lie wholly inside its margin; an
SD-ratio interval uses independent SD MCSEs on the log-ratio scale and must lie
wholly inside its margins. Diagnostics require Rhat <1.01, bulk/tail ESS >=400,
zero divergences and zero trajectory limit hits. Inconclusive diagnostics are
never treated as equivalent. Truth-error summaries are recorded separately;
five simulations do not establish calibration.

An independent analytic-posterior check used four chains, 500 warmup and 5000
draws per chain. Both precisions have zero divergences, largest Rhat <1.001
and bulk ESS >6500. Their means and SDs satisfy the separate six-MCSE analytic
regression check against exact inverse-gamma moments. Mean shifts are 0.0117
reference SD for log spectrum and 0.0242 for spectrum. The joint equivalence
assessment remains **inconclusive**: SD-ratio intervals [1.0098,1.0763] and
[1.0101,1.1242] cross the 1.05 margin. Agreement with analytic moments is not
a substitute for the tighter precision-equivalence test.

An isolated CPU timing pair for the constant case gave warmed target/gradient
medians of 4.34 / 4.12 microseconds (fp64/fp32), warmed sampling of 1.083 / 1.027
seconds, and log-spectrum bulk ESS rates of 1341 / 1134 per second. Five batches
of 100 synchronized target calls include Python dispatch overhead. Compilation,
warmup, first sampling call, warmed sampling and float64 reconstruction are
recorded separately; warmup and first sampling include compilation. The warmed
sampling repeat is discarded for inference. These tiny CPU results do not
establish an efficiency benefit at equivalent statistical precision. Shared
float64 preparation was not separately timed, so full workflow speedup remains
unmeasured.

Input storage is 8 / 4 bytes and retained samples 32,000 / 16,000 bytes in that
constant example. Actual process peak RSS was 643,547,136 / 652,148,736 bytes:
there is no measured peak-memory reduction there. Each policy has a fresh
process and retains no other policy's compiled artifacts. RSS includes imports,
compilation and diagnostics; it is not dedicated device-memory instrumentation.
Accelerator speed and memory remain untested.

Runnable commands use the repository environment:

```bash
JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  .venv/bin/pytest -m 'precision and not slow'
JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  .venv/bin/pytest tests/test_precision.py -m slow
JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  .venv/bin/python examples/precision_comparison.py --mode pilot \
  --output tests/test-output/precision-pilot
JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  .venv/bin/python examples/precision_comparison.py --mode analytic \
  --output tests/test-output/precision-analytic
```

For the bounded tuning check, use `--worker --policy fp64 --case spline`
(or fp32 with `JAX_ENABLE_X64=0`), `--fixture
tests/test-output/precision-pilot/fixture_2718.npz`, `--chains 4 --warmup 1000
--draws 2000 --target-accept .99 --sampler-seed 20000` (20010 for fp32),
and `--output tests/test-output/precision-tuning/fp64.json` (fp32.json).
The output's parent directory must exist. The isolated timing pair uses
`--case constant`, 500 warmup, 1000 draws, acceptance .9 and seeds 10000/10010.

[Compact numerical and posterior results](precision_results.csv) are retained
with this report. Full local JSON manifests, exact seeds/settings, data hashes,
dtype audits, timings and comparison intervals are in
`tests/test-output/precision-{pilot,analytic,numerical-followup,tuning,timing}/`;
these generated artifacts are ignored by Git. Early pilot manifests preserve
the source hashes at execution; the later retained-penalty numerical diagnostic
has its own manifest. No full posterior spectral grids are saved.

Permanent coverage is in `tests/test_precision.py`: five fast mathematical
regressions plus two slow tests for sampler execution/dtypes and analytic
posterior moments. Numerical failures propagate as failures; unsupported
stress regimes are explicitly characterized, not skipped or certified.
The X64-enabled full suite passed **66 tests**. The default X64-disabled
stationary/core/precision selection passed **21 tests**, with two slow precision
tests deselected. This includes public univariate and two-channel inference,
positivity/PD/coherence, ArviZ conversion, plots and result roundtrips.
Scoped Ruff checks and `git diff --check` also pass.

The next scientific step is to establish a reliable stationary float64
reference with the existing model before attempting posterior certification.
The existing noncentered parameterization was not explored in this study;
its prior/parameterization regression tests passed in the existing suite.
Selective float64 prior arithmetic deserves a bounded follow-up with complete
sampler-state audits. Neither further model families nor a precision API should
be added on the strength of this pilot.

The required Graphify rebuild was attempted using the installed CLI interpreter.
Its overwrite safeguard refused a graph with 1231 nodes against the existing
1253-node graph, citing possibly missing extraction chunks. The existing graph
was preserved; the graph refresh remains incomplete.

Files changed in this investigation:

| File | Added / deleted lines | Purpose |
| --- | ---: | --- |
| `.gitignore` | 3 / 0 | Retain the compact results CSV |
| `docs/_toc.yml` | 1 / 0 | Include the report in the documentation |
| `docs/development.md` | 3 / 0 | Link the investigation |
| `pyproject.toml` | 1 / 0 | Register the precision marker |
| `src/log_psplines/inference/model.py` | 15 / 4 | Propagate supplied operator dtype |
| `docs/precision_results.csv` | 111 / 0 | 110 compact result rows |
| `examples/precision_comparison.py` | 946 / 0 | Explicit mathematical checks and study commands |
| `tests/test_precision.py` | 112 / 0 | Five fast and two slow regressions |
| `docs/precision.md` | 234 / 0 | Bounded findings and reproducible commands |
