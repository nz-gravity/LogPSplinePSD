# Stage 0 audit and frozen protocol

Scope: stages 0–3 of the supplied 1 October 2026 investigation brief. No
posterior-selected schedule, adaptive refit, LISA, or multivariate extension.
The manuscript source is not changed by this experiment.

Checkout: `7b317130787c9d8d3b29f5933de4acbde3556a54`. Initial working-tree change:
`.github/workflows/pypi.yml`; preserved. Read CONTRIBUTING.md and graph report;
no graph wiki exists. Use `.venv/bin/python`, `JAX_ENABLE_X64=true` before import.
Installed: Python 3.12.12, JAX/jaxlib 0.9.1, NumPyro 0.20.0, NumPy 2.4.3,
SciPy 1.17.1, Optax 0.2.7, xarray 2026.2.0; CPU. Default JAX X64 was false;
all commands in this investigation explicitly enable it.

## Equation-to-code audit

| Quantity | Existing implementation / audited meaning |
|---|---|
| Window | `tang_moving_periodogram`: L=2m+1 real samples, no padding or demeaning |
| Retained cycles | B=floor((n-2m)/(thin*m)); B complete m-frequency cycles |
| Start index | s=thin*m*b+j, b=0,...,B-1, j=0,...,m-1 (zero based) |
| Centre | c=s+m (zero based); stored u=(c+1)/n, one-based convention |
| Rung | omega=2*pi*(j+1)/(2m+1) radians/sample; f=omega/(2*pi*dt) Hz |
| Coefficient | z=sum_{r=0}^{2m} x[s+r] exp(-i*omega*r)/sqrt(2*pi*(2m+1)) |
| Power/count | scattered adapter: P=2*abs(z)^2, counts=2; no gridding |
| Likelihood | `power_whittle_log_likelihood`: -sum(eta+I*exp(-eta)) |
| Model scale | exp(eta) is moving-periodogram coefficient variance, not PSD/Hz |
| White noise | E[I]=variance/(2*pi); interior one-sided physical PSD=4*pi*dt*S |
| Window duration | sample-count duration (2m+1)*dt; centre span 2m*dt; both saved |
| Physical centre | Raw zero-based sample centre is (n*u-1)*dt; u is not seconds |
| Paired model | `prepare_power_model`: einsum(pi,ij,pj), exact coordinates |
| Prior | Existing tensor eigenbasis; HalfNormal roughness; same callable VI/NUTS |
| Reconstruction | `_collect_power_samples`, `power_draws_from_basis`, `power_result_spectra`, `PSDResult` |
| LS2 truth | Existing generator uses k/n, digital S_white=1. Divide `get_true_psd` by 2*pi and evaluate at u-1/n to compare exact sample centres |

The one-sided conversion applies to interior positive frequencies only. Endpoint
weights differ; all evaluated bands exclude DC/Nyquist. Changing dt changes
frequency coordinates and physical PSD conversion, not native I. Integrating
4*pi*dt*S over [0,1/(2dt)] gives the white-noise variance. No physical PSD
conversion is applied to likelihood observations.

The implementation is audited against [Tang et al., arXiv:2303.11561v1](https://arxiv.org/pdf/2303.11561v1),
Section 2's moving-periodogram construction. Definitions 1–3 were read from the
v1 PDF (retrieved with curl after web text retrieval failed). The paper uses
Bi=ceil((T-m)/(i*m)) under its extended-observation notation and discusses
partial end-cycle inclusion in simulations; this checkout deliberately keeps
only B=floor((n-2m)/(thin*m)) complete cycles supported by the supplied raw
record. Thus this is the package's conservative full-support boundary rule,
not an exact reproduction of the paper's finite-record end-cycle policy. The package docstring says 2026;
that is not verification of the published version. DOI
10.1080/01621459.2025.2594191 could not be fetched; no published theorem or
numbering is imported. Default thin=2 retains whole cycles, not every second
ordinate. Other integer thinning values are implementation options without a
claimed theoretical guarantee.

Existing: exact scattered transform, PowerData geometry, prior, NUTS,
reconstruction, generic `fit_vi` with diag/lowrank/mvn. Missing at baseline:
scalar public VI dispatch; shared VI coefficient reconstruction; persisted VI
diagnostics; stationarity Jensen-gap diagnostics; paired fixed-window study.
The existing scattered PLS initialization materializes P*Kt*Kf; bound it by row
chunks while preserving its normal equations. No likelihood or prior replacement.

[NumPyro guides](https://num.pyro.ai/en/stable/autoguide.html) constrain positive
sites through bijections. Reuse installed guides, clip requested lowrank to the
latent dimension, and initialize at the existing model's PLS sites. NUTS remains
the default. VI has one chain and no NUTS statistics. Experimental timings label
the first optimizer chunk as compilation-inclusive, not pure compilation.

## Frozen first protocol

`examples/vi_nuts_validation/archive/dynamic_whittle/configs/smoke.toml` is frozen before fits:
n=4096, dt=1, m=8/16/32/64, thin=2, paired seeds 3101/3102. Final seeds
9101–9110 remain unopened. Fixed cubic 16x10 coefficient basis on u=[0,1],
f=[0,0.5]. Dense grid 129x65; posterior mean spectrum; normalized uniform
trapezoidal quadrature on interior f=[0.08,0.42]. The shortest window has native
frequencies 1/17,...,8/17; report edge risk separately, no claims at unsupported
DC/Nyquist. Native-support and common m=64 boundary masks are reported.

Families: stationary white; exactly initialized stationary AR(1), rho=.7 and
innovation variance 1; amplitude-modulated AR(1) with specified smooth mixed
envelope; existing LS2; stochastic stable AR(2) resonance (r=.96) drifting
between .14 and .26 cycles/sample with 512-sample burn-in; abrupt variance jump.
Frozen-coefficient AR(2) spectrum is a local target, not finite covariance.
Main evidence is raw time series, not independent exponential pixels.
Larger-basis projections check representation error before interpreting window
bias; selected larger-basis fits are a separate sensitivity. A Tang-type
order comparison uses m=round(n^(1/3))=16, proportionality fixed at 1.
Best fixed among four uses truth only as an optimistic descriptive reference.
No held-back performance claim is possible from two development seeds.

Transform audit uses n=512 and 4000 independent raw Gaussian replicates: exact
C=A Sigma A^H and P=A Sigma A^T for white, AR(1), and amplitude-modulated AR(1).
Compare m=8/32, predetermined guarded [0,256)/[256,512) epochs m=8/32,
and a deliberately naive all-rung overlapping spectrogram. Every row's raw
support is saved. Separate mean leakage from dependence and noncircularity.
Predeclared marginal mean tolerance: five Monte Carlo standard errors; power
variance/selected covariance errors are reported descriptively with block MC
standard errors, not a single p-value CI gate.

Small NUTS calibration: n=1024, same declared physical domains, 8x6 basis,
stationary/slow/mixed at m=8/32; two chains, 600 warmup, 1000 draws, target .95.
Compare diag at two initialization seeds and lowrank:10 at seed7101. Diagnostics
must qualify these as references; failures trigger a separately recorded repair,
not silent exclusion. Compare full-draw D distributions on fixed complete
windows; do not select posterior horizons (stage 4).

Truth horizons use full raw support, epsilon=.005/.02/.08, both raw and prefix
rules, and explicit censoring. They are stationarity diagnostics, not risk oracles.
Adaptation is assessed only after the fixed-window landscape and model diagnostics.

Reproduce:
```bash
JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode transforms
JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode sweep
JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/archive/dynamic_whittle/run_investigation.py --mode calibration
```
