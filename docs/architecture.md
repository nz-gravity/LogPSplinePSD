# Stationary PSD architecture

The public entry point is `fit(data, config) -> PSDResult`:

```python
import numpy as np
from log_psplines import TimeSeries, PipelineConfig, fit

series = TimeSeries(data=np.random.default_rng(7).normal(size=(256, 2)))
result = fit(series, PipelineConfig(n_knots=6, vi_steps=200,
                                  n_warmup=100, n_samples=200))
frequency = result.frequency
spectral_draws = result.spectral_density  # (chain, draw, F, C, C)
auto_spectra = result.psd
coherence = result.coherence
result.to_netcdf("inference_data.nc")
```

These short example chains illustrate the interface, not convergence.

## Code navigation

`pipeline.py` prepares stationary observations and scalar components, then runs
coarse VI (when requested), full VI and blocked NUTS. `make_pipeline` remains
available for inspecting the prepared calculation before calling `.run()`.
It lives in the same file, rather than a separate factory module.

- `basis/splines.py`: `SplineBasis`, B-spline construction and normalized
  integrated-derivative penalties. No data preparation, plotting or NumPyro.
- `models/spectrum.py`: `LogPSpline(frequency, time=None)` and scalar evaluation.
- `models/matrix.py`: `SpectralMatrix(C)`, with modified-Cholesky construction
  `S = inv(T) D inv(T)^H`, `T[j,l] = -theta[j,l]`, `D = diag(exp(log_variance))`.
- `likelihoods/`: pure JAX Whittle and channel-factor Wishart likelihoods.
  The latter sums to the full matrix likelihood and reduces to the former
  for one channel. Both omit data-only constants.
- `data/`: `TimeSeries.data` always has shape `(N, C)`. `WishartData` retains
  replicate counts, duration, window bandwidth and scaling metadata.
- `preprocessing/`: FFT/Wishart and Welch construction, masking, coarse
  graining and knot selection. `compute_fft`, `compute_wishart` and
  `empirical_spectrum` are functions in `preprocessing/periodogram.py`.
- `inference/components.py`: one collection of scalar models for diagonal,
  real off-diagonal and imaginary off-diagonal components. The duplicate
  component registry has been removed. Analytical spectra can guide knot
  placement without entering the scalar model or likelihood.
- `inference/model.py`: NumPyro priors, scalar evaluation, likelihood calls
  and preparation of model arguments. `vi.py` and `nuts.py` retain factorized
  VI, warm starts, per-channel tuning and blocked NUTS. Evidence remains an
  optional inference operation in `inference/evidence.py`.
- `results.py` and `arviz_utils/`: `PSDResult`, storage, posterior reconstruction,
  quantiles and ArviZ interoperability. `.idata` exposes the original DataTree.
  Existing ArviZ variable names and coordinates are preserved. Result properties
  reorder axes to put matrix dimensions last and restore physical units.

`PSDResult.from_netcdf(path)` reloads posterior/model metadata and reconstructs
spectral draws. It does not recreate live NumPyro optimizers or stage objects.
`save(outdir)` also writes diagnostic plots. `to_netcdf(path)` only stores data.

## Time dependence is an extension point

A stationary scalar component accepts weights `(Kf,)` and returns `(F,)`.
A future time basis will accept weights `(Kt, Kf)` and evaluate
`Bt @ weights @ Bf.T`, returning `(T, F)`. Passing a time basis currently raises
`NotImplementedError` on evaluation. There is no separate time-varying class.

`SpectralMatrix` accepts scalar values with any leading dimensions. Inputs
`(..., C)` and `(..., C*(C-1)//2)` produce `(..., C, C)`. This includes both
`(F, C, C)` and a future `(T, F, C, C)` without changing matrix algebra.
Posterior sample axes use the same rule. It does not define temporal priors.

`PSDResult.time` is reserved for a future time grid and is `None` for all fits
produced here. Future `moving_periodogram.py` or `wdm.py` preprocessors will
produce time-frequency observations. No moving-periodogram inference, WDM
likelihood, 2D penalty or temporal smoothing parameter is implemented.

## Migration

- `MultivariateTimeseries(y=...)` becomes `TimeSeries(data=...)`.
- `MultivarFFT` becomes `WishartData`; FFT/statistic construction moves to
  `preprocessing.periodogram` (TimeSeries convenience methods remain).
- `LogPSplines` becomes `LogPSpline(SplineBasis.from_knots(...))`.
  Initial weight fitting lives in `inference.initialisation`.
- `MultivariateLogPSplines` is replaced by prepared `SpectralComponents`
  for inference and the independent `SpectralMatrix` for matrix algebra.
- Model storage functions live in `arviz_utils.spline_storage`, and basis
  plotting in `plotting.basis.plot_spline_basis`.
- `PipelineConfig` is in `config.py`; `PipelineResult` becomes `PSDResult`.
- Old `pipeline/`, `psplines/` and `datatypes/` modules are removed rather
  than retained as aliases. `run_mcmc` remains a small DataTree-returning
  convenience API for existing diagnostic workflows.
