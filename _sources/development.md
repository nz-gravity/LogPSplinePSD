# Development Guide

This page contains notes for developers contributing to LogPSplinePSD.

## Version Control and Releases

### Commit Types for Releases

**Release-triggering commits:**

- `feat:` - New features → minor version bump (0.1.0 → 0.2.0)
- `fix:` - Bug fixes → patch version bump (0.1.0 → 0.1.1)
- `perf:` - Performance improvements → patch version bump
- `feat!:`, `fix!:`, `perf!:` - Breaking changes → major version bump (0.1.0 → 1.0.0)

**Non-release commits:**

- `docs:` - Documentation only
- `style:` - Code formatting, whitespace
- `refactor:` - Code restructuring without behavior change
- `test:` - Adding or updating tests
- `build:` - Build system changes
- `ci:` - CI configuration changes
- `chore:` - Maintenance tasks

## Building Documentation

To build the documentation locally:

```bash
source .venv/bin/activate
cd docs
# The docs in this repo are RST/Sphinx-based and require Jupyter Book 1.x.
# Ensure you have installed the dev extras (or at least `jupyter-book<2.0`).
../.venv/bin/jupyter-book build .
```

The built HTML documentation will be in `docs/_build/html/`.

## Running Tests

```bash
.venv/bin/python -m pytest tests/
```

## Typechecking (Jaxtyping + Beartype)

Install dev extras so `jaxtyping` and `beartype` are available:

```bash
.venv/bin/python -m pip install -e '.[dev,typecheck]'
```

Run static type checking across the package:

```bash
.venv/bin/python -m mypy --config-file pyproject.toml src/log_psplines
```

The mypy configuration in `pyproject.toml` includes scoped overrides for a
small set of external libraries without stubs. The package source under
`src/log_psplines` is checked end-to-end without per-module suppressions.

Runtime checks are enabled by default when dependencies are installed. You can
disable runtime enforcement with:

```bash
LOG_PSPLINES_RUNTIME_TYPECHECK=0 .venv/bin/python -m pytest tests/test_runtime_typecheck.py
```


# PSD architecture

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
  component registry has been removed. Observation-driven preparation lives
  in `inference.initialisation.prepare_components`; the collection has no
  constructor that reads data or fits coefficients.
- `inference/model.py`: NumPyro priors, scalar evaluation, likelihood calls
  and preparation of model arguments. `vi.py` and `nuts.py` retain factorized
  VI, warm starts, per-channel tuning and blocked NUTS. Evidence remains an
  optional inference operation in `inference/evidence.py`. The blocked stages
  have no generic base class and accept only the arguments they use.
  Stationary and power fitting share `inference.nuts.run_nuts`.
- `results.py` and `arviz_utils/`: `PSDResult`, storage, posterior reconstruction,
  quantiles and ArviZ interoperability. `.idata` exposes the original DataTree.
  Existing ArviZ variable names and coordinates are preserved. Result properties
  reorder axes to put matrix dimensions last and restore physical units.

`PSDResult.from_netcdf(path)` reloads posterior/model metadata and reconstructs
spectral draws. It does not recreate live NumPyro optimizers or stage objects.
`save(outdir)` writes data first, then summaries and diagnostic plots. Unexpected
reporting errors propagate. Spectrum plots are named `posterior_spectrum.png`;
there is no fallback that substitutes a trace plot. `to_netcdf(path)` only stores
data. Figure creation lives in `plotting/results.py`, diagnostic table writing
in `diagnostics/report.py`, and result packing in `arviz_utils/to_arviz.py`.

## Time dependence: shared scalar models

A stationary scalar component accepts weights `(Kf,)` and returns `(F,)`.
A time basis accepts weights `(Kt, Kf)` and evaluates
`Bt @ weights @ Bf.T`, returning `(T, F)`. This evaluator uses an optimized
tensor contraction and does not require a dense Kronecker basis. There is no
separate time-varying class.

`SpectralMatrix` accepts scalar values with any leading dimensions. Inputs
`(..., C)` and `(..., C*(C-1)//2)` produce `(..., C, C)`. This includes both
`(F, C, C)` and a future `(T, F, C, C)` without changing matrix algebra.
Posterior sample axes use the same rule. It does not define temporal priors.

`fit(PowerSpectrum, PowerSplineConfig, model=LogPSpline(...))` now samples
scalar time-frequency surfaces with the package's WDM tensor prior. It calls
`inference/power.py` directly. The optional `preprocessing/wdm.py` adapter
produces powers/counts; inference has no transform dependency.

`PSDResult.time` contains the time grid for these fits. Coefficients and both
bases are stored for reconstruction and NetCDF round trips. The old standalone
scalar storage helper still rejects time bases; TV fit storage uses the common
result's explicit `power_basis` group instead. Existing stationary plotting
helpers remain stationary; `PSDResult.save()` renders a surface for TV fits.

The historical stationary prior and VI/blocked-NUTS path are unchanged.
Multivariate TV inference, moving-periodogram adapters and TV VI remain future
work. See [the time-varying PSD notes](time-varying-psd.md) for conventions and LS2 checks.

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
  than retained as aliases. The canonical public API is the single `fit()`
  entry point, with `make_pipeline()` available for explicit orchestration.

## Explicit contracts after the cleanup

- `get_psd_dataset(result.idata)` handles both stationary and scalar TV fits.
  Its labeled axes are `(chain, draw, channel, channel_aux, [time,] frequency)`.
  `PSDResult.spectral_density` moves the matrix axes to the end. Missing sample
  groups may be skipped; corrupt selected groups raise their original error.
- `PipelineConfig.chain_method` reaches NumPyro. The unused `design_from_vi`
  and `design_from_vi_tau` options have been removed. This does not remove the
  separate low-level design-weight fitting function.
- VI pointwise likelihoods are not currently computed. Their group is absent,
  so LOO metrics stay unavailable instead of being computed from zero arrays.
- `SplineBasis` records `penalty_normalization`, `penalty_ridge` and
  `knot_convention`. Constructor keywords `normalization` and `ridge` expose
  these choices. Historical defaults are unchanged: `from_knots` uses max
  normalization, ridge 1e-6 and breakpoints; `from_grid` uses trace normalization,
  no ridge and clamped knots. Storage records the choices and preserves exact
  operators for clamped bases. Old stationary files retain historical defaults.
- `SpectralComponents.from_multivar_fft(...)` becomes
  `inference.initialisation.prepare_components(...)`.
  `components.compute_design_weights(S)` becomes
  `inference.initialisation.fit_design_weights(components, S)`.

The cleanup did not transfer moving-periodogram preprocessing or implement
multivariate TV inference. Keep those additions separate from changes to priors.
