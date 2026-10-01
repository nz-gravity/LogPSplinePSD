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
export JAX_ENABLE_X64=true
cd docs
# The docs in this repo are RST/Sphinx-based and require Jupyter Book 1.x.
# Install .[dev,wdm] first; notebook execution needs ipykernel and the WDM extra.
../.venv/bin/jupyter-book build .
```

The built HTML documentation will be in `docs/_build/html/`. Jupyter Book
executes the tutorial notebooks and caches results for unchanged code cells.
A fresh CI checkout executes all three tutorials, including the time-varying
scaling sweep, and fails on execution errors. The Colab installation cells
carry the standard `skip-execution` tag, so docs builds test the already
installed checkout. Colab runs those cells normally.

## Running Tests

Use the project environment for all commands:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,wdm]'
JAX_ENABLE_X64=true python -m pytest -m 'not slow'  # routine CI checks
JAX_ENABLE_X64=true python -m pytest               # full release checks, including recovery fits
```

The suite is grouped by behavior:

- `tests/unit/`: spline/likelihood definitions, datasets and preprocessing.
- `tests/stationary/`: stationary priors, public fits and result round trips.
- `tests/time_varying/`: power geometry, ANOVA, parametric spectra and reconstruction.

`slow` marks the longer stationary and WDM recovery fits. Short inference
contract tests run first in CI, followed by a separate recovery step. Both
steps must pass before the release workflow can run. Install the `wdm` extra for full release
validation; otherwise the WDM recovery test skips. Test outputs use pytest's
isolated temporary directories, shown in failures. To retain them at a known
location, pass `--basetemp=tests/test-output/run` (pytest clears this directory
at the start of each run).

## Git Hooks

Install the development dependencies, then install prek's Git hooks:

```bash
uv sync --all-extras --dev
uv run prek install
```

Run hooks on staged files with `uv run prek run`, or on the whole repository
with `uv run prek run --all-files`.

## Typechecking (Jaxtyping + Beartype)

Install dev extras so the pytest-only `beartype` checker is available.
`jaxtyping` is a regular dependency for the array annotations:

```bash
.venv/bin/python -m pip install -e '.[dev,typecheck]'
```

Run static type checking across the package:

```bash
.venv/bin/python -m mypy --config-file pyproject.toml src/log_psplines
```

Mypy checks the ordinary Python types. Jaxtyping's symbolic dimensions are
checked by pytest at runtime rather than inferred by mypy. The repository
currently reports additional typing errors, including missing third-party
stubs.

Pytest's `--jaxtyping-packages` option applies `beartype` to package functions
and dataclasses only while tests import them. Normal imports and inference runs
do not install a type-checking hook or wrap source functions.
`tests/unit/test_core.py` includes negative checks that require `TypeCheckError`
for mismatched array shapes, integer arrays where floats are required, and
incorrect scalar types. These fail if the pytest import hook is disabled.
Run just those checks with:

```bash
JAX_ENABLE_X64=true .venv/bin/python -m pytest tests/unit/test_core.py -k typechecking
```

Run the suite with:

```bash
JAX_ENABLE_X64=true .venv/bin/python -m pytest tests/
```


# PSD architecture

`fit()` returns one `PSDResult` for either inference method. Its `posterior`
and `spectrum` are the fitted draws and reconstructed spectral density.
`result.vi` holds VI loss and guide diagnostics when `method="vi"`.
`result.to_arviz()` provides an ArviZ view for sampling diagnostics.

The stationary path is `TimeSeries` -> `WishartData` -> `prepare_model()` ->
blocked channel NUTS or VI -> `reconstruct_stationary_spectrum()` ->
`PSDResult`. `StationaryConfig` controls this path. The Cholesky channel
models in `inference/model.py` share the Wishart likelihood, while
`inference/initialisation.py` prepares scalar spline components.
`StationaryConfig.analytical_psd` supplies an optional reference spectral
matrix for density-based knot placement. It does not center the coefficient
prior.

The scalar time-varying path is transform -> `PowerData` -> `fit_power()` ->
`PSDResult`, configured by `PowerConfig` and an explicit `LogPSpline`.
`PowerData` accepts two coordinate geometries:

- Rectangular grid: `power (T, F)`, `time (T,)`, `frequency (F,)`.
  WDM preprocessing is one possible producer. Model evaluation uses
  `Bt @ W @ Bf.T`.
- Paired ordinates: `power (P,)`, `time (P,)`, `frequency (P,)`.
  A moving periodogram is one possible producer. Model evaluation uses a
  paired contraction at each observed coordinate.

`PowerData` always has time coordinates. Stationary fits use `WishartData`.
A `PowerPartition` can pool rectangular powers for the likelihood while the
result remains on the original grid. Both coordinate geometries share the
power likelihood and tensor P-spline prior.

`preprocessing.knot_locator.allocate_components()` places interior knots for
each named pilot component separately. Supply finite, smooth pilot values
formed from training data, knot counts, and minimum spacings in the same
coordinates used by the spline bases. The variation-quantile rule uses RMS
marginal derivatives and a 10% uniform density floor by default. Pass the
returned knots to `SplineBasis.from_grid(interior_knots=...)`; knot placement
is independent of the likelihood partition.

`SplineBasis` and `LogPSpline` construct the scalar model. `SpectralComponents`
groups stationary Cholesky components; `SpectralMatrix` reconstructs positive
definite matrices. `models/reconstruction.py` owns stationary reconstruction
and chunked PSD quantiles. `results.py` stores labeled posterior, spectrum,
sampler statistics and observed data. NetCDF saves these native values;
`diagnostics/sampling.py` reports NUTS and VI behaviour directly from
`PSDResult`; `diagnostics/spectrum.py` compares fitted spectra to supplied
truth. `plotting/` renders spectra and sampling diagnostics from `PSDResult`.

Preprocessing diagnostics assess the input spectral matrix and chosen spline
model before inference. `preprocessing/diagnostics.py` computes eigenvalue
separation, model component curves, and component knot locations.
`preprocessing/checks.py` handles warnings and saves figures drawn by
`plotting/preprocessing.py`. Post-fit diagnostics in `diagnostics/` assess
sampler behaviour and recovery of the fitted spectrum.

Multivariate time-varying inference is not yet implemented.


## Scientific precision and posterior summaries

Scientific CI and documentation execution set `JAX_ENABLE_X64=true` before
Python starts. The precision contract checks both requested JAX float64 arrays
and the stationary model's prepared data, bases, and penalties. Model
preparation follows JAX's configured precision, with no forced float32 casts.
The test suite does not change global JAX precision after import.

The summary audit found these input and reduction paths:

| Path | Input | Draws and memory | Matrix/coherence support | Callers and API |
| --- | --- | --- | --- | --- |
| `compute_psd_quantiles` | components `(draw, F, C)` or `(chain, draw, F, C)`; theta has `C*(C-1)/2` components | Reconstructs frequency chunks; originally defaulted to 50 draws and selected the first chain for 4-D inputs. Now defaults to all flattened draws; an explicit cap is labelled a limited summary. | Separate real/imaginary quantiles; optional coherence computed per draw | Documented module function, deterministic reconstruction tests; uses shared `spectral_quantiles` reducer |
| `PSDResult.quantiles` | spectra `(chain, draw, [T,] F, C, C)` or cached summary | Verified all-draw cache first; otherwise complete spectrum only. Originally could reduce an uncached preview. | Componentwise complex, real, imaginary, magnitude, coherence | Public result API; diagnostics and plotting |
| `power_result_spectra` | evaluator outputs `(chain, draw, T, frequency_chunk, C)` | Every draw enters quantiles and means; only stored preview is capped. Frequency-chunked. | Scalar/independent-channel diagonal spectra; no full correlated matrix reconstruction | Internal; spline and parametric power inference |
| Plotting's former `_spectral_quantiles` | stored `(chain, draw, F, C, C)` | Previously reduced stored draws per panel, which could be a preview. Removed. | Transformed each draw for panel-specific quantiles | Matrix plotting now requests each needed kind once through `PSDResult.quantiles` |

One shared reducer in `models/reconstruction.py` takes quantiles after the
requested per-draw transformation. Power inference retains its separate
frequency-chunked evaluator and mean/geometric-mean calculations. This avoids
combining different forward models into a new inference abstraction.

Cached summary variables carry `num_chains` and `draws_per_chain`, which are
checked against the posterior and preserved in NetCDF. Magnitude and coherence
quantiles are computed from transformed draws, never from PSD element quantiles.
The low-level reconstruction routine still defaults to a maximum of 50 output
draws for preview use; `n_samples_max=None` explicitly reconstructs all draws.
