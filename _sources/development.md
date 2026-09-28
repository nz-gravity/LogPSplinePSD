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
do not install a type-checking hook or wrap source functions. Run the suite with:

```bash
.venv/bin/python -m pytest tests/
```


# PSD architecture

`fit()` returns one `PSDResult` for either inference method. Its `posterior`
and `spectrum` are the fitted draws and reconstructed spectral density.
`result.vi` holds VI loss and guide diagnostics when `method="vi"`.
`result.to_arviz()` provides an ArviZ view for sampling diagnostics.

The stationary path is `TimeSeries` -> `WishartData` -> `prepare_model()` ->
blocked channel NUTS or VI -> `reconstruct_stationary_spectrum()` ->
`PSDResult`. `PipelineConfig` controls this path. The Cholesky channel
models in `inference/model.py` share the Wishart likelihood, while
`inference/initialisation.py` prepares scalar spline components.
`PipelineConfig.analytical_psd` supplies an optional reference spectral
matrix for density-based knot placement. It does not center the coefficient
prior.

The scalar time-varying path is transform -> `PowerData` -> `fit_power()` ->
`PSDResult`, configured by `PowerSplineConfig` and an explicit `LogPSpline`.
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

`SplineBasis` and `LogPSpline` construct the scalar model. `SpectralComponents`
groups stationary Cholesky components; `SpectralMatrix` reconstructs positive
definite matrices. `models/reconstruction.py` owns stationary reconstruction
and chunked PSD quantiles. `results.py` stores labeled posterior, spectrum,
sampler statistics and observed data. NetCDF saves these native values;
`diagnostics/sampling.py` reports NUTS and VI behaviour directly from
`PSDResult`; `diagnostics/spectrum.py` compares fitted spectra to supplied
truth. `plotting/` renders spectra and sampling diagnostics from `PSDResult`.

Multivariate time-varying inference is not yet implemented.
