# Reusing the WDM implementation

Scalar time-frequency fitting now runs through the existing `fit()` entry
point. It uses `LogPSpline`, with the same `SplineBasis` type for time and
frequency, and returns `PSDResult`. No separate TV model hierarchy or second
pipeline was added. The stationary Wishart/VI/blocked-NUTS path is unchanged.

## What moved

| WDM responsibility | Destination |
| --- | --- |
| Clamped B-splines and trace-normalized roughness | `SplineBasis.from_grid`, `basis/penalty.py` |
| Tensor surface evaluation | `LogPSpline.__call__` |
| Eigen prior, Gamma precision sampling, centered/non-centered coefficients and least-squares start | `inference/power.py` |
| Power/count likelihood | `likelihoods/whittle.py` |
| Observations and exact real-component counts | `PowerSpectrum` in `data/spectral.py` |
| Transform and boundary trimming | `preprocessing/wdm.py` (optional `wdm-transform` dependency) |
| Coefficients, basis grids, posterior samples, diagnostics and persistence | `PSDResult` |

`PowerSplineConfig` contains the power-likelihood prior and NUTS settings. It
is distinct from the historical stationary configuration because the two
priors have different normalization and hyperparameters. The caller supplies
the scalar model explicitly, so knot selection stays outside inference.

## Small example

Install the optional transform with `pip install -e '.[wdm]'`. Already prepared
powers/counts require no WDM dependency.

```python
import numpy as np
from log_psplines import (
    TimeSeries, SplineBasis, LogPSpline, PowerSplineConfig, fit,
)
from log_psplines.preprocessing.wdm import wdm_periodogram

series = TimeSeries(data=x, t=np.arange(len(x)) * dt)
data = wdm_periodogram(series, nt=32)
model = LogPSpline(
    frequency=SplineBasis.from_grid(data.frequency / data.frequency[-1], 4),
    time=SplineBasis.from_grid(data.time, 4),
)
result = fit(data, PowerSplineConfig(), model=model)
# result.psd: (chain, draw, time, frequency)
result.save("output")
result.to_netcdf("fit.nc")
```

Time is rescaled by full series duration. Output values retain WDM
**coefficient-variance units**; they are not silently converted to PSD/Hz.
Full surface draws are reconstructed on request. Stored results contain
compact coefficients and both basis matrices, not a surface per MCMC step.

## Scientific conventions

- WDM bases use trace-normalized, unregularized marginal penalties. Historical
  stationary `from_knots` uses maximum-entry normalization plus a ridge.
  These conventions are intentionally distinct. Use `from_grid` for source
  WDM parity; substituting a stationary penalty changes the prior.
- The smoothing precisions retain `Gamma(shape=2, rate=1)` priors, sampled on
  the log scale with the original Jacobian correction. Posterior `phi_time`
  and `phi_freq` sites contain **log** precision. The joint null space has
  its original weak proper prior (`null_precision=1e-4`, `ridge_eps=1e-6`).
- One real WDM coefficient contributes `(power=w**2, count=1)`. A complex
  Fourier ordinate in the stationary convention would require
  `(power=2*FFT_power/duration, count=2)`. The half factor is intentional.
- Missing cells have both zero power and zero count. Source log-linear filling
  is used only for initialization; it never replaces likelihood observations.
- Pooling must sum original powers and exact retained counts. Adaptive bin
  selection and coarse-graining adapters have not yet been transferred.

## Validation and scope

`tests/reference/capture_ls2.py` extracts the original source functions and
records their hashes. The source working tree has unrelated uncommitted work;
it was read, not modified. The fixtures are generated under this package's
JAX/NumPyro environment, isolating code changes from dependency-version changes.
Tests need neither the sibling checkout nor its pipeline. The optional
transform parity test needs `wdm-transform`.

The fixed-seed LS2 case has 512 samples, dt=0.1, nt=32, four interior knots per
axis, 24 warmup steps and 16 retained draws. Both centered and non-centered
paths compare bases, penalties, preprocessing, initial sites, posterior target,
gradients, draws, reconstructed PSDs and sampler diagnostics against source.
NetCDF round trips and plot generation are also checked. Run:

```sh
.venv/bin/python -m pytest tests/test_power_inference.py -q
.venv/bin/python docs/studies/ls2_transfer.py
```

The second command writes comparisons under `tests/test_output/ls2_transfer`.
These are **short numerical integration checks**, not convergence, coverage or
LS2 recovery claims. A larger matched campaign remains a separate validation.

Not yet transferred: moving-periodogram/STFT adapters, adaptive knots and
binning, fixed-reference residual models, alternate exploratory hyperpriors,
TV VI, large-grid chunked summaries, or multivariate TV inference. Scalar
surface outputs already compose with `SpectralMatrix`; its trailing matrix
axes require no new TV matrix class.

### Recorded comparison

The 24/16 reference and port agree to about `7e-15` maximum relative PSD
difference. Both reproduce the same deliberately short-chain problems:
16/16 depth-limit hits for non-centered sampling and four divergences for
centered sampling. Those are not silently treated as successful inference.

A second paired run used 200 warmup steps, 100 draws and maximum tree depth 8,
with the same data, basis and seed. Both parameterizations had zero divergences
and zero depth-limit hits in both implementations. Maximum relative PSD
variation between source and port was `7.11e-15` (non-centered) and `7.22e-15`
(centered). This remains a single-realization numerical comparison.

To reproduce that longer check without overwriting the test fixtures:

```sh
.venv/bin/python tests/reference/capture_ls2.py /path/to/wdm_psd \
  --warmup 200 --draws 100 --max-tree-depth 8 --output /tmp/ls2-reference
.venv/bin/python docs/studies/ls2_transfer.py \
  --reference /tmp/ls2-reference/ls2_wdm.npz \
  --output tests/test_output/ls2_transfer_long
```

The final full package suite passed with **85 tests and 3 existing skips**,
including the stationary frozen-reference tests. Targeted Ruff checks and
`git diff --check` passed. The optional WDM adapter was exercised with
`wdm-transform==0.5.0` installed in the package's `.venv`.

### Maintenance cleanup

Both fit paths now share the NUTS execution helper and the public
`get_psd_dataset` reconstruction contract. Reporting moved out of `PSDResult`,
which retains data access, persistence and a small `save` convenience method.
The stationary stages no longer inherit unused generic stage classes.
See `architecture.md` for the configuration and storage changes.

`docs/studies/ls2_validation.py` adds a bounded three-realization check with
two centered chains per realization. It compares pointwise intervals to an
independent 512-realization WDM-power Monte Carlo reference in native variance
units. Its results are descriptive; this is not a calibrated coverage study.
