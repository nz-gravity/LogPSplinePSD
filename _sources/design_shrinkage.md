# Soft Shrinkage Toward a Design PSD

## Overview

By default the multivariate P-spline model shrinks all spline weights toward **zero**, which corresponds to a flat log-PSD prior. When a physics-based reference (design) PSD is available — for example the LISA instrument noise model — you can instead shrink the posterior toward that design. This typically reduces bias and tightens credible intervals in the low-SNR regime where the data alone are weakly informative.

## Method

The model parameterises each Cholesky component (diagonal log-variance `delta_j` and off-diagonal coupling `theta_{j,l}`) as a P-spline:

```
component(f) = B(f) @ w
```

The standard smoothness penalty is:

```
log p(w | phi) ∝ (k/2) log(phi) - (phi/2) w^T P w
```

where `P` is the second-difference penalty matrix and `k = rank(P)`.

With design shrinkage the penalty is shifted so that deviations from the **design weight** `w_design` are penalised instead of deviations from zero:

```
residual = w - w_design
log p(w | phi) ∝ (k/2) log(phi) - (phi/2) residual^T P residual
```

An optional isotropic level-shrinkage term (scale `tau`) can be added to penalise the overall level of the residual:

```
log p(w | phi, tau) ∝ (k/2) log(phi) - (phi/2) residual^T P residual
                     - (1 / 2 tau^2) sum(residual^2)
```

The design weights `w_design` are obtained by fitting P-spline coefficients to the Cholesky decomposition of the supplied design PSD matrix. Specifically, given the design matrix `S` at model frequencies:

1. Compute the Cholesky factor: `S = L L^H`
2. Extract diagonal targets: `log_delta_sq_j = 2 log |L[f, j, j]|`
3. Form the unit lower-triangular factor: `T_inv = L / diag(L)`, then `T = T_inv^{-1}`
4. Extract off-diagonal targets: `theta_{j,l} = -T[f, j, l]` for `j > l`
5. Fit spline weights to each target curve via gradient descent (`init_weights`)

## Parameters

| Parameter    | Type                          | Default | Description                                                                              |
|--------------|-------------------------------|---------|------------------------------------------------------------------------------------------|
| `design_psd` | `np.ndarray` or `(freqs, psd)` | `None`  | Reference PSD at model frequencies (shape `(N, p, p)`, complex Hermitian), or a `(freqs_hz, psd_array)` tuple which is interpolated to the model grid. Units must match the **original** (pre-standardisation) data. |
| `tau`        | `float`                       | `None`  | Scale of the isotropic level-shrinkage term. Larger values → weaker shrinkage. `None` disables the L2 term entirely. |

## Backward Compatibility

Both parameters default to `None`. When `design_psd=None` the behaviour is **identical** to the standard model: `residual = w`, tau term absent. No existing call site needs to change.

## Usage

```python
from log_psplines import run_mcmc
import numpy as np

# true_psd: (N, p, p) complex array at FFT frequencies, original units
# freqs_hz: (N,) frequency array matching true_psd rows

result = run_mcmc(
    timeseries=data,
    sampler="multivar_blocked_nuts",
    n_knots=10,
    n_samples=1000,
    n_warmup=500,
    design_psd=(freqs_hz, true_psd),   # (freqs, psd) tuple → auto-interpolated
    tau=1.0,                            # level-shrinkage scale; None = off
)
```

## When to Use

- **Known instrument noise floor** (e.g. LISA, ground-based detectors): supply the analytical noise PSD as `design_psd`.
- **Low SNR / short data segments**: the design PSD acts as an informative prior, preventing the spline from fitting noise fluctuations.
- **Sensitivity analysis**: run with `design_psd` set to different reference models and compare posteriors.

## Empirical Results

On a 3-channel VAR(2) simulation (n = 512, 5 knots):

| Metric            | No design | With design |
|-------------------|-----------|-------------|
| RIAE              | 0.270     | 0.214       |
| CI coverage       | 0.458     | 0.589       |
| Diag CI width     | 0.248     | 0.147       |
| Off-diag CI width | 0.168     | 0.123       |

Supplying the true PSD as the design reduced estimation error by ~20% and improved coverage from 46% to 59% compared to the uninformative (zero-centred) prior.
