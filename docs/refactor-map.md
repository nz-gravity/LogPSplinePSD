# Stationary architecture refactor

Baseline: ab761338e93b3fefc6fb1c1fabaa7f1d3b1cd154.

| Existing responsibility | Destination |
| --- | --- |
| Basis and integrated-derivative penalty in psplines/initialisation.py | basis/splines.py |
| Scalar evaluation in LogPSplines | models/spectrum.py: LogPSpline |
| Data-driven knot selection and initial weights | preprocessing and inference/components.py |
| Modified-Cholesky construction | models/matrix.py: SpectralMatrix |
| NumPyro likelihood arithmetic | likelihoods/whittle.py and wishart.py |
| NumPyro model + model_kwargs | inference/model.py |
| VI/NUTS stage implementations | inference/vi.py and nuts.py |
| Pipeline configuration | config.py |
| Pipeline factory and orchestration | pipeline.py, with fit(data, config) |
| PipelineResult, storage and posterior presentation | results.py and arviz_utils |
| MultivariateTimeseries / MultivarFFT | data: TimeSeries / WishartData |
| Pipeline preprocessing | preprocessing/spectral.py |

Keep the existing one-channel Wishart inference path, hyperpriors, clipping,
normalization, initialization, factorized VI and blocked NUTS. Domain-specific
analytical PSD inputs remain preprocessing inputs. No time-varying inference.

The baseline penalty is an integrated derivative penalty from scikit-fda,
normalized by its maximum entry and regularized with 1e-6 on the diagonal.
Some old docstrings call it a finite-difference penalty. This refactor preserves
the implementation, not that inaccurate description.

Validation: frozen deterministic baseline arrays and fixed-seed small fits,
existing preprocessing, storage, diagnostics and inference tests, followed by
the complete suite. Sampling smoke tests are not convergence claims.
