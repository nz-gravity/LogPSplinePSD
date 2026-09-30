LogPSplinePSD
=============

``LogPSplinePSD`` estimates power spectral densities (PSDs) with Bayesian
log-P-splines. It supports univariate and multivariate time series, fits smooth
spectral matrices with NumPyro/JAX, and returns ``PSDResult`` outputs with
ArviZ-compatible ``xarray.DataTree`` posteriors.

Highlights
----------

- Log-domain P-spline models for positive PSDs.
- Multivariate Wishart likelihoods for spectral matrices.
- Stationary NUTS or VI; factorised multivariate sampling.
- Optional frequency-domain coarse graining.
- Scalar time-varying fits from WDM powers, with moving periodograms as an alternative.
- Multivariate time-varying fits from proper complex coefficient grids.
- Posterior PSD quantiles, coherence summaries, and diagnostic plots.

Install
-------

For development, use the repository virtual environment:

.. code-block:: bash

   source .venv/bin/activate
   python -m pip install -e '.[dev]'

For package use:

.. code-block:: bash

   python -m pip install LogPSplinePSD

Five-Minute Example
-------------------

.. code-block:: python

   from log_psplines import StationaryConfig, fit
   from log_psplines.example_datasets import VARMAData

   data = VARMAData.ar(order=4, n_samples=8192, fs=64.0, seed=7)

   result = fit(
       data.ts,
       StationaryConfig(
           n_knots=16,
           knot_kwargs={"method": "density"},
           vi_steps=200,
           n_warmup=100,
           n_samples=200,
           rng_key=7,
           true_psd=data.get_true_psd(),
       ),
   )

   frequency = result.frequency
   psd_draws = result.psd

Choosing a workflow
-------------------

All workflows return ``PSDResult`` through ``fit``:

.. list-table:: Supported observations and configurations
   :header-rows: 1

   * - Estimation
     - Data
     - Configuration
   * - Stationary, one or multiple channels
     - ``TimeSeries`` or ``WishartData``
     - ``StationaryConfig``
   * - Time-varying, one channel (WDM)
     - ``wdm_periodogram`` → ``PowerData``
     - ``PowerConfig`` (tensor or ANOVA)
   * - Time-varying, multiple channels (complex coefficients)
     - ``local_wishart_grid`` or ``WishartGridData.from_coefficients``
     - ``PowerConfig(structure="anova")``

WDM is the primary scalar time-varying workflow. Install the optional
``LogPSplinePSD[wdm]`` dependency to transform a one-channel ``TimeSeries``:

.. code-block:: python

   from log_psplines import PowerConfig, fit, wdm_periodogram

   powers = wdm_periodogram(series, nt=128)
   result = fit(powers, PowerConfig(structure="anova"))

The sample count must be divisible by ``nt``, and both ``nt`` and the quotient
must be even. Returned powers are in WDM coefficient-variance units, with
time divided by the full duration. See the
`time-varying example <docs/examples/timevarying-example.ipynb>`_ for a complete
workflow including truth conversion.

**Multivariate WDM inference is not implemented.** The complex GridTV path
retains cross-spectrum magnitude and phase, but its observation model does
not apply to real WDM coefficients. Fitting each WDM channel independently
estimates only diagonal powers. See
`multivariate GridTV <docs/multivariate-gridtv.rst>`_ for the supported complex
workflow and its assumptions.

Next Steps
----------

- Follow the `univariate example <docs/examples/univar-example.ipynb>`_.
- Explore the `multivariate example <docs/examples/multivariate-example.ipynb>`_.
- Try the `time-varying example <docs/examples/timevarying-example.ipynb>`_.
- Read the `spectral conventions <docs/conventions.rst>`_.

Architecture
------------

See the `development notes <docs/development.md>`_ for the shared scalar/matrix
model and time-frequency evaluation. Transform adapters stay in preprocessing;
likelihoods consume powers/counts or summed cross-channel statistics. Scalar
and multivariate TV models share ANOVA fields and spectral reconstruction.

Documentation
-------------

Build the docs locally with:

.. code-block:: bash

   source .venv/bin/activate
   .venv/bin/jupyter-book build docs

The public docs focus on package usage, configuration, outputs, API reference,
and implementation notes. Domain-specific examples are intentionally kept out of
the main docs for now and can be added later as separate studies.

References
----------

Eilers, P. H. C., & Marx, B. D. (1996). *Flexible smoothing with B-splines and
penalties*. Statistical Science, 11(2), 89-121.

Maturana-Russel, J., & Meyer, R. (2021). *P-spline spectral density estimation
with a discrete penalty*. arXiv:1905.01832.
