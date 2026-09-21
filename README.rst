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
- VI warm starts and factorised multivariate NUTS.
- Optional frequency-domain coarse graining.
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

   from log_psplines import PipelineConfig, fit
   from log_psplines.example_datasets import VARMAData

   data = VARMAData.ar(order=4, n_samples=8192, fs=64.0, seed=7)

   result = fit(
      data.ts,
       PipelineConfig(
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

Next Steps
----------

- Read the `five-minute guide <docs/five-minute.rst>`_.
- Learn the `PSDResult interface <docs/results.rst>`_.
- Follow the `multivariate example <docs/multivariate.rst>`_.
- Explore `time-varying PSDs <docs/time-varying-psd.md>`_.

Architecture
------------

See `the architecture and migration guide <docs/architecture.md>`_ for the
shared scalar/matrix model and time-frequency evaluation.
Scalar time-varying power inference is available through ``fit()``; see
`the time-varying PSD guide <docs/time-varying-psd.md>`_. Multivariate TV inference
is not yet implemented.

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
