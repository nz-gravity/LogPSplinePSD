LogPSplinePSD
=============

``LogPSplinePSD`` estimates power spectral densities (PSDs) with Bayesian
log-P-splines. It supports univariate and multivariate time series, fits smooth
spectral matrices with NumPyro/JAX, and returns ArviZ-compatible
``xarray.DataTree`` outputs for diagnostics and plotting.

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

Quick Example
-------------

.. code-block:: python

   from log_psplines.example_datasets.varma_data import VARMAData
   from log_psplines.mcmc import run_mcmc
   from log_psplines.pipeline.config import PipelineConfig

   data = VARMAData(n_samples=256, fs=64.0, seed=7)

   idata = run_mcmc(
       data.ts,
       PipelineConfig(
           n_knots=6,
           n_warmup=50,
           n_samples=100,
           vi_steps=200,
           outdir="runs/varma_quickstart",
       ),
   )

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
