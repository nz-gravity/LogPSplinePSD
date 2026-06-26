LogPSplinePSD
=============

``LogPSplinePSD`` estimates power spectral densities (PSDs) with Bayesian
log-P-splines. It is built around multichannel frequency-domain inference:
time series are converted into Wishart sufficient statistics, smooth spline
models are fitted with NumPyro/JAX, and results are returned as ArviZ-compatible
``xarray.DataTree`` objects.

The package is useful when you need posterior uncertainty on a univariate PSD,
a multivariate spectral matrix, cross spectra, or coherence. The implementation
keeps the main spectral invariants explicit: PSD diagonals are positive,
multivariate spectral matrices are Hermitian positive definite, and coherence
is derived from the reconstructed matrix.

What Is Included
----------------

- A high-level pipeline interface via :func:`log_psplines.mcmc.run_mcmc` and
  :func:`log_psplines.make_pipeline`.
- Data containers for time-domain series and frequency-domain Wishart
  statistics.
- P-spline models for diagonal PSD terms and complex off-diagonal structure.
- VI warm starts and factorised multivariate NUTS stages.
- Optional frequency-domain coarse graining for large frequency grids.
- ArviZ-style outputs, diagnostic summaries, posterior PSD quantiles, and
  plotting helpers.

What Is Not Covered Here
------------------------

Application-specific gravitational-wave examples are intentionally left out of
this documentation set for now. The public docs focus on package concepts,
synthetic examples, configuration, outputs, and API reference. Domain examples
can be added later as separate case studies.

Install
-------

Use the project virtual environment during development:

.. code-block:: bash

   source .venv/bin/activate
   python -m pip install -e '.[dev]'

For package use outside the repository:

.. code-block:: bash

   python -m pip install LogPSplinePSD

Where To Start
--------------

- :doc:`quickstart` shows a small synthetic run.
- :doc:`configuration` explains the main knobs in ``PipelineConfig``.
- :doc:`data_preprocessing` describes accepted inputs and FFT preprocessing.
- :doc:`outputs` explains the returned ``DataTree`` and saved diagnostics.
- :doc:`technical_notes` links the implemented likelihood to the code.
- :doc:`api` lists the most useful public classes and functions.

References
----------

.. _Eilers1996:

Eilers, P. H. C., & Marx, B. D. (1996). *Flexible smoothing with B-splines and
penalties*. Statistical Science, 11(2), 89-121.
`DOI:10.1214/ss/1038425655 <https://doi.org/10.1214/ss/1038425655>`_.

.. _MaturanaRussel2021:

Maturana-Russel, J., & Meyer, R. (2021). *P-spline spectral density estimation
with a discrete penalty*. `arXiv:1905.01832 <https://arxiv.org/abs/1905.01832>`_.
