LogPSplinePSD
=============

``LogPSplinePSD`` estimates power spectral densities (PSDs) with Bayesian
log-P-splines. It is built around multichannel frequency-domain inference:
time series are converted into Wishart sufficient statistics, smooth spline
models are fitted with NumPyro/JAX. Time-varying PSDs are a WIP.



<video autoplay muted loop playsinline preload="metadata">
  <source src="animation/logpspline-hero.webm" type="video/webm">
</video>



Install
-------


For package use outside the repository:

.. code-block:: bash

   python -m pip install LogPSplinePSD


Use the project virtual environment during development:

.. code-block:: bash

   source .venv/bin/activate
   python -m pip install -e '.[dev]'



Where To Start
--------------

Follow the pages in this order:

1. :doc:`five-minute` runs a small univariate analysis.
2. :doc:`results` explains the returned ``PSDResult``.
3. :doc:`configuration` covers the settings for a real analysis.
4. :doc:`multivariate` introduces spectral matrices and coherence.
5. :doc:`time-varying-psd` introduces scalar time-varying PSDs.
6. :doc:`outputs` documents saved files and diagnostic checks.
7. The remaining pages contain preprocessing details, technical notes, and the
   :doc:`api` reference.

References
----------

.. _Eilers1996:

Eilers, P. H. C., & Marx, B. D. (1996). *Flexible smoothing with B-splines and
penalties*. Statistical Science, 11(2), 89-121.
`DOI:10.1214/ss/1038425655 <https://doi.org/10.1214/ss/1038425655>`_.

.. _MaturanaRussel2021:

Maturana-Russel, J., & Meyer, R. (2021). *P-spline spectral density estimation
with a discrete penalty*. `arXiv:1905.01832 <https://arxiv.org/abs/1905.01832>`_.
