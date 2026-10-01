Home
====

``LogPSplinePSD`` estimates power spectral densities (PSDs) with Bayesian
log-P-splines. It is built around multichannel frequency-domain inference:
time series are converted into Wishart sufficient statistics, smooth spline
models are fitted with NumPyro/JAX. Scalar time-varying inference is also
available for WDM and moving-periodogram powers.


.. raw:: html

    <video autoplay muted loop playsinline preload="metadata">
       <source src="logpspline-hero.webm" type="video/webm">
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

1. :doc:`Stationary PSD <examples/univar-example>` fits a single-channel spectrum.
2. :doc:`Multivariate stationary PSD <examples/multivariate-example>` fits spectral matrices and coherence.
3. :doc:`Time-varying PSD <examples/timevarying-example>` fits scalar WDM and moving-periodogram powers.

The :doc:`api` covers configuration, results, spectral conventions and advanced
reference-spectrum fits. Contributor instructions are in the repository's
`CONTRIBUTING.md <https://github.com/nz-gravity/LogPSplinePSD/blob/main/CONTRIBUTING.md>`_.

References
----------

.. _Eilers1996:

Eilers, P. H. C., & Marx, B. D. (1996). *Flexible smoothing with B-splines and
penalties*. Statistical Science, 11(2), 89-121.
`DOI:10.1214/ss/1038425655 <https://doi.org/10.1214/ss/1038425655>`_.

.. _MaturanaRussel2021:

Maturana-Russel, J., & Meyer, R. (2021). *P-spline spectral density estimation
with a discrete penalty*. `arXiv:1905.01832 <https://arxiv.org/abs/1905.01832>`_.

Vajpeyi, A., Meyer, R., Maturana-Russel, P., & Liu, J. (2026).
*Multivariate Bayesian P-spline estimation of spectral density matrices,
with application to LISA TDI noise*.
`arXiv:2607.04833 <https://arxiv.org/abs/2607.04833>`_.
`DOI:10.48550/arXiv.2607.04833 <https://doi.org/10.48550/arXiv.2607.04833>`_.

Download the `BibTeX citation <https://github.com/nz-gravity/LogPSplinePSD/blob/main/CITATION.bib>`_.
