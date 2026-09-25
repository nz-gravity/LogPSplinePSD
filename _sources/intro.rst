LogPSplinePSD
=============

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

Follow the pages in this order:

1. :doc:`examples/univar-example` runs a univariate analysis.
2. :doc:`examples/multivariate-example` introduces spectral matrices.
3. :doc:`examples/timevarying-example` fits scalar time-varying spectra.
4. :doc:`wdm-power` shows array-based WDM likelihood compression.
5. :doc:`conventions` explains units and spectral conventions.
6. :doc:`development` gives implementation notes, followed by the :doc:`api`
   reference.

References
----------

.. _Eilers1996:

Eilers, P. H. C., & Marx, B. D. (1996). *Flexible smoothing with B-splines and
penalties*. Statistical Science, 11(2), 89-121.
`DOI:10.1214/ss/1038425655 <https://doi.org/10.1214/ss/1038425655>`_.

.. _MaturanaRussel2021:

Maturana-Russel, J., & Meyer, R. (2021). *P-spline spectral density estimation
with a discrete penalty*. `arXiv:1905.01832 <https://arxiv.org/abs/1905.01832>`_.
