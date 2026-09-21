Five-Minute Example
===================

This example fits a stationary PSD to a simulated univariate AR(4) time
series. The example dataset provides the theoretical PSD, so the posterior can
be compared with a known truth. It uses the public ``fit`` entry point and
returns a ``PSDResult``.

Install first
-------------

For a package installation:

.. code-block:: bash

   python -m pip install LogPSplinePSD

For a checkout of this repository:

.. code-block:: bash

   source .venv/bin/activate
   python -m pip install -e '.[dev]'

Create and fit a signal
-----------------------

The repository's ``VARMAData.ar`` helper creates a reproducible AR(4) series
and exposes its theoretical PSD through ``get_true_psd``. The fit uses 16
density-adaptive knots, which place more spline flexibility where the observed
spectral structure is concentrated.

.. code-block:: python

   from log_psplines import PipelineConfig, fit
   from log_psplines.example_datasets import VARMAData

   data = VARMAData.ar(
      order=4,
         n_samples=8192,
      fs=64.0,
      seed=7,
   )
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

Inspect the result
------------------

``PSDResult.frequency`` is the retained frequency grid. For a univariate fit,
``PSDResult.psd`` contains posterior draws with shape ``(chain, draw,
frequency)``.

.. code-block:: python

   frequency = result.frequency
   psd_draws = result.psd

The rendered example below is produced from the same fixed seed and small
inference settings. The black line is the theoretical AR(4) PSD; the blue line
is the posterior median and the shaded region is its 90% posterior interval.
The final two frequency samples are omitted because the example truth is not
reliable at that boundary.

.. image:: _static/five-minute-psd.png
   :alt: Posterior median and 90 percent interval overlaid with the theoretical AR(4) PSD on a linear frequency axis
   :width: 85%

The posterior should follow the broad structure of the truth. A longer series
provides more frequency resolution and more information per fitted spectrum;
that is why this example uses 8192 samples rather than the smaller 2048-sample
smoke test. The inference settings are still for a quick check, not for a
final scientific analysis.

Next steps
----------

- :doc:`results` explains the ``PSDResult`` properties and persistence.
- :doc:`configuration` describes the main analysis settings.
- :doc:`data_preprocessing` explains accepted data and frequency selection.
- :doc:`multivariate` points to the two-channel example.
