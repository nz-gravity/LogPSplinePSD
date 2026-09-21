Five-Minute Example
===================

This example fits a stationary PSD to one noisy time series. It uses the
public ``fit`` entry point and returns a ``PSDResult``.

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

The time vector gives the sampling interval. The signal below contains a
4 Hz oscillation plus Gaussian noise.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from log_psplines import PipelineConfig, TimeSeries, fit

   rng = np.random.default_rng(7)
   fs = 64.0
   t = np.arange(512) / fs
   y = np.sin(2 * np.pi * 4 * t) + 0.5 * rng.normal(size=t.size)

   series = TimeSeries(data=y, t=t)
   result = fit(
       series,
       PipelineConfig(
           n_knots=8,
           vi_steps=200,
           n_warmup=100,
           n_samples=200,
           rng_key=7,
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
   median_psd = np.median(psd_draws, axis=(0, 1))

   plt.loglog(frequency, median_psd)
   plt.xlabel("Frequency [Hz]")
   plt.ylabel("PSD")
   plt.show()

The posterior should show a clear feature near 4 Hz. The small settings above
are for a quick check, not for a final scientific analysis. Increase the
number of spline knots and inference draws after checking that the pipeline
runs on your data.

Next steps
----------

- :doc:`results` explains the ``PSDResult`` properties and persistence.
- :doc:`configuration` describes the main analysis settings.
- :doc:`data_preprocessing` explains accepted data and frequency selection.
- :doc:`multivariate` points to the two-channel example.
