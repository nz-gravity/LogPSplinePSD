Understanding ``PSDResult``
============================

``fit(data, config)`` returns a ``PSDResult``. It keeps the posterior samples,
the frequency grid, and convenient reconstructed spectral quantities together.

Core properties
---------------

``result.frequency``
   The retained positive-frequency grid in Hz. DC is removed by preprocessing.

``result.psd``
   Auto-spectral posterior draws. For a univariate stationary fit, the shape is
   ``(chain, draw, frequency)``. For a multivariate fit, channel dimensions are
   appended after frequency.

``result.spectral_density``
   Reconstructed spectral-matrix draws with shape
   ``(chain, draw, frequency, channel, channel_aux)``. Time-varying scalar fits
   add a ``time`` axis before ``frequency``.

``result.coherence``
   Coherence reconstructed from the spectral matrix. It is available for
   multivariate fits.

``result.posterior``
   The posterior as an ``xarray.Dataset``. Use this when you need named
   coordinates or model parameters rather than reconstructed spectra.

Save and reload
---------------

Use ``to_netcdf`` when you only need the inference data:

.. code-block:: python

   result.to_netcdf("runs/example/inference_data.nc")

Use ``save`` to write inference data, summary tables, plots, and diagnostics:

.. code-block:: python

   result.save("runs/example")

Reload a saved inference result with:

.. code-block:: python

   from log_psplines import PSDResult

   result = PSDResult.from_netcdf("runs/example/inference_data.nc")

For the files written by ``save`` and the checks to perform before interpreting
an analysis, see :doc:`outputs`.
