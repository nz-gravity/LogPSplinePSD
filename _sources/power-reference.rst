Power data with a reference spectrum
====================================

Supply a time-frequency power grid, a fixed reference spectrum, and optional
simulation truth to the public ``fit`` function. The package builds the scalar
tensor-product log-P-spline from ``PowerConfig``. Domain-specific construction
of the reference and truth belongs in the calling analysis.

.. code-block:: python

   import numpy as np

   from log_psplines import PowerConfig, PowerData, PowerPartition, fit

   data = PowerData(
       power=summed_power,
       counts=component_counts,
       time=time_grid,
       frequency=frequency_grid,
       units="WDM coefficient variance",
   )
   config = PowerConfig(
       n_interior_knots_time=8,
       n_interior_knots_freq=12,
       roughness_scale=10.0,
       centered=True,
       n_warmup=500,
       n_samples=1000,
   )
   result = fit(
       data,
       config,
       reference=reference_power,
       true_psd=truth_power,
   )
   result.save("results/example")

``reference_power`` and ``truth_power`` must be positive arrays matching the
native ``(time, frequency)`` grid and the units of ``data.power / data.counts``.
The reference is fixed during inference: the spline estimates a smooth log
correction to it. Truth is excluded from the likelihood and used only for
posterior checks and plots. The native reference is restored to every posterior
spectrum draw and truth is retained in ``result.truth`` and the saved NetCDF.

With a ``PowerPartition``, each native power is divided by its own reference
value before pooling. The exact component counts are pooled separately. The
result is reconstructed on the original grid. Use ``reference=None`` for a
free log-P-spline surface, or pass an explicit ``LogPSpline`` or
``ANOVALogPSpline`` model when its basis or decomposition needs to be set by
the caller.

The WDM adapter returns coefficient-variance powers, not PSD per Hz. Convert
the supplied reference and truth to those same units before calling ``fit``;
the package does not infer a transform calibration from an analytic PSD.
