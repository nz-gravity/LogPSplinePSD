API Reference
=============

This page documents the public entry points most users need. Lower-level
helpers remain importable from their modules, but are not all part of the
stable user surface.

Fitting and results
-------------------

.. autofunction:: log_psplines.fit.fit

.. autoclass:: log_psplines.results.PSDResult
   :members:

.. autoclass:: log_psplines.config.StationaryConfig
   :members:
   :undoc-members:

.. autoclass:: log_psplines.config.PowerConfig
   :members:

Data Containers
---------------

.. autoclass:: log_psplines.data.TimeSeries
   :members:

.. autoclass:: log_psplines.data.WishartData
   :members:

.. autoclass:: log_psplines.data.EmpiricalPSD
   :members:

.. autoclass:: log_psplines.data.PowerData
   :members:

Time-Varying Preprocessing
--------------------------

.. autofunction:: log_psplines.preprocessing.wdm.wdm_periodogram

.. autofunction:: log_psplines.preprocessing.moving_periodogram.moving_periodogram

.. autofunction:: log_psplines.preprocessing.moving_periodogram.scattered_moving_periodogram

.. autoclass:: log_psplines.preprocessing.power_partition.PowerPartition
   :members:

.. autofunction:: log_psplines.preprocessing.power_partition.mask_power

.. autofunction:: log_psplines.preprocessing.power_partition.select_power_partition

.. autofunction:: log_psplines.preprocessing.power_partition.coarse_grain_power

Spline Models
-------------

.. autoclass:: log_psplines.basis.SplineBasis
   :members:

.. autoclass:: log_psplines.models.matrix.SpectralMatrix
   :members:

.. autoclass:: log_psplines.models.spectrum.LogPSpline
   :members:

.. autofunction:: log_psplines.models.spectrum.build_spline

.. autoclass:: log_psplines.inference.components.SpectralComponents
   :members:

Knot Initialisation
-------------------

.. autofunction:: log_psplines.preprocessing.knot_locator.init_knots

.. autoclass:: log_psplines.preprocessing.knot_locator.Component
   :members:

.. autofunction:: log_psplines.preprocessing.knot_locator.variation_profiles

.. autofunction:: log_psplines.preprocessing.knot_locator.quantile_knots

.. autofunction:: log_psplines.preprocessing.knot_locator.allocate_components

Coarse Graining
---------------

.. autoclass:: log_psplines.preprocessing.coarse_grain.CoarseGrainConfig
   :members:

.. autoclass:: log_psplines.preprocessing.coarse_grain.CoarseGrainSpec
   :members:

.. autofunction:: log_psplines.preprocessing.coarse_grain.compute_binning_structure

.. autofunction:: log_psplines.preprocessing.coarse_grain.apply_coarse_grain_multivar_fft


Diagnostics
-----------

.. autofunction:: log_psplines.diagnostics.sampling_diagnostics

.. autofunction:: log_psplines.diagnostics.spectrum_diagnostics

.. autofunction:: log_psplines.diagnostics.save_diagnostics
