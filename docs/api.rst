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

Posterior summary semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``result.spectrum`` stores reconstructed draws. Stationary fits currently store
every draw; ``PowerConfig.spectrum_draws`` can limit power fits to a preview.
``result.psd`` and ``result.coherence`` describe those stored draws.
``result.spectrum_summary`` contains all-draw summaries with the posterior chain
and draw counts attached to each variable, including after a NetCDF round trip.

``result.quantiles()`` returns cached all-draw quantiles when available, or
computes them from complete stored draws. A preview alone raises an error.
``kind="real"``, ``"imag"``, ``"magnitude"``, and ``"coherence"`` select the
quantity summarized. Magnitude and coherence require their own cached quantiles
or complete draws; they cannot be derived from elementwise spectral quantiles.
Requested percentiles absent from a preview's cache must be reconstructed from
the saved posterior and model data. Parametric results also need the original
forward model.

.. autofunction:: log_psplines.models.reconstruction.compute_psd_quantiles

This lower-level routine reconstructs multivariate draws in frequency chunks.
It uses every chain and draw by default. An explicit ``n_samples_max`` produces
a deliberately limited summary. Its real, imaginary, and optional coherence
quantiles share the same reduction implementation as ``PSDResult.quantiles``.

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

.. autofunction:: log_psplines.plotting.plot_basis_diagnostics

Inspect the basis actually used by a model, including its knots and penalty:

.. code-block:: python

   from log_psplines.plotting import plot_basis_diagnostics

   fig, axes = plot_basis_diagnostics(model.frequency, path="basis.png")

For stationary component models use
``components.diagonal_models[channel].frequency``; for a tensor model inspect
``model.frequency`` and ``model.time`` separately. The diagnostic plots these
stored operators directly and does not construct replacement bases.

.. autofunction:: log_psplines.plotting.compute_confidence_intervals

This generic draw-array helper supports percentile and simultaneous bands.
It is separate from spectral-matrix summaries; use ``PSDResult.quantiles`` for
fitted spectra and their nonlinear transformations.

.. autofunction:: log_psplines.diagnostics.sampling_diagnostics

.. autofunction:: log_psplines.diagnostics.spectrum_diagnostics

.. autofunction:: log_psplines.diagnostics.save_diagnostics


Spectral conventions
--------------------

.. include:: conventions.rst
   :start-line: 3


Reference-spectrum power fits
-----------------------------

.. include:: power-reference.rst
   :start-line: 3
