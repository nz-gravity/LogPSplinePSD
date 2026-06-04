API Reference
=============

This page documents the public entry points most users need. Lower-level
helpers remain importable from their modules, but are not all part of the
stable user surface.

High-Level Pipeline
-------------------

.. autofunction:: log_psplines.mcmc.run_mcmc

.. autofunction:: log_psplines.pipeline.make_pipeline.make_pipeline

.. autoclass:: log_psplines.pipeline.config.PipelineConfig
   :members:
   :undoc-members:

Data Containers
---------------

.. autoclass:: log_psplines.datatypes.multivar.MultivariateTimeseries
   :members:

.. autoclass:: log_psplines.datatypes.multivar.MultivarFFT
   :members:

.. autoclass:: log_psplines.datatypes.multivar.EmpiricalPSD
   :members:

Spline Models
-------------

.. autoclass:: log_psplines.psplines.psplines.LogPSplines
   :members:

.. autofunction:: log_psplines.psplines.psplines.build_spline

.. autoclass:: log_psplines.psplines.multivar_psplines.MultivariateLogPSplines
   :members:

Knot Initialisation
-------------------

.. autofunction:: log_psplines.psplines.knots_locator.knot_locator.init_knots

Coarse Graining
---------------

.. autoclass:: log_psplines.preprocessing.coarse_grain.CoarseGrainConfig
   :members:

.. autoclass:: log_psplines.preprocessing.coarse_grain.CoarseGrainSpec
   :members:

.. autofunction:: log_psplines.preprocessing.coarse_grain.compute_binning_structure

.. autofunction:: log_psplines.preprocessing.coarse_grain.apply_coarse_grain_multivar_fft

ArviZ Helpers
-------------

.. autofunction:: log_psplines.arviz_utils.open_inference_data

.. autofunction:: log_psplines.arviz_utils.save_inference_data

.. autofunction:: log_psplines.arviz_utils.get_psd_dataset

.. autofunction:: log_psplines.arviz_utils.get_multivar_posterior_psd_quantiles

.. autofunction:: log_psplines.arviz_utils.get_multivar_vi_psd_quantiles

.. autofunction:: log_psplines.arviz_utils.get_weights
