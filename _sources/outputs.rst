Outputs and Diagnostics
=======================

Return Value
------------

``run_mcmc`` returns an ``xarray.DataTree``. Important groups include:

``posterior``
   NUTS posterior samples for spline weights and model parameters.

``sample_stats``
   Per-channel sampler diagnostics such as acceptance rate, step size, tree
   depth, and log probability.

``observed_data``
   Frequency grid and empirical PSD-like data derived from the Wishart
   statistics.

``vi_posterior`` and ``vi_sample_stats``
   VI draws, losses, and warm-start diagnostics when VI is enabled.

``prior_predictive`` and ``posterior_predictive``
   Reconstructed spectral quantities used by plotting and diagnostics when
   available.

Saved Files
-----------

When ``PipelineConfig(outdir=...)`` is set, the pipeline writes:

``inference_data.nc``
   NetCDF serialisation of the returned ``DataTree``.

``posterior_predictive.png``
   PSD matrix summary plot.

``diagnostics/vi_summary.csv``
   VI convergence and loss summary.

``diagnostics/nuts_summary.csv``
   NUTS diagnostics and, when a truth PSD is supplied, error metrics.

``diagnostics/vi_loss.png``
   VI loss trace.

``diagnostics/traces.png`` and ``diagnostics/energy.png``
   Standard MCMC trace and energy diagnostics.

Some files are conditional. For example, preprocessing eigenvalue plots are
written for multivariate frequency-domain inputs when an output directory is
available.

Loading Results
---------------

.. code-block:: python

   from log_psplines.arviz_utils import open_inference_data

   idata = open_inference_data("runs/example/inference_data.nc")

Extracting PSD Summaries
------------------------

.. code-block:: python

   from log_psplines.arviz_utils import (
       get_multivar_posterior_psd_quantiles,
       get_psd_dataset,
   )

   psd_draws = get_psd_dataset(idata, source="posterior")
   q = get_multivar_posterior_psd_quantiles(idata)

``get_psd_dataset`` returns posterior draws when available. Quantile helpers
return compact arrays suitable for plotting and reporting.

Diagnostics Checklist
---------------------

- Check that posterior PSD diagonals are positive.
- For multivariate runs, check Hermitian symmetry and positive definiteness of
  reconstructed spectral matrices.
- Check coherence lies in ``[0, 1]``.
- Inspect NUTS divergences, tree-depth hits, and effective sample size.
- Compare VI and NUTS posterior summaries when using VI warm starts.
- If ``true_psd`` was supplied, review RIAE, L2, and coverage metrics in the
  saved summaries.
