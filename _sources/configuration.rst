Configuration
=============

Most user-facing behaviour is controlled by
:class:`log_psplines.pipeline.config.PipelineConfig`. The configuration is a
flat dataclass so runs can be saved, logged, and reproduced without nested
state.

Minimal Configuration
---------------------

.. code-block:: python

   from log_psplines.pipeline.config import PipelineConfig

   config = PipelineConfig(
       n_knots=8,
       n_warmup=500,
       n_samples=1000,
       rng_key=42,
       outdir="runs/example",
   )

Spline Options
--------------

``n_knots``
   Number of interior spline knots. It may be an integer shared by all
   components or a dictionary for component families.

``degree``
   B-spline degree. The default is cubic splines.

``diffMatrixOrder``
   Difference order for the P-spline penalty. The default penalises second
   differences.

``knot_kwargs``
   Extra keyword arguments passed to knot initialisation. Use this for
   specialised knot placement while keeping the pipeline interface stable.

Frequency Selection
-------------------

``fmin`` and ``fmax``
   Optional lower and upper frequency limits in Hz. The DC bin is always
   dropped before fitting.

``exclude_freq_bands``
   Tuple of ``(low, high)`` bands to remove after applying ``fmin`` and
   ``fmax``. This is useful for known contaminated bands.

``Nb``
   Number of non-overlapping time-domain blocks used to build Wishart
   statistics. ``Nb`` must divide the number of samples.

``wishart_window`` and ``wishart_detrend``
   Optional block taper and detrending mode used during FFT preprocessing.
   Non-rectangular windows apply an equivalent-noise-bandwidth correction in
   the likelihood.

VI and NUTS
-----------

``init_from_vi``
   Run stochastic variational inference before NUTS and use the fitted guide to
   initialise the sampler. Enabled by default.

``only_vi``
   Stop after the VI stage. This is the fastest way to check data scaling,
   frequency selection, and spline flexibility.

``vi_steps``, ``vi_lr``, ``vi_guide``
   Optimisation settings for the VI warm start.

``n_warmup``, ``n_samples``, ``num_chains``
   Standard NUTS run length controls.

``target_accept_prob`` and ``max_tree_depth``
   NumPyro NUTS tuning controls. Per-channel values can be supplied with
   ``target_accept_prob_by_channel`` and ``max_tree_depth_by_channel``.

Coarse VI and Coarse Graining
-----------------------------

``coarse_grain_config``
   Coarse grain the frequency grid used by the full inference stage.

``coarse_grain_config_vi``
   Coarse grain only the VI warm-start stage.

``auto_coarse_vi``
   Let the pipeline choose a coarse VI grid when the full grid is large.

``use_coarse_vi_for_init``
   Use the coarse VI result as the initial point for full-resolution inference.

Output and Evidence
-------------------

``outdir``
   If set, the pipeline writes NetCDF inference data, posterior predictive
   plots, and diagnostic tables/figures.

``compute_lnz``
   Estimate log evidence with MorphZ when possible. Leave as ``None`` to use
   the pipeline default.

``true_psd``
   Optional reference PSD used only for diagnostics and error summaries. It can
   be an array aligned to the analysis grid or a ``(freq, psd)`` tuple to be
   interpolated.
