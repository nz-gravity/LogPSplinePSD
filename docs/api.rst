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

Stationary mean removal
~~~~~~~~~~~~~~~~~~~~~~~

All stationary time-domain entry points remove each channel's full-record
mean before splitting the data into blocks and applying the chosen taper.
This is a fixed convention: no means or linear trends are fitted separately
inside the blocks. The fitting pipeline combines global centering with
channel standardization, then passes the centered data directly to the FFT.
The direct FFT, Wishart and empirical Welch/CSD helpers use the same convention.

Subtracting a block mean before tapering would change its Fourier coefficients
by a frequency-dependent term:

.. math::

   \mathcal{F}\{w(x-\bar{x}_b)\}_k
   = \mathcal{F}\{wx\}_k - \bar{x}_b\mathcal{F}\{w\}_k.

For a non-rectangular taper, this term can reach retained frequencies and
distort a quiet spectrum such as LISA's low-frequency T channel. Global
centering avoids the additional block-mean term. It removes a full-record
constant offset; it does not remove time-dependent drift or eliminate the
ordinary spectral leakage and bin-averaging effects of the analysis window.

**Migration:** remove ``wishart_detrend`` from ``StationaryConfig`` and
``detrend`` from calls to ``compute_wishart``, ``TimeSeries.to_wishart_stats``
and ``empirical_spectrum`` (including ``get_empirical_psd`` forwarding).
These arguments have been removed, including their former ``False`` value;
there is no replacement selector. The window selection is unchanged.
``fit`` does not reprocess supplied ``WishartData``. Regenerate statistics
previously constructed with per-block detrending to use the new convention.

``WishartData.preprocessing`` records the applied ``mean_removal`` and
``window``. These descriptions survive masking and coarse-graining and are
saved in ``result.metadata`` and NetCDF files. When precomputed statistics
have unspecified preprocessing, the result records ``"unknown"`` rather
than inferring their history from the fitting configuration.

Stationary Gaussian marginalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both stationary NUTS and VI integrate the Gaussian cross-spectrum spline
coefficients out of the numerical fit. The diagonal-spectrum coefficients
and all smoothing scales remain unknown. Each retained draw restores the
cross-spectrum coefficients from their joint conditional Gaussian before
reconstructing the spectral matrix, so ``weights_theta_*`` and their
uncertainty remain available in ``result.posterior``. This is the only
stationary multivariate inference path; no configuration switch is needed.

VI approximates the remaining marginal posterior and restores the same exact
conditional Gaussian. It does not become an exact posterior method. With
``vi_posterior_draws <= 0``, the multivariate result contains one guide-median
point for the remaining parameters and one conditional cross-spectrum draw;
this is not a joint posterior median or an uncertainty summary.

Marginalization preserves the posterior, but raw VI losses need not match
those from the previous explicit implementation: the integrated Gaussian
includes fixed normalization constants previously omitted by the centered
coefficient prior. The Whittle likelihood also omits data-only constants.
Do not interpret these losses as directly comparable model evidence or
Bayes factors without restoring consistent normalization.

Enable ``JAX_ENABLE_X64=true`` before Python starts for stationary multivariate
fits. The first Cholesky row has no cross-spectrum coefficients to integrate.
See the :doc:`multivariate example <examples/multivariate-example>` for the
completion-of-the-square derivation and a comparison of NUTS and VI.

Multivariate time-varying Gaussian marginalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``WishartGridData`` uses the same Gaussian integration for its ANOVA
cross-spectrum fields. Conditional on the diagonal log-variance field and
smoothing scales, both the frequency-mean and time-frequency interaction
coefficients enter the complex regression linearly. They are integrated
jointly; NUTS samples only diagonal-field coefficients and smoothing scales.
There is no explicit cross-coefficient sampling option. Enable JAX float64.

The Gaussian calculation preserves the ANOVA null-mode and roughness priors.
It evaluates residual power at the conditional mean plus the prior quadratic,
avoiding subtraction of large nearly equal quadratic forms. After sampling,
one joint conditional draw restores all cross-field coefficients for each
retained draw. Original-basis ``weights_g_theta_*`` and
``weights_eta_theta_*`` remain available, and nonlinear spectral summaries
use the full reconstructed posterior. Saved ``log_likelihood_block_*`` values
are the ordinary row likelihood evaluated at these conditional draws.
``row_latent_dimensions`` counts sampled parameters; ``row_coefficient_counts``
includes reconstructed coefficients. This experimental matrix time-varying
path currently supports NUTS; its VI path is not implemented.

The mathematical stationary limit requires a zero time contribution and
matching frequency bases, Gaussian priors, counts and normalization. The
ANOVA frequency prior differs from ``StationaryConfig``'s default prior,
so the default fitters need not return identical posteriors. Tests compare
the marginal integral, gradients and conditional covariance under matched
priors, including masked observations and non-unit ENBW. A flexible ANOVA fit
to a stationary realization is a separate sampling/recovery check; it does
not by itself establish repeated-noise interval calibration.

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

.. autoclass:: log_psplines.data.WishartGridData
   :members:

Time-Varying Preprocessing
--------------------------

Local FFT and moving-periodogram transforms remove each channel's full-record
mean once before extracting windows. They preserve local mean variation;
there is no per-window detrending. ``local_wishart_grid`` accepts the ordinary
``window`` taper specification, defaulting to rectangular windows. Coefficients
are normalized by ``sqrt(2 / (fs * sum(window**2)))`` in one-sided PSD units;
DC and an even-length Nyquist ordinate are omitted.

``WishartGridData.enbw`` defaults to one and is computed for local FFT tapers.
As in the stationary likelihood, ENBW divides both the log-determinant and
quadratic terms. It changes information weighting, not ``Y/counts`` or the
integer raw counts. This is an approximation for taper-induced dependence,
not a proof that adjacent coefficients are independent. ENBW and preprocessing
history survive masking, pooling and result serialization. Externally supplied
statistics retain unknown preprocessing history unless it is supplied.

.. autofunction:: log_psplines.preprocessing.wdm.wdm_periodogram

.. autofunction:: log_psplines.preprocessing.wishart_grid.local_wishart_grid

.. autofunction:: log_psplines.preprocessing.moving_periodogram.multivariate_moving_periodogram

.. autofunction:: log_psplines.preprocessing.wishart_grid.coarse_grain_wishart_grid

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


Variational inference
---------------------

.. autofunction:: log_psplines.inference.vi.fit_vi

.. autoclass:: log_psplines.inference.vi.VIResult
   :members:
