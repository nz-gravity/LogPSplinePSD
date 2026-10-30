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

``StationaryConfig(marginalize_theta=True)`` is the default for both NUTS and
VI. It integrates the Gaussian cross-spectrum spline coefficients out of
the numerical fit, leaving the diagonal-spectrum coefficients and all
smoothing scales unknown. Each retained draw restores the cross-spectrum
coefficients from their joint conditional Gaussian before reconstructing
the spectral matrix.

Set ``marginalize_theta=False`` to sample all spline coefficients and scales
jointly. Both formulations use the same priors, bases and Wishart likelihood,
and return ``weights_theta_*``, spectra and pointwise likelihoods in the same
format. ``result.metadata["theta_parameterization"]`` records ``"marginalized"``
or ``"sampled"`` and survives NetCDF serialization. A scalar first row has
no theta coefficients and uses the explicit model with either setting.

Marginalised VI approximates the remaining marginal posterior and restores
the same exact conditional Gaussian. Explicit VI approximates the joint
posterior over all coefficients and scales. These approximating families can
differ even though the exact target is the same. With
``vi_posterior_draws <= 0``, the result contains one guide-median point plus
one conditional theta draw when marginalization is enabled; this is a
preview, not an uncertainty summary.

Marginalization preserves the posterior, but raw VI losses need not match
those from explicit fitting: the integrated Gaussian includes fixed
normalization constants omitted by the centered coefficient prior. The
Whittle likelihood also omits data-only constants.
Do not interpret these losses as directly comparable model evidence or
Bayes factors without restoring consistent normalization.

Enable ``JAX_ENABLE_X64=true`` before Python starts for stationary multivariate
fits. For marginalised rows, fixed design cross-products are cached using
exact B-spline support, with local frequency blocks avoiding products with
known zeros. Design-response
products share the 32 MiB cache budget per row; oversized designs use uncached
contractions. The stable quadratic uses its stationarity at the conditional
mean for differentiation, and the log determinant uses its exact
``trace(H^-1 dH)`` derivative from the existing precision factor. Dense,
centered marginalised NUTS rows start with an inverse-Fisher metric for the
log-innovation weights and still adapt the metric during warmup. These
optimizations preserve the likelihood, priors and joint conditional
reconstruction; finite MCMC trajectories can change.

With marginalization enabled, ``StationaryConfig(theta_backend="auto")``
uses exact banded algebra on CPU for rows with at least 256 theta
coefficients and a reordered half-bandwidth
smaller than one eighth of the coefficient count. Small or wide systems and
other device backends retain dense algebra. Set ``theta_backend="dense"`` or
``"banded"`` to select either implementation explicitly. ``theta_backend``
is ignored when ``marginalize_theta=False``; explicit fitting does not prepare
the marginalization caches or conditional-draw operators. The structural
ordering includes every overlapping design column and every nonzero prior
entry, without numerical thresholding or a change to the penalty ridge.
Banded Cholesky and triangular solves retain the full conditional Gaussian;
selected inversion computes exactly the covariance entries needed by the
log-determinant gradient. The first gradient does not form a dense covariance.

Auto selection also falls back to dense, uncached contractions if the fixed
Gram cache exceeds its memory budget; explicitly requesting ``"banded"``
raises an error in that case. The backend setting is stored in result
metadata; ``"auto"`` can select different implementations for different rows.
The current size/bandwidth rule is conservative and based on CPU measurements,
not a performance guarantee for every problem. Fixed design arrays and host
preparation still use dense storage. These optimizations target repeated
likelihood evaluation, rather than eliminating all dense setup allocations.
Marginalization trades fewer sampled coefficients for repeated matrix algebra;
neither formulation is always faster. Compare effective samples per second
and convergence on the same data and model when choosing between them.
See the :doc:`multivariate example <examples/multivariate-example>` for the
completion-of-the-square derivation and a comparison of NUTS and VI.

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


Variational inference
---------------------

.. autofunction:: log_psplines.inference.vi.fit_vi

.. autoclass:: log_psplines.inference.vi.VIResult
   :members:
