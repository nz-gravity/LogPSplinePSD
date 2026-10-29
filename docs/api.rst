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
Log variances are not clipped: stationary and matrix-grid fits use the same
likelihood, including after a change of physical units. Inputs and calculations
must remain representable in the chosen floating-point precision.
See the :doc:`multivariate example <examples/multivariate-example>` for the
completion-of-the-square derivation and a comparison of NUTS and VI.

Automatic spline coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every Gaussian spline field that remains numerically sampled uses automatic
information-scaled coordinates: stationary scalar and multivariate delta
fields, scalar power tensor/ANOVA fields, and matrix grid/paired delta fields.
Gaussian cross-spectrum coefficients remain analytically marginalized.
There is no centered/noncentered selector. Remove the former
``StationaryConfig.smoothing_parameterization`` and ``PowerConfig.centered``
arguments; no replacement argument is needed.

In the penalty eigenbasis, let ``a_k`` be a coefficient's current prior
standard deviation and ``h_k`` its fixed expected log-variance information.
The model samples

.. math::

   c_k=(a_k^{-2}+h_k)^{-1/2},\qquad
   z_k\sim N(0,a_k^2/c_k^2),\qquad s_k=c_k z_k.

The physical coefficient still has exactly ``s_k ~ Normal(0, a_k)``. The
Normal density includes the parameter-dependent transformation Jacobian;
independent tests check both density and smoothing-scale gradients. Each
model keeps its own penalty, ridge, null-mode prior and hyperprior.

The information is computed once from squared eigenbasis design entries and
observation counts. Complex stationary observations use ``Nb*Nh*eta/enbw``;
matrix cells use ``counts/enbw``; scalar real-power components use
``counts/2``. Masked cells contribute zero. No truth spectrum or preliminary
posterior fit is used. Tight prior directions approach noncentering, while
well-measured directions approach fixed likelihood scaling.

The same coordinates are used by NUTS and supported VI fits. VI still makes
an approximation: changing coordinates changes what a restricted Gaussian
guide can represent, so improved VI accuracy is not guaranteed. Physical
weights and spectra remain in the posterior, alongside sampled coordinates.
Parametric model parameters without Gaussian spline priors are unchanged.

Multivariate Cholesky splines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The target is the Hermitian spectral density matrix :math:`S(t,f)`: its
diagonal entries are channel PSDs, and its off-diagonal entries are complex
cross-spectra. Write :math:`h_j=\log D_{jj}` and let :math:`T` be unit lower
triangular with :math:`T_{jl}=-\theta_{jl}` for :math:`j>l`. Then

.. math::

   D=\operatorname{diag}(e^{h_1},\ldots,e^{h_p}),\qquad
   S^{-1}=T^\dagger D^{-1}T,\qquad S=T^{-1}D(T^{-1})^\dagger.

There are :math:`p^2` real spline fields: :math:`p` log innovation variances
and the real and imaginary parts of :math:`p(p-1)/2` regression coefficients.
The regression coefficients are not themselves CSD entries. For two channels,

.. math::

   S=\begin{pmatrix}
      e^{h_1} & e^{h_1}\theta_{21}^*\\
      e^{h_1}\theta_{21} & e^{h_2}+e^{h_1}|\theta_{21}|^2
   \end{pmatrix}.

Every reconstructed draw is positive definite. Elementwise posterior medians
and quantiles need not themselves form a positive-definite matrix.
The :doc:`multivariate time-varying tutorial <examples/multivariate-timevarying-example>`
walks through a simulated complex-coefficient example, reconstruction and
sampling diagnostics.

``WishartGridData`` uses ordinary ``LogPSpline`` fields: one spline for each
log innovation variance and each real or imaginary regression coefficient.
A frequency-only model has ``time=None``. A tensor model evaluates each
field as ``Bt @ W @ Bf.T``, or at exact paired coordinates. There is no
multivariate ANOVA decomposition or alternative matrix ANOVA path.

At each observed cell, normalized complex coefficients have covariance
:math:`S(t,f)`. With :math:`Y=\sum_{r=1}^{n}x_rx_r^\dagger`, the likelihood is

.. math::

   \log L=-\sum_{t,f}\left[n_{tf}\log\det S(t,f)
           +\operatorname{tr}\{S(t,f)^{-1}Y_{tf}\}\right]+\mathrm{constant}

for independent zero-mean proper complex Gaussian coefficients and unit ENBW.
The local Fourier frontends use this as a Whittle approximation; tapering,
leakage and dependence between observations affect that approximation. The
stored ENBW divides the log-likelihood as described below. Real same-cell WDM
outer products describe a real covariance, not the full complex CSD. A multivariate WDM observation
model connecting cross-cell covariances to :math:`S(t,f)` is not implemented.

Conditional on the diagonal field and smoothing scales, cross-spectrum
coefficients still enter the complex regression linearly. They are integrated
jointly; NUTS samples diagonal coefficients and smoothing scales. One joint
conditional Gaussian draw restores every cross-field coefficient for each
retained draw. There is no explicit cross-coefficient sampling option.
Enable ``JAX_ENABLE_X64=true``. Matrix grid/paired inference currently
supports NUTS; its VI path is not implemented.

For time-major vectorized coefficients, the Gaussian precision is

.. math::

   Q = \sigma_t^{-2}(P_t\otimes I_f)
       + \sigma_f^{-2}(I_t\otimes P_f).

Each field has independent HalfNormal smoothing scales, with the scale
specified by ``PowerConfig.roughness_scale``. The frequency penalty ``Pf``
is positive definite and includes its existing ridge; the time penalty
``Pt`` is positive semidefinite. No additional tensor ridge or special
null-mode prior is added. The default matrix frequency basis uses the
stationary max-normalization and ridge convention. A supplied model retains
its supplied penalty. With no explicit configuration, the matrix entry point
uses roughness scale 1.28; an explicit
``PowerConfig`` controls its own settings.

The Gaussian calculation evaluates residual power at the conditional mean
plus the prior quadratic, avoiding subtraction of large nearly equal terms.
Original-basis coefficient draws remain in ``result.posterior``. Nonlinear
spectral summaries use the reconstructed posterior, and saved
``log_likelihood_block_*`` values are the ordinary row likelihood at these
draws. ``row_latent_dimensions`` counts sampled parameters;
``row_coefficient_counts`` also includes reconstructed coefficients.

The exact stationary restriction is ``LogPSpline(frequency_basis, time=None)``:
the time basis is the constant one, the time penalty is zero, and no unused
time smoothing parameter is sampled. Its frequency precision is exactly
``Pf / sigma_f**2``. Equality with a stationary fit also requires matching
the basis, penalty, smoothing hyperprior, counts, duration normalization
and ENBW. Tests compare marginal integrals, gradients and conditional
covariances under that restriction, including masks and non-unit ENBW.
Increasing a second-order time penalty alone does not enforce stationarity,
because a linear time trend is also in its null space.

Stationary simulation controls precede time-varying recovery studies.
Passing algebraic tests or one fitted realization does not establish
repeated-noise interval calibration. Marginalization reduces the sampled
dimension but introduces a coefficient-sized Gaussian factorization per
likelihood evaluation; runtime still depends on basis size and sampler geometry.

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
