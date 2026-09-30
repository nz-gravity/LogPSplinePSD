Multivariate rectangular GridTV
===============================

``WishartGridData`` retains complex cross-channel information. The public
workflow uses ``fit(data, PowerConfig(structure="anova"))`` and returns the
existing ``PSDResult`` with dimensions
``(chain, draw, time, frequency, channel, channel_aux)``. Each modified-Cholesky
row is sampled independently using the existing NUTS runner. This is a
rectangular proper-complex path; scattered multivariate data and real WDM
coefficients require different statistical support. For scalar time-varying
work, use ``wdm_periodogram`` and ``PowerData``. The local FFT adapter on this
page is an additional complex-data workflow, not a multivariate WDM adapter.

Preparation and normalization
-----------------------------

For coefficient vectors with covariance equal to the physical spectrum,
``Y_B = sum_i x_i x_i^H = U_B U_B^H``. Store sums and the integer number of
independent proper complex observations in each cell. Factor arrays have
shape ``(T,F,C,R)``; counts have shape ``(T,F)`` or can be supplied as a scalar.
The compact factor width ``R <= C`` is independent of the count. A single
observation may have rank-one statistics even with several channels.

The likelihood omits data-only constants:

.. math::

   \log L = -\sum_B [n_B\log\det S_B +
                         \operatorname{tr}(S_B^{-1}Y_B)].

The count multiplies the log determinant once; the quadratic already contains
summed statistics. Zero-count cells are neutralized before residuals and
exponentials, so their entire likelihood and gradients vanish. Active factors
and counts are validated. Factor compression uses the existing eigensolver,
clips only roundoff negative eigenvalues, and adds no diagonal power.

``local_wishart_grid(series, segment_length)`` supplies a minimal conversion
from ``TimeSeries``. It uses complete, nonoverlapping rectangular segments:
``x = sqrt(2/(fs*L)) * rfft(segment)``. Thus ``E[x x^H]`` approximates the
one-sided PSD in the units of the input data squared per Hz. There is no
standardization, overlap, taper, detrending, or ENBW correction. Time is the
mean sample time of the segment; frequency is ``k*fs/L``. DC and the even-length
Nyquist coefficient are excluded because they are real. Odd-length segments
retain every strictly positive interior frequency.

Independence, properness, and local stationarity are Whittle approximations for
finite colored/time-varying data. Disjoint segments do not establish exact
independence in such records. This implementation does not assign effective
counts to overlapping/correlated transforms. ``from_coefficients`` rejects
real-valued arrays; casting WDM coefficients to complex cannot establish a
compatible observation model.

The normalized grid path calls the existing Wishart row likelihood with
``duration=enbw=eta=1`` and without clipping the log spectrum. Historical
stationary FFT scaling and clipping remain the defaults for stationary fits.
For scalar comparison, one proper complex observation corresponds to *two*
real-component counts in ``power_whittle_log_likelihood``: use summed power
``2*Y`` and count ``2*n``. A real WDM power with count one has a different
observation convention.

Fixed bins and reference grids
------------------------------

.. code-block:: python

   from log_psplines import (
       PowerConfig, local_wishart_grid, coarse_grain_wishart_grid, fit,
   )

   native = local_wishart_grid(series, segment_length=64)
   # Default time_bin=frequency_bin=1 performs no pooling.
   likelihood_data = coarse_grain_wishart_grid(
       native, time_bin=1, frequency_bin=2,
   )
   result = fit(likelihood_data, PowerConfig(structure="anova"))

Pooling sums outer-product statistics and counts, including for already-pooled
inputs. Smaller trailing rectangles retain their actual observations. Bin
coordinates are unweighted means of the *input cell coordinates*; recursively
pooling existing bins therefore need not reproduce the coordinates of a
single pooling operation on the original grid. The original ``reference_time``
and ``reference_frequency`` survive either operation and masking with
``native.mask(observed_cells)``.

One bin-centre spectrum is exact under the adopted likelihood only if the
spectrum is constant within the bin. Otherwise it is an approximation to
diagonal power and complex cross-spectrum magnitude/phase. Time pooling loses
within-bin temporal information. The posterior output on the finer original
grid is interpolation, not recovery of discarded observations.

ANOVA priors and reconstruction
-------------------------------

Every scalar Cholesky field uses ``h(t,f)=g(f)+eta(t,f)``. The fields are the
diagonal log variances and signed real/imaginary theta components, ordered by
rows of the strict lower triangle. Only diagonal log variances are
exponentiated. Reconstruction reuses
``S=inv(T) D inv(T)^H``, ``T[j,l]=-theta[j,l]``, and ``D[j,j]=exp(v[j])``.

The time basis is centred once on the preserved reference time grid, before
masking or pooling. Its reduced-basis transform and transformed penalty are
reused at every likelihood location and output chunk. Zero mean is defined on
that reference grid, not on observed cells, bin centres, or arbitrary finer
evaluation grids. An explicit ``ANOVALogPSpline`` must use that same time grid;
its frequency grid may specify finer in-domain output locations. Its ``design``
method can evaluate finer time coordinates with the fixed transform.

The scalar ANOVA hierarchy is shared without changing its prior: independent
``sigma_g ~ HalfNormal(roughness_scale)`` and
``sigma_eta ~ HalfNormal(interaction_scale)`` for *each field*, with the existing
penalty eigenbasis, frequency null precision, ridge, and deviation null-space
treatment. Deterministic designs and penalty factorizations are shared.
Hyperparameters and coefficients are never shared between fields or row fits.
``PowerConfig.centered=False`` (the default) samples standard-Normal ``z_g``
and ``z_eta`` coordinates and applies the learned conditional scales at every
evaluation. Physical ``g``/``eta`` coefficients remain available for
reconstruction and diagnostics. ``centered=True`` samples the physical
coefficients directly. Both parameterizations have the same prior; the
standard-Normal coordinates help when an interaction scale approaches zero.
The setting now applies to scalar ANOVA and complex GridTV as well as tensor
models. Previously ANOVA ignored it and always used centered coefficients.

For ``K=Kf*(1+Kt_eta)``, row ``j`` has ``K*(1+2*j)`` coefficients and
``2*(1+2*j)`` hyperparameters. Total coefficients are ``C**2*K``. Metadata reports
actual row sizes. Fit one modest basis first; time centring requires its
reduced basis to have full rank on the reference grid.

Posterior sites store coefficients, not evaluated surfaces. Joint posterior
draws pair the same chain/draw indices from independent row fits. The existing
frequency-chunked reducer constructs full matrices and all-draw summaries.
``spectrum_draws`` retains an evenly spaced preview per chain;
``spectrum_chunk_size`` limits intermediate frequency arrays. ``quantiles()``
uses cached 5/50/95 entrywise summaries over *all* draws; componentwise complex
quantiles need not themselves be positive-definite matrices. The cached
``coherence_quantiles`` are quantiles of coherence computed per full draw.
``result.coherence`` refers to the materialized preview draws.
``spectrum_diagnostics`` uses the cached median of per-draw coherence for its
coherence error, so reducing the stored preview does not change this metric.

The final complex128 preview still occupies
``16*num_chains*retained_draws*T*F*C**2`` bytes. Chunking does not reduce that
allocation. The fit logs its estimated size before allocating; all-draw
summaries also occupy output-grid memory. ``model_data`` saves the fixed output
bases and time transform, allowing coefficient reconstruction after a native
NetCDF round trip. Host evaluation and matrix reconstruction retain NumPy
float64 precision independently of JAX's inference setting. JAX inference
still follows ``JAX_ENABLE_X64``; use float64 for physical scales outside the
float32 likelihood's numerical range. NetCDF storage retains the project's existing HDF5 complex
extension convention. ArviZ conversion, per-row sampling diagnostics, and
diagonal surface plots use the existing helpers.

Reproducible example and benchmark
----------------------------------

From the repository root:

.. code-block:: bash

   JAX_ENABLE_X64=1 .venv/bin/python docs/examples/multivariate_gridtv.py \
     --channels 2 \
     --outdir tests/test-output/tvvar-gridtv

The command uses the existing ``TVVARData`` simulator, local Fourier
preprocessing, public ``fit``, native results, plots, and diagnostics. It fits
the same realization without pooling, with frequency-only bins of width two,
and with time-frequency bins of width two in each axis. The basis, priors,
reference centring, seeds, and sampler settings remain fixed. ``--channels 3``
exercises a modest three-row feasibility case.
``--time-knots`` and ``--frequency-knots`` control interior-knot counts in
the example for explicit basis sensitivity checks. Defaults use five time and
ten frequency interior knots, two chains, 800 warmup and 1,000 retained draws
per chain, non-centered coordinates and target acceptance 0.97. The comparison
plot includes 90% posterior intervals for the first selected mode. These are inference settings,
not a guarantee of recovery from this short realization.

``--knot-placement quantile`` uses ``wishart_grid_knots`` from
``preprocessing.knot_locator``, which reuses ``variation_profiles`` and
``quantile_knots`` from the existing knot locator. It first smooths the observed
matrix sums and counts with positive Gaussian weights, then extracts diagonal
logs and signed real/imaginary Cholesky fields. Native rank-one cells are never
inverted or modified. Normalized variation profiles are combined across fields
to choose one shared basis; the allocator retains its 10% uniform floor and
minimum spacing set by the coarsest selected likelihood grid. Pilot smoothing
uses a time width of max(1,T/32) native cells and one frequency cell. Truth is
used only after basis selection, for diagnostics.

Knots and reference centring are selected once before the pooling comparison.
The final likelihood uses the original sums/counts, not the smoothed pilot.
The posterior conditions on the selected knots. Uniform versus quantile
placement changes the basis and its integrated-derivative penalty while
retaining the same prior family and hyperprior settings.

``--time-bin``, ``--frequency-bin`` and ``--modes`` select the fixed pooling
comparison. ``--spectrum-draws`` and ``--spectrum-chunk-size`` limit the stored
preview and reconstruction intermediates for larger grids.

Truth is the simulator's *frozen-time local VAR spectral matrix*, evaluated at
segment-centre rescaled time and interior Fourier frequencies. It is not the
exact covariance of a finite nonstationary segment Fourier transform. Saved
bin-averaged local truth and bin-centre pointwise local truth are labelled
separately; neither enters the likelihood.

The benchmark reports largest-row latent dimension, explicit compiled
log-density/gradient compilation and synchronized median call cost, per-row
warmup and sampling phase durations, reconstruction time, mean NUTS gradient
steps per retained draw, ESS per sampling second, diagnostics, and major
array sizes. HMC phase durations include their own compilation/initialization
overhead, so they are not pure steady-state NUTS costs. Compilation reuse,
initialization and dispatch can dominate these small cases; elapsed ratios
are not universal pooling speedups.

``log_diagonal_rmse`` is the square root of the equal-cell/channel mean squared
log median/truth ratio. Real and imaginary cross-spectrum RMSEs use strict
upper-triangle entries, normalized cellwise by
``sqrt(truth_ii*truth_jj)``. ``coherence_rmse`` compares posterior median squared
coherence to local truth over the same pairs/cells. Additional existing
``spectrum_diagnostics`` metrics retain their documented definitions. One
realization's interval coverage is descriptive, not a calibration study.

Tests compare blocked likelihoods/gradients to independent matrix calculations,
pooling/compression equivalence for constant within-bin spectra, rank-deficient
and omitted cells, scalar normalization, complex signs/conjugation, fixed-grid
ANOVA centring and chunks, FFT scaling/endpoints, and one-/two-channel public
sampling/results. Very short sampler tests check finite execution and matrix
invariants; they do not test tight recovery. Use larger/repeated examples to
assess recovery, convergence, and prior/model sensitivity before research use.

Deferred support
----------------

This path does not add WDM quadratures, scattered multivariate inference,
adaptive pooling, matrix-reference whitening, conditional Gaussian/Gibbs
updates, analytic marginalization, mixed stationary/TV fields, a new VI stage,
or a new evidence calculation.

Implementation and recovery checks: 2026-09-30
----------------------------------------------

The full suite passed: **80 tests**, zero failures, in 52.75 seconds. Tests
cover stationary, scalar TV, complex GridTV, optional WDM, and parametric
paths. New regressions compare centered and non-centered joint densities
including the transformation Jacobian, check physical field equivalence even
at sigma_eta=1e-5, and recover an independent smooth complex covariance from
64 replicates per cell. Short execution tests alone do not establish recovery.

The original TVVAR feasibility pilot was a poor recovery demonstration. Its
quadratic bases had only five frequency columns and three time-deviation
columns, and its 120-warmup/120-draw chains had R-hat up to 2.10 (two channels)
and 2.22 (three channels), with divergences and repeated tree-depth saturation.
Similar pooled curves were compatible with shared underfitting, not proof
that pooling preserved the signal.

A deterministic companion separates representation and transform effects:

.. code-block:: bash

   JAX_ENABLE_X64=1 .venv/bin/python docs/examples/diagnose_multivariate_gridtv.py

It projects independently recovered Cholesky truth fields into each basis,
without using those oracle coefficients in inference. Log-diagonal RMSE is
0.155 for the original basis, 0.091 with only frequency knots increased,
0.133 with only time knots increased, and 0.011 with both increased to the
new defaults (Kf=13, Kt_eta=7, K=104). Both frequency and time capacity matter.

The companion also propagates the TVVAR state and cross covariances and
contracts them with the exact rectangular DFT. Expected finite-window Fourier
covariance has log-diagonal RMSE 0.021 relative to frozen local truth and
retains its peak. The transform approximation does not explain the large
original discrepancy. On the seed-913 realization, the fixed local truth
has only 0.080 more log likelihood than its stationary Cholesky-field mean;
the expected gain is 4.534. This fixed-parameter comparison is not marginal
evidence or a general test for stationarity.

The corrected two-channel inference retains the same 1,024 time samples,
16-by-31 native grid, seed and learned priors. It uses the documented new
basis and sampler defaults. All three pooling modes have zero divergences
and zero tree-depth hits; maximum reported R-hat is 1.02 and minimum bulk ESS
is 371 across rows/modes. Native log-diagonal RMSE improves from 0.193 to
0.137; frequency-only and time-frequency RMSEs are both 0.140. Native coherence
RMSE is 0.080. Coefficient blocks are [104,312], with latent dimensions
[106,318]. The spectra remain Hermitian and positive definite.

**The corrected short-record posterior still underestimates the peak.** At
3.25 Hz and the plotted middle time, native S11 has median 0.072 and a 90%
interval [0.057,0.091], while local truth is 0.121. This is remaining inference
bias toward less time variation, not a feature recovered by fixing the
sampler. Longer chains with the original basis also leave its RMSE near
0.188. Four times as many time samples (64 segments) reduce native RMSE to
0.104, but still underestimate the peak. None of these single-record checks
establish calibration or eliminate regularization sensitivity.

For a separate likelihood/model recovery control:

.. code-block:: bash

   JAX_ENABLE_X64=1 .venv/bin/python docs/examples/diagnose_multivariate_gridtv.py \
     --replicates 16

This draws independent proper-complex Gaussian vectors under the same local
spectral covariance; it is explicitly **not** a time-domain TVVAR realization.
With 16 observations per native cell, log-diagonal RMSE is 0.060 and coherence
RMSE is 0.035; both rows have zero divergences, R-hat 1.01 and minimum bulk ESS
370. The imaginary cross spectrum remains less well recovered than the large
real component. This control demonstrates recovery with more informative
compatible observations, not validation of finite-window independence or
full scientific calibration.

The original pilots, corrected short-record run, and longer-record run are
saved separately under ``tests/test-output/tvvar-gridtv-{2,3}``,
``tests/test-output/tvvar-gridtv-corrected``, and
``tests/test-output/tvvar-gridtv-longer``. Each contains native results,
pointwise/bin-averaged truth, posterior plots and ``benchmark.json`` with
sampling diagnostics, timings and allocated-array sizes. Deterministic
checks and the complex recovery control are under
``tests/test-output/gridtv-diagnosis``. Output directories are ignored artifacts;
the example commands reproduce them.

Larger data with quantile knots and fixed pooling
-------------------------------------------------

The following controlled run uses 32,768 time samples, segment length 128,
eight time and sixteen frequency interior knots, and 4-by-2 pooling:

.. code-block:: bash

   JAX_ENABLE_X64=1 .venv/bin/python docs/examples/multivariate_gridtv.py \
     --segments 256 --segment-length 128 \
     --time-knots 8 --frequency-knots 16 --knot-placement quantile \
     --time-bin 4 --frequency-bin 2 --modes time-frequency \
     --spectrum-draws 4 --spectrum-chunk-size 4 \
     --outdir tests/test-output/tvvar-large-quantile

Replace ``quantile`` with ``uniform`` and change the output directory for
the matching comparison. Both runs use fs=16 Hz, seed 913, quadratic bases,
the same learned hyperpriors, two chains, 800 warmup and 1,000 retained draws
per chain, target acceptance 0.97, tree depth 10 and diagonal mass matrices.
The simulator retains two temporal modulation cycles in rescaled time; a
larger record therefore provides more observations per cycle. It also has
finer native frequency resolution than the original length-64 segments.

The native 256-by-63 grid contains 16,128 complex observations. Pooling uses
64-by-32 likelihood locations with all counts retained, including the smaller
frequency edge bin. Interior bins span 32 seconds and 0.25 Hz. Reconstruction
uses the original grid. Coefficient blocks are [209,627], with latent
dimensions [211,633]. Fewer likelihood locations do not imply an eightfold
runtime reduction: pooled factor width and NUTS trajectory lengths also matter.

.. list-table:: Single-realization recovery on the larger pooled dataset
   :header-rows: 1

   * - Knot placement
     - Log-diagonal RMSE
     - Coherence RMSE
     - Normalized real cross RMSE
     - Normalized imaginary cross RMSE
   * - Uniform
     - 0.0556
     - 0.0301
     - 0.0493
     - 0.00884
   * - Quantile
     - 0.0542
     - 0.0286
     - 0.0470
     - 0.00883

Both runs have zero divergences and zero tree-depth hits. Maximum reported
R-hat is 1.01 for uniform and 1.02 for quantile; minimum bulk ESS is 284 and
234, respectively. Fits took approximately 95 and 96 seconds including
reconstruction on the local CPU. The shared peak and coherence features are
much better recovered than in the corrected short-record fit (RMSE 0.137
and 0.080). Residual peak bias remains. Quantile placement gives a modest
additional improvement here; the main gain combines more data, finer FFT
resolution and adequate basis capacity. This single pair does not establish
a general ranking of knot schemes or separate those three effects.

For these particular bins, arithmetic bin-averaged local truth versus
bin-centre local truth differs by log-diagonal RMSE 0.00163 and coherence RMSE
0.000628. These are deterministic approximation checks, not a native-versus-
pooled posterior comparison. Wider bins still need a new check, especially
around narrow spectral features and changing cross-spectral phase.

Saved runs are ``tests/test-output/tvvar-large-{uniform,quantile}``; the inspected
combined plot and comparison metrics are in
``tests/test-output/tvvar-large-comparison``. The added fixed-seed regression
checks deterministic knot selection from rank-one observed data, spacing,
unchanged sums/counts and full-reference centring.
