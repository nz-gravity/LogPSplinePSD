Multivariate Factorised Likelihood: Math to Code
================================================

This page documents the *implemented* multivariate likelihood and
parameterisation used by
:class:`log_psplines.samplers.multivar.multivar_blocked_nuts.MultivarBlockedNUTSSampler`.


Overview
--------

The multivariate pipeline is:

1. Build frequency-domain sufficient statistics from a multichannel time series
   using block-averaged FFTs.
2. Optionally coarse-grain the retained frequency axis by summing Wishart
   statistics within equal-sized bins.
3. Fit a Cholesky-parameterised spectral density matrix with P-splines,
   sampling each Cholesky row as an independent NumPyro/NUTS problem.

The implementation is organised around the Whittle/Wishart sufficient statistic
at each retained Fourier frequency.

Code pointers
-------------

The links below point to the current repository layout:

- `Wishart FFT construction (MultivarFFT.compute_wishart) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/datatypes/multivar.py>`_
- `Blocked NumPyro likelihood (_blocked_channel_model) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/multivar/multivar_blocked_nuts.py>`_
- `Shared P-spline prior block (sample_pspline_block) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/pspline_block.py>`_
- `Coarse graining (apply_coarse_grain_multivar_fft) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/preprocessing/coarse_grain.py>`_
- `PSD reconstruction (reconstruct_psd_matrix) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/psplines/multivar_psplines.py>`_
- `Wishart and PSD helpers (U_to_Y, Y_to_S, Y_to_U) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/datatypes/multivar_utils.py>`_
- `ArviZ export for multivariate samples <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/arviz_utils/to_arviz.py>`_

Notation and sufficient statistics
----------------------------------

The multivariate Whittle approximation is implemented in terms of the
frequency-domain Wishart statistic

.. math::

   Y(f_k) = \sum_{b=1}^{N_b} d_b(f_k) d_b(f_k)^H,

with degrees of freedom :math:`N_b` equal to the number of non-overlapping
time-domain blocks.

The code stores a factorisation

.. math::

   Y(f_k) = U(f_k) U(f_k)^H,

where the columns of :math:`U(f_k)` are the eigenvector-weighted components
:math:`\sqrt{\lambda_\ell^{(k)}} v_\ell^{(k)}`. These are exposed as
``u_re`` and ``u_im`` on
:class:`log_psplines.datatypes.multivar.MultivarFFT`.

.. ADD LINK TO CODE


Data to Wishart statistics
--------------------------

The sufficient statistics are computed by
:func:`log_psplines.datatypes.multivar.MultivarFFT.compute_wishart`.
Given time-domain data ``x`` with shape ``(n, p)``, the code:

- splits ``x`` into ``Nb`` contiguous non-overlapping blocks,
- detrends each block by removing its mean,
- applies an optional taper/window,
- uses ``np.fft.rfft`` on each block,
- drops the DC bin,
- applies a one-sided PSD normalisation,
- rescales by ``sqrt(duration)`` so the likelihood keeps an explicit
  :math:`1/T` factor,
- forms ``Y(f_k)`` and stores a factor ``U(f_k)`` such that
  ``Y(f_k) = U(f_k) U(f_k)^H``.

A subtle but important implementation detail is the window scaling. The code
computes the normalized equivalent noise bandwidth ``enbw`` and later divides
the full log-likelihood by ``enbw``. This is a calibration correction for
non-rectangular windows; rectangular windows give ``enbw = 1``.

FFT and PSD normalisation
-------------------------

The derivation in ``overleaf`` defines a DFT with an explicit
:math:`\Delta_t` factor and writes the Whittle likelihood with a :math:`1/T`
scaling:

.. math::

   \Delta_f = \frac{1}{n\Delta_t} = \frac{1}{T}

.. math::

   d(f_k) = \Delta_t \sum_{t=1}^{n} Z_t \exp\left(-2\pi i \frac{k}{n} t\right).

The implementation uses a one-sided Welch-style normalisation inside
:func:`~log_psplines.datatypes.multivar.MultivarFFT.compute_wishart`, but keeps
the observation-duration factor explicit in the likelihood.

Practical consequences:

- the DC bin is dropped for numerical stability,
- the retained grid is the positive-frequency ``rfft`` grid,
- ``fft_data.duration`` stores the per-block duration in seconds,
- PSD conversions divide by ``duration`` so exported spectra remain in the same
  one-sided per-Hz convention across plotting and diagnostics,
- tapered windows widen the posterior through the ``enbw`` correction.

Cholesky parameterisation
-------------------------

The blocked sampler uses the inverse parameterisation

.. math::

   S(f_k)^{-1} = T(f_k)^H D(f_k)^{-1} T(f_k),

where:

- :math:`D(f_k)` is diagonal with entries :math:`\delta_j(f_k)^2`,
- :math:`T(f_k)` is unit lower-triangular with strictly-lower entries
  :math:`-\theta_{jl}(f_k)`.

In code, the diagonal field is represented as

.. math::

   \log \delta_j(f_k)^2 = \sum_m B_m(f_k) w^{(\delta)}_{j,m},

and each off-diagonal coefficient has its own real and imaginary spline field

.. math::

   \Re\,\theta_{jl}(f_k) = \sum_m B_m(f_k) w^{(\Re)}_{jl,m},
   \qquad
   \Im\,\theta_{jl}(f_k) = \sum_m B_m(f_k) w^{(\Im)}_{jl,m}.

The implemented blocked sampler samples distinct weight vectors for every
:math:`\theta_{jl}` within row :math:`j`.

.. PROVIDE CODE LINKS, EXPLAIN EINSUM NOTATION THAT WE USE FOR THIS

Per-row factorisation of the likelihood
---------------------------------------

With the parameterisation above, the complex-Wishart log-likelihood separates
into one term per Cholesky row. Let :math:`u_j(f_k)` denote the :math:`j`-th
row of :math:`U(f_k)`, i.e. the replicate vector for channel :math:`j` at
frequency :math:`f_k`. Define the row residual

.. math::

   r_j(f_k) = u_j(f_k) - \sum_{l<j} \theta_{jl}(f_k) u_l(f_k).

Up to constants, the blocked model implements

.. math::

   \log \mathcal{L}_j
   \propto
   -N_b N_h \sum_k \log\big(\delta_j(f_k)^2\big)
   - \sum_k \frac{\|r_j(f_k)\|_2^2}{T\,\delta_j(f_k)^2},

and then divides the full block contribution by ``enbw``.

Key mappings:

- :math:`N_b` is ``fft_data.Nb``, the number of averaged blocks,
- :math:`T` is ``fft_data.duration``, the duration of each block,
- :math:`N_h` is ``fft_data.Nh``, the constant coarse-bin size,
- ``u_re`` and ``u_im`` store the rows of :math:`U(f_k)`.

This is implemented in ``_blocked_channel_model`` inside
:mod:`log_psplines.samplers.multivar.multivar_blocked_nuts`.

Coarse graining
---------------

Coarse graining is performed by
:func:`log_psplines.preprocessing.coarse_grain.apply_coarse_grain_multivar_fft`.
Within each coarse bin :math:`J_h`, the code sums Wishart matrices

.. math::

   \bar Y_h = \sum_{f \in J_h} Y(f)

and recomputes :math:`\bar U_h` so that
:math:`\bar Y_h = \bar U_h \bar U_h^H`.

The coarse-bin multiplicity is stored as the scalar ``fft_data.Nh``. For the
equal-sized bins currently implemented, each coarse bin behaves like a Wishart
statistic with effective degrees of freedom :math:`N_b N_h`.

A point that is easy to get wrong: the quadratic term uses the already summed
sufficient statistics, so there is no extra ``Nh`` factor inserted there. Only
the log-determinant scaling changes.

Blocked versus unified multivariate NUTS
----------------------------------------

The repository currently contains one multivariate NUTS implementation:

- :class:`log_psplines.samplers.multivar.multivar_blocked_nuts.MultivarBlockedNUTSSampler`

It fits each Cholesky row as an independent NUTS problem. There is no separate
all-parameters-joint multivariate NUTS sampler in ``src/log_psplines`` at
present.

PSD reconstruction and downstream outputs
-----------------------------------------

Posterior samples of ``log_delta_sq`` and ``theta_re``/``theta_im`` are
converted into spectral density matrices by
:meth:`log_psplines.psplines.multivar_psplines.MultivariateLogPSplines.reconstruct_psd_matrix`.
For each frequency the code builds :math:`T(f_k)` and
:math:`D(f_k) = \mathrm{diag}(\delta_j(f_k)^2)`, then returns

.. math::

   S(f_k) = T(f_k)^{-1} D(f_k) T(f_k)^{-H}.

ArviZ exports are aligned with the Wishart-based likelihood: the multivariate
observed data are stored as periodogram-like spectral matrices derived from the
Wishart factors, rather than as retained mean FFT components.

Prior on :math:`\phi`: implementation versus theory
----------------------------------------------------

The draft math writes the hierarchical P-spline prior as

.. math::

   w_j \mid \phi_j \sim \mathcal{N}(0, (\phi_j P_j)^{-1})

with hyperpriors

.. math::

   \phi_j \mid \delta_j \sim \mathrm{Gamma}(\alpha_\phi, \delta_j \beta_\phi),
   \qquad
   \delta_j \sim \mathrm{Gamma}(\alpha_\delta, \beta_\delta).

In the implementation, ``delta`` is sampled from that Gamma prior using the
NumPyro rate parameterisation, but :math:`\phi \mid \delta` is *not* sampled
from the exact Gamma distribution. Instead,
:func:`log_psplines.samplers.pspline_block.sample_pspline_block` samples
:math:`\log \phi` from a Normal distribution chosen to moment-match the Gamma.

For

.. math::

   \phi \mid \delta \sim \mathrm{Gamma}(\alpha_\phi, \mathrm{rate}=\beta_\phi \delta),

this gives the log-normal approximation

.. math::

   \log \phi \mid \delta \sim \mathcal{N}(\mu, \sigma^2)

with

.. math::

   \sigma^2 = \log\left(1 + \frac{1}{\alpha_\phi}\right),
   \qquad
   \mu = \log\left(\frac{\alpha_\phi}{\beta_\phi \delta}\right) - \frac{\sigma^2}{2}.

Why use the approximation?
^^^^^^^^^^^^^^^^^^^^^^^^^^

The practical reason is posterior geometry. Large or tiny values of
:math:`\phi` interact strongly with the spline weights through the quadratic
penalty, and the exact hierarchical model can be awkward for VI-based
initialisation and NUTS warmup. The moment-matched log-normal prior is a
pragmatic regularisation of that geometry.

Why not just use the original prior?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You *can*. The main trade-off is between theoretical fidelity and numerical
robustness.

Reasons to prefer the original Gamma prior:

- it matches the written model exactly,
- it preserves the intended tail behaviour,
- it avoids changing the implied shrinkage when posterior inference on
  :math:`\phi` matters substantively.

Reasons the implementation currently does not use it directly:

- direct sampling of :math:`\phi` tended to produce brittle warmup and more
  difficult curvature for the existing centred spline-weight parameterisation,
- the current sampler path relies heavily on stable VI initialisation,
- the approximation was an expedient way to make the blocked sampler usable on
  realistic multivariate runs.

The more precise statement is that the current code does *not* merely
reparameterise the exact Gamma prior; it changes the prior to a log-normal
approximation with matching first two moments.

If exact prior fidelity becomes important, a better first step would be to keep
sampling on the log scale but use the *exact transformed Gamma density* for
``log_phi`` rather than the moment-matched Normal. That would preserve the
original prior while still avoiding a hard positivity constraint in the sampler.
Whether that is stable enough in practice is an empirical question.

Symbol to code mapping
----------------------

- :math:`p` number of channels -> ``fft_data.p`` / ``self.p``
- :math:`f_k` frequency grid -> ``fft_data.freq`` / ``self.freq``
- :math:`N_b` block count -> ``fft_data.Nb`` / ``self.Nb``
- :math:`N_h` coarse-bin size -> ``fft_data.Nh`` / ``self.Nh``
- :math:`T` block duration -> ``fft_data.duration`` / ``self.duration``
- :math:`U(f_k)` Wishart factor -> ``fft_data.u_re + 1j * fft_data.u_im``
- :math:`\log \delta_j(f_k)^2` -> deterministic nodes ``log_delta_sq_{j}``
- :math:`\theta_{jl}(f_k)` -> deterministic nodes ``theta_re_{j}``,
  ``theta_im_{j}``
