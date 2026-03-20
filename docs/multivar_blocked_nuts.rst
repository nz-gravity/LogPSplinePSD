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

- `Wishart FFT construction (MultivarFFT.compute_wishart) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/datatypes/multivar.py#L159-L299>`_
- `Blocked NumPyro likelihood (_blocked_channel_model) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/multivar/multivar_blocked_nuts.py#L55-L239>`_
- `Shared P-spline prior block (sample_pspline_block) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/pspline_block.py#L59-L137>`_
- `Coarse graining (apply_coarse_grain_multivar_fft) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/preprocessing/coarse_grain.py#L288-L343>`_
- `PSD reconstruction (reconstruct_psd_matrix) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/psplines/multivar_psplines.py#L531-L580>`_
- `Wishart and PSD helpers (U_to_Y, Y_to_S, Y_to_U) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/datatypes/multivar_utils.py#L141-L224>`_
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

Relevant code:

- diagonal P-spline block construction:
  `delta block in _blocked_channel_model <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/multivar/multivar_blocked_nuts.py#L122-L139>`_
- off-diagonal real and imaginary blocks:
  `theta blocks in _blocked_channel_model <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/multivar/multivar_blocked_nuts.py#L145-L188>`_

The Einstein-summation calls in the implementation are just compact matrix
algebra:

- ``jnp.einsum("nk,k->n", basis_delta, weights)`` evaluates a spline basis on
  the frequency grid,
- ``jnp.einsum("fl,flr->fr", theta_re, u_re_prev)`` and its companions compute
  the lower-triangular row regressions used to form the residuals.

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
:mod:`log_psplines.samplers.multivar.multivar_blocked_nuts`; the custom
log-likelihood contribution is added via
``numpyro.factor("likelihood_channel_*", log_likelihood)``.

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
---------------------------------------------------

The draft math writes the hierarchical P-spline prior as

.. math::

   w_j \mid \phi_j \sim \mathcal{N}(0, (\phi_j P_j)^{-1})

with hyperpriors

.. math::

   \phi_j \mid \delta_j \sim \mathrm{Gamma}(\alpha_\phi, \delta_j \beta_\phi),
   \qquad
   \delta_j \sim \mathrm{Gamma}(\alpha_\delta, \beta_\delta).

In the implementation, ``delta`` is sampled directly from that Gamma prior
using the NumPyro rate parameterisation. For :math:`\phi \mid \delta`, the
sampled variable is
:math:`\eta = \log \phi`, but the prior is still the *exact* Gamma prior on
:math:`\phi`, expressed on the log scale through a change of variables.

For

.. math::

   \phi \mid \delta \sim \mathrm{Gamma}(\alpha_\phi, \mathrm{rate}=\beta_\phi \delta),

write :math:`\eta = \log \phi`, so :math:`\phi = e^\eta`. Then the induced
log-density for :math:`\eta` is

.. math::

   \log p(\eta \mid \delta)
   =
   \log p_\Gamma(e^\eta \mid \delta) + \eta.

That final ``+ eta`` term is the Jacobian from the transformation
:math:`\phi = e^\eta`.

In code, :func:`log_psplines.samplers.pspline_block.sample_pspline_block`
implements this by:

- sampling ``log_phi`` from a simple reference distribution,
- setting ``phi = exp(log_phi)``,
- evaluating the exact Gamma log-density at ``phi``,
- adding the Jacobian term ``+ log_phi``,
- correcting the reference density with ``numpyro.factor``.

This means the sampler keeps the geometry benefits of working on the log scale
without changing the prior itself.

Why we do not need an explicit likelihood object
------------------------------------------------

NumPyro models do not require a separate ``Likelihood(...)`` object. Instead,
the model defines the joint log density by accumulating contributions from:

- ``numpyro.sample(name, distribution)`` for variables with a named
  probability distribution,
- ``numpyro.factor(name, log_term)`` for custom log-density terms that are
  easier to write directly than as a built-in distribution.

In this codebase:

- the exact transformed-Gamma prior on ``log_phi`` is implemented by
  ``numpyro.sample`` plus a correcting ``numpyro.factor`` in
  ``sample_pspline_block``,
- the P-spline Gaussian penalty on the weights is also imposed through
  ``numpyro.factor`` in ``sample_pspline_block``,
- the blocked Whittle/Wishart likelihood is imposed through
  ``numpyro.factor(f"likelihood_channel_{channel_label}", log_likelihood)`` in
  ``_blocked_channel_model``.

So the likelihood is still there mathematically. It is just expressed directly
as a log-density contribution, rather than wrapped in a separate explicit
likelihood class. NumPyro then combines all ``sample`` and ``factor`` terms
into the full posterior that NUTS sees.

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
