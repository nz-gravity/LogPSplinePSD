Multivariate blocked NUTS: math ↔ code
====================================

This page documents the *implemented* multivariate likelihood and
parameterisation used by the blocked sampler
:class:`log_psplines.samplers.multivar.multivar_blocked_nuts.MultivarBlockedNUTSSampler`.

It is intended as a durable bridge between the draft derivation in our paper
 and the code under ``src/log_psplines/``.

Overview
--------

The multivariate pipeline is:

1. Build frequency-domain sufficient statistics from a multichannel time series
   using block-averaged FFTs.
2. (Optional) coarse-grain the frequency axis by summing Wishart statistics
   across bins.
3. Fit a Cholesky-parameterised spectral density matrix with P-splines,
   sampling each Cholesky row as an *independent* NumPyro/NUTS problem.

The implementation is designed around the Whittle/Wishart sufficient statistic

.. math::

   Y(f_k) = \sum_{b=1}^{\nu} d^{(b)}(f_k)\, d^{(b)}(f_k)^H

with degrees of freedom :math:`\nu` equal to the number of non-overlapping
blocks (called ``n_blocks`` in code).

Data → Wishart statistics
-------------------------

The sufficient statistics are computed by
:func:`log_psplines.datatypes.multivar.MultivarFFT.compute_wishart`.

Given time-domain data ``x`` with shape ``(n_time, p)``, the code:

- splits ``x`` into ``n_blocks`` contiguous non-overlapping blocks,
- detrends each block (constant detrend),
- applies a taper/window (default: Hann),
- uses ``np.fft.rfft`` to work on the positive-frequency grid,
- applies a one-sided PSD normalisation factor (Nyquist bin handled specially),
- forms the per-frequency matrix

  .. math::

     Y(f_k) = \sum_{b=1}^{\nu} D_b(f_k)\, D_b(f_k)^H,

  where :math:`D_b(f_k)` is the tapered/normalised FFT vector for block :math:`b`.

The code then eigendecomposes :math:`Y(f_k)` and stores

.. math::

   Y(f_k) = U(f_k) U(f_k)^H,

where the columns of :math:`U(f_k)` are :math:`u_\ell^{(k)} = \sqrt{\lambda_\ell^{(k)}}\, v_\ell^{(k)}`.

Those eigenvector-weighted components :math:`U(f_k)` are the arrays
``u_re``/``u_im`` exposed on :class:`log_psplines.datatypes.multivar.MultivarFFT`.

Cholesky parameterisation
-------------------------

The blocked sampler follows the Rosen/Orihara-style inverse parameterisation:

.. math::

   S(f_k)^{-1} = T(f_k)^H\, D(f_k)^{-1}\, T(f_k),

where

- :math:`D(f_k)` is diagonal with entries :math:`\delta_j(f_k)^2`,
- :math:`T(f_k)` is unit lower-triangular with strictly-lower entries
  :math:`-\theta_{jl}(f_k)`.

In code, the diagonal field is represented as

.. math::

   \log \delta_j(f_k)^2 = \sum_m B_m(f_k)\, w^{(\delta)}_{j,m}

and each off-diagonal coefficient has its own real/imag spline field

.. math::

   \Re\,\theta_{jl}(f_k) = \sum_m B_m(f_k)\, w^{(\Re)}_{jl,m},
   \qquad
   \Im\,\theta_{jl}(f_k) = \sum_m B_m(f_k)\, w^{(\Im)}_{jl,m}.

The blocked sampler actually samples distinct weight vectors per
:math:`\theta_{jl}` (within a fixed row :math:`j`).

Per-row factorisation of the likelihood
---------------------------------------

With the parameterisation above, the complex-Wishart/Whittle log-likelihood can
be written as a sum of row-wise terms (one term per Cholesky row).

Let :math:`u_j(f_k)` denote the :math:`j`-th row of :math:`U(f_k)`, i.e. the
vector of replicate components for channel :math:`j` at frequency :math:`f_k`.

Define the row residual

.. math::

   r_j(f_k) = u_j(f_k) - \sum_{l<j} \theta_{jl}(f_k)\, u_l(f_k).

The blocked model implements (up to constants)

.. math::

   \log \mathcal{L}_j
   \;\propto\;
   -\nu\,\sum_k w_k\,\log\big(\delta_j(f_k)^2\big)
   \; - \sum_k \frac{\|r_j(f_k)\|_2^2}{\delta_j(f_k)^2}.

Key points:

- :math:`\nu` is the number of averaged blocks (``fft_data.nu``).
- :math:`w_k` are optional frequency weights (``freq_weights``). They are used
  to scale the log-determinant term when coarse graining is enabled.
- The quadratic term uses the *summed sufficient statistics* directly, so it
  does not multiply by :math:`w_k` again.

This is implemented in the internal NumPyro model
``_blocked_channel_model`` inside
:mod:`log_psplines.samplers.multivar.multivar_blocked_nuts`.

Coarse-graining compatibility
-----------------------------

Coarse graining for multivariate FFT/Wishart statistics is performed by
:func:`log_psplines.coarse_grain.multivar.coarse_grain_multivar_fft`.

Within each coarse bin :math:`J_h`, it sums

.. math::

   \bar Y_h = \sum_{f\in J_h} Y(f)

and recomputes :math:`\bar U_h` so that :math:`\bar Y_h = \bar U_h\bar U_h^H`.

The returned `weights` vector equals the bin member counts :math:`N_h` and
should be passed as ``freq_weights``. With this choice, each bin behaves like a
Wishart statistic with effective degrees of freedom :math:`\nu N_h`.

Symbol ↔ code mapping
---------------------

The table below lists the most important objects and where they appear.

- :math:`p` (number of channels) → ``fft_data.n_dim`` / ``self.n_channels``
- :math:`f_k` (frequency grid) → ``fft_data.freq`` / ``self.freq``
- :math:`\nu` (block count / Wishart DOF) → ``fft_data.nu`` / ``self.nu``
- :math:`U(f_k)` (eigen replicates) → ``fft_data.u_re`` + i ``fft_data.u_im``
- :math:`\log \delta_j(f_k)^2` → deterministic nodes ``log_delta_sq_{j}``
- :math:`\theta_{jl}(f_k)` → deterministic nodes ``theta_re_{j}``, ``theta_im_{j}``
- coarse-bin member counts :math:`N_h` → ``fft_data.freq_bin_counts`` (stored)
- likelihood weights :math:`w_k` → ``config.freq_weights`` / ``self.freq_weights``

PSD reconstruction
------------------

Posterior samples of ``log_delta_sq`` and ``theta_re``/``theta_im`` are
converted into a spectral density matrix by
:meth:`log_psplines.psplines.multivar_psplines.MultivariateLogPSplines.reconstruct_psd_matrix`.

For each frequency, it builds :math:`T(f_k)` with strictly-lower entries
:math:`-\theta_{jl}(f_k)`, constructs :math:`D(f_k)=\mathrm{diag}(\delta_j(f_k)^2)`,
and returns

.. math::

   S(f_k) = T(f_k)^{-1}\, D(f_k)\, T(f_k)^{-H}.

Notes on scaling conventions
----------------------------

The derivation in ``docs/maths.tex`` writes the Whittle likelihood with an
explicit observation-time factor :math:`T`.

The implementation normalises the FFTs to one-sided PSD units during
:func:`~log_psplines.datatypes.multivar.MultivarFFT.compute_wishart`. In that
normalisation, the time/frequency-resolution factors are absorbed into the
construction of :math:`Y(f_k)` (and therefore :math:`U(f_k)`). As a result,
there is no explicit :math:`T` factor in the NumPyro likelihood.

One way to see that this is consistent is to view it as a deterministic
reparameterisation of the Fourier coefficients.

In one common Whittle convention,

.. math::

   d(f_k) \;\dot\sim\; \mathcal{CN}\!\left(0,\; T\,S(f_k)\right),

which yields a quadratic term of the form

.. math::

   \exp\!\left(-\frac{1}{T}\, d(f_k)^H S(f_k)^{-1} d(f_k)\right).

If instead we define the rescaled coefficient

.. math::

   	ilde d(f_k) = \frac{d(f_k)}{\sqrt{T}},

then

.. math::

   	ilde d(f_k) \;\dot\sim\; \mathcal{CN}\!\left(0,\; S(f_k)\right),

and the quadratic term becomes

.. math::

   \exp\!\left(-\tilde d(f_k)^H S(f_k)^{-1} \tilde d(f_k)\right)

with no explicit :math:`T`.

The code effectively makes this kind of deterministic rescaling (along with the
standard one-sided/Welch normalisation factors) when constructing
:math:`Y(f_k)` from the block FFTs. This is fine as long as you interpret
:math:`S(f)` in the implementation as being expressed in the corresponding
one-sided PSD convention.
