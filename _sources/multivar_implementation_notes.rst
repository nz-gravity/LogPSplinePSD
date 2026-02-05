Multivariate implementation notes
=================================

This page collects implementation-level notes that are easy to forget when
reading the multivariate derivation in ``overleaf``.

FFT / PSD normalisation
-----------------------

The derivation in ``overleaf`` defines a DFT with an explicit
:math:`\Delta_t` factor and writes the Whittle likelihood with a :math:`1/T`
scaling.

For convenient cross-referencing, here is the exact LaTeX for the frequency
resolution and DFT definition from ``overleaf``:

.. math::

  \Delta_f = \frac{1}{n \Delta_t} = \frac{1}{T}\, .

.. math::

  \d(f_k) = \Delta_t\sum_{t=1}^{n} \Z_t\exp \left(-2\pi i \frac{k}{n} t \right)\, .

The implementation uses a one-sided Welch-style normalisation inside
:func:`log_psplines.datatypes.multivar.MultivarFFT.compute_wishart`.
Source: `src/log_psplines/datatypes/multivar.py#MultivarFFT.compute_wishart <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/datatypes/multivar.py#L165-L298>`_

Practical consequences:

- the DC bin is dropped (numerical stability on log axes and to avoid leakage
  dominating the first bin);
- the rFFT produces the positive-frequency grid; the normalisation ensures that
  :func:`log_psplines.spectrum_utils.wishart_matrix_to_psd` produces spectra in
  one-sided “per Hz” units;
- the Whittle observation-duration factor :math:`T` is tracked explicitly as
  ``fft_data.duration`` (seconds per time-domain block) and enters the
  multivariate likelihood via an explicit :math:`1/T` in the quadratic term.
  PSD conversions therefore divide by ``duration`` to preserve the same PSD
  convention across diagnostics/plotting.

Coarse graining: weights vs bin counts
--------------------------------------

When coarse graining multivariate data, two frequency-weight-like objects exist:

- `freq_bin_counts`: the raw member count :math:`Nh` per coarse bin, stored on
  :class:`log_psplines.datatypes.multivar.MultivarFFT`.
- `freq_weights`: the weights actually used to scale the log-determinant term in
  the NumPyro likelihood.

By default, `run_mcmc` passes `freq_weights = Nh`. This matches the coarse-bin
approximation where each bin has effective DOF :math:`N_b Nh`.

If users choose to normalise or temper `freq_weights` for sampler geometry, the
raw `freq_bin_counts` remain available so PSD conversion does not drift.

See:

- :class:`log_psplines.samplers.multivar.multivar_base.MultivarBaseSampler`
- :func:`log_psplines.coarse_grain.multivar.apply_coarse_grain_multivar_fft`
  (`source <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/coarse_grain/multivar.py#L16-L138>`_)
- :func:`log_psplines.spectrum_utils.compute_effective_Nb`

Blocked vs unified multivariate NUTS
------------------------------------

Two multivariate NUTS implementations exist:

- Blocked sampler: :class:`log_psplines.samplers.multivar.multivar_blocked_nuts.MultivarBlockedNUTSSampler`

  - fits each Cholesky row as an independent NUTS problem,
  - samples distinct spline-weight blocks for each off-diagonal
    :math:`\theta_{jl}(f)` within a row.

- Unified sampler: :class:`log_psplines.samplers.multivar.multivar_nuts.MultivarNUTSSampler`

  - fits all parameters jointly,
  - currently uses a *shared* spline field for all off-diagonal real parts and a
    shared field for all imaginary parts (tiled across :math:`\theta` indices).

If the intended model is “one spline per :math:`\theta_{jl}`”, the blocked
sampler aligns more closely with that intent.

Noise-floor option (blocked sampler)
------------------------------------

The blocked sampler can add an “innovation noise floor” in variance space,
replacing

.. math::

   \delta_j(f)^2 \mapsto \delta_j(f)^2 + \mathrm{floor}(f)

inside the likelihood. This can stabilise inference in near-null bands where
variances would otherwise collapse.

See:

- :class:`log_psplines.samplers.multivar.multivar_blocked_nuts.MultivarBlockedNUTSConfig`
- :func:`log_psplines.samplers.multivar.multivar_blocked_nuts.compute_noise_floor_sq`

Diagnostics: coarse-bin likelihood equivalence
----------------------------------------------

The helper :func:`log_psplines.diagnostics.coarse_grain_checks.coarse_bin_likelihood_equivalence_check`
(`source <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/diagnostics/coarse_grain_checks.py#L383-L460>`_)
tests the key coarse-graining invariant:

- sum of fine-grid log-likelihood contributions within a bin should match the
  coarse-bin log-likelihood (up to bin-constant offsets),
- there should be no “extra” factor of :math:`Nh` introduced in the quadratic
  term.

This is useful when changing coarse-grain logic or the frequency weighting
convention.
