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

Coarse graining: scalar ``Nh``
------------------------------

When coarse graining multivariate data, the coarse-bin membership is carried as
the scalar ``fft_data.Nh``. This scalar is used directly to scale the
log-determinant term in the NumPyro likelihood, so each bin has effective DOF
:math:`N_b N_h`. There is no separate per-bin weighting vector.

See:

- :class:`log_psplines.samplers.multivar.multivar_base.MultivarBaseSampler`
- :func:`log_psplines.preprocessing.coarse_grain.multivar.apply_coarse_grain_multivar_fft`
  (`source <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/coarse_grain/multivar.py#L16-L138>`__)

Blocked vs unified multivariate NUTS
------------------------------------

The repository currently contains a single multivariate NUTS implementation:

- Blocked sampler:
  :class:`log_psplines.samplers.multivar.multivar_blocked_nuts.MultivarBlockedNUTSSampler`

  - fits each Cholesky row as an independent NUTS problem,
  - samples distinct spline-weight blocks for each off-diagonal
    :math:`\theta_{jl}(f)` within a row.

If you are looking for a “unified” (all-parameters-joint) multivariate NUTS
sampler, it is not currently implemented in ``src/log_psplines/``. The blocked
sampler is the reference implementation for the multivariate model described in
the technical notes.

Diagnostics: coarse-bin likelihood equivalence
----------------------------------------------

The helper :func:`log_psplines.diagnostics.coarse_grain_checks.coarse_bin_likelihood_equivalence_check`
(`source <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/diagnostics/coarse_grain_checks.py#L383-L460>`__)
tests the key coarse-graining invariant:

- sum of fine-grid log-likelihood contributions within a bin should match the
  coarse-bin log-likelihood (up to bin-constant offsets),
- there should be no “extra” factor of :math:`N_h` introduced in the quadratic
  term.

This is useful when changing coarse-grain logic or the frequency weighting
convention.
