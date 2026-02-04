Coarse-Graining
================

The multivariate PSD pipeline optionally groups nearby Fourier frequencies into
coarse bins so that each bin contributes a single aggregated Wishart statistic.
This documents the design of that discretization and shows how the implementation
realizes the theoretical approximations stated in the repository.

Frequency bins
--------------

The retained fine-frequency grid \(\{f_1,\dots,f_{N_l}\}\subset[f_{\min},f_{\max}]\)
is coarse-grained by dividing it into consecutive, disjoint subsets \(J_h\). Each
\(J_h\) contains an **odd** number \(N_h\) of Fourier frequencies, and \(\bar f_h\)
denotes the *midpoint Fourier frequency* of \(J_h\) (the middle member on the
discrete Fourier grid, not an average).

The binning logic
is implemented by :func:`log_psplines.coarse_grain.preprocess.compute_binning_structure`,
which returns :class:`log_psplines.coarse_grain.preprocess.CoarseGrainSpec`.
Only **linear, full-band** binning is supported. Exactly one construction mode
must be chosen:

- ``n_freqs_per_bin``: fixed membership with equal-size (odd) bins that divide
  the retained frequency count exactly.
- ``n_bins``: fixed bin count with sizes as equal as possible, all odd.

The spec stores

- the masks that select the retained frequencies,
- indices to group points into contiguous bins,
- the member count \(N_h\) per bin, and
- bin widths (diagnostics only).

Aggregating FFT data
--------------------

:func:`log_psplines.coarse_grain.multivar.coarse_grain_multivar_fft` takes the
:class:`log_psplines.datatypes.multivar.MultivarFFT` and :class:`CoarseGrainSpec`
and builds the coarse representation used during sampling. The frequencies are
grouped by bin across the **entire** retained band. Within each \(J_h\), the
individual Wishart components \(\Y(f)=\U(f)\U(f)^H\) are **summed** to form
\(\bar \Y_h = \sum_{f\in J_h}\Y(f)\), and the sum is re-diagonalized to obtain a
single \(\U_h\) per bin. The function :func:`log_psplines.spectrum_utils.sum_wishart_outer_products`
implements the matrix sum, and :func:`numpy.linalg.eigh` recovers the eigenvectors
and eigenvalues that encode \(\bar \Y_h\).

The returned :class:`log_psplines.datatypes.multivar.MultivarFFT` now has
`len(spec.f_coarse)` frequencies. `coarse_grain_multivar_fft` also returns a
`weights` array giving the member count \(N_h\) for each coarse bin.

Likelihood scaling
------------------

The weights scale **only** the log-determinant term so that each bin behaves like
a Wishart observation with \(N_b N_h\) degrees of freedom:

.. math::

    \log \mathcal{L} \propto - \sum_{h=1}^{N_c} N_b N_h \log |\S(\bar f_h)|
    - \sum_\nu \u^{(h)*}_\nu \S(\bar f_h)^{-1} \u^{(h)}_\nu.

When coarse graining is enabled, :class:`log_psplines.samplers.multivar.multivar_base.MultivarBaseSampler`
accepts the `weights` vector (usually via :class:`log_psplines.coarse_grain.config.CoarseGrainConfig`)
and stores it as `freq_weights`. The NumPyro model
:func:`log_psplines.samplers.multivar.multivar_nuts.multivariate_psplines_model`
multiplies `log_delta_sq` by `freq_weights`, ensuring the total log-det term
matches the aggregated \(N_b N_h\) DOF. The trace term uses the **summed**
statistic \(\bar \Y_h\) directly, so no additional \(N_h\) factors appear.
Coarse graining therefore increases information linearly in \(N_h\).

No log-binning, hybrid schemes, or “preserve low frequencies” modes are
implemented.
