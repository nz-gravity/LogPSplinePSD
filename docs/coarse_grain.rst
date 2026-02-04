Coarse-Graining
================

The multivariate PSD pipeline optionally groups nearby Fourier frequencies into
coarse bins so that each bin contributes a single aggregated Wishart statistic.
This documents the design of that discretization and shows how the implementation
realizes the theoretical approximations stated in the repository.

Frequency bins
--------------

The fine-frequency grid \(\{f_1,\dots,f_{N_l}\}\) can be coarse-grained by dividing
it into subsequent and disjoint subsets \(J_h\). Each \(J_h\) contains \(N_h\)
Fourier frequencies and \(\bar f_h\) denotes the *midpoint Fourier frequency* of
interval \(J_h\) (the middle member on the discrete Fourier grid).

The binning logic
is implemented by :func:`log_psplines.coarse_grain.preprocess.compute_binning_structure`,
which returns :class:`log_psplines.coarse_grain.preprocess.CoarseGrainSpec`.
Bins may be constructed with logarithmic or linear spacing via
:class:`log_psplines.coarse_grain.config.CoarseGrainConfig`. For an exact
"paper style" discretization with equal-length bins and midpoint frequencies,
set ``binning="linear"``, ``representative="middle"`` and choose an odd
``n_freqs_per_bin`` that evenly divides the retained frequency count.

The spec stores

- the masks that select the retained frequencies,
- the optional low/high split (when ``keep_low=True``),
- indices to group points into contiguous bins,
- the member count \(N_h\) per bin, and
- bin widths (primarily for univariate weight construction).

Aggregating FFT data
--------------------

:func:`log_psplines.coarse_grain.multivar.coarse_grain_multivar_fft` takes the
:class:`log_psplines.datatypes.multivar.MultivarFFT` and :class:`CoarseGrainSpec`
and builds the coarse representation used during sampling. The frequencies are
grouped by bin (optionally retaining a low-frequency region unchanged when
``keep_low=True``). Within each \(J_h\), the individual Wishart components
\(\Y(f)=\U(f)\U(f)^H\) are summed to form
\(\bar \Y_h = \sum_{f\in J_h}\Y(f)\), and the sum is re-diagonalized to obtain a
single \(\U_h\) per bin. The function :func:`log_psplines.spectrum_utils.sum_wishart_outer_products`
implements the matrix sum, and :func:`numpy.linalg.eigh` recovers the eigenvectors
and eigenvalues that encode \(\bar \Y_h\).

The returned :class:`log_psplines.datatypes.multivar.MultivarFFT` now has
`len(spec.f_coarse)` frequencies. `coarse_grain_multivar_fft` also returns a
`weights` array giving the member count \(N_h\) for each coarse bin (and `1` for
any frequencies retained without aggregation).

Likelihood scaling
------------------

The weights are used to scale the log-determinant term in the multivariate log-
likelihood so that each bin behaves like a Wishart observation with
\(N_b N_h\) degrees of freedom:

.. math::

    \log \mathcal{L} \propto - \sum_{h=1}^{N_c} N_b N_h \log |\S(\bar f_h)|
    - \sum_\nu \u^{(h)*}_\nu \S(\bar f_h)^{-1} \u^{(h)}_\nu.

When coarse graining is enabled, :class:`log_psplines.samplers.multivar.multivar_base.MultivarBaseSampler`
accepts the `weights` vector (usually via :class:`log_psplines.coarse_grain.config.CoarseGrainConfig`)
and stores it as `freq_weights`. The NumPyro model
:func:`log_psplines.samplers.multivar.multivar_nuts.multivariate_psplines_model`
multiplies `log_delta_sq` by `freq_weights`, ensuring the total log-det term
matches the aggregated \(N_b N_h\) DOF. The trace term already uses the summed
power in the aggregated \(\bar \Y_h\), so no further adjustments are needed.
