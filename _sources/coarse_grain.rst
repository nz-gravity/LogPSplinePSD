Coarse-Graining
================

The multivariate PSD pipeline optionally groups nearby Fourier frequencies into
coarse bins so that each bin contributes a single aggregated Wishart statistic.
This documents the design of that discretization and shows how the implementation
realizes the theoretical approximations stated in the repository.

Frequency bins
--------------

The fine-frequency grid \(\{f_1,\dots,f_{N_l}\}\) is split into low-frequency
points (kept untouched) and one or more high-frequency logarithmic bins. Each bin
contains a contiguous subset \(J_h\) with midpoint \(\bar f_h\). The binning logic
is implemented by :func:`log_psplines.coarse_grain.preprocess.compute_binning_structure`,
which returns :class:`log_psplines.coarse_grain.preprocess.CoarseGrainSpec`.

The spec stores

- the masks that select the retained frequencies,
- the low/high boundaries,
- indices to reorder high-frequency points into contiguous bins,
- the count of members per bin, and
- bin widths used for frequency-weight scaling.

Aggregating FFT data
--------------------

:func:`log_psplines.coarse_grain.multivar.coarse_grain_multivar_fft` takes the
:class:`log_psplines.datatypes.multivar.MultivarFFT` and :class:`CoarseGrainSpec`
and builds the coarse representation used during sampling. Low frequencies are
copied unchanged. High frequencies are grouped by bin, the individual Wishart
components \(\Y(f)=\U(f)\U(f)^H\) are summed within each \(J_h\) to form
\(\bar \Y_h = \sum_{f\in J_h}\Y(f)\), and the sum is re-diagonalized to obtain a
single \(\U_h\) per bin. The function :func:`log_psplines.spectrum_utils.sum_wishart_outer_products`
implements the matrix sum, and :func:`numpy.linalg.eigh` recovers the eigenvectors
and eigenvalues that encode \(\bar \Y_h\).

The returned :class:`log_psplines.datatypes.multivar.MultivarFFT` now has
`len(spec.f_coarse)` frequencies; the first entries correspond to the unmodified
low frequencies while the remaining entries represent the aggregated high bins.
`coarse_grain_multivar_fft` also returns a `weights` array where low frequencies
contribute `1` and each coarse bin contributes its member count \(N_h\).

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
