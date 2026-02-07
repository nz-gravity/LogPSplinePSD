Coarse-Graining
================

The multivariate PSD pipeline optionally groups nearby Fourier frequencies into
coarse bins so that each bin contributes a single aggregated Wishart statistic.
This documents the design of that discretization and shows how the implementation
realizes the theoretical approximations stated in the repository.

Frequency bins
--------------

The retained fine-frequency grid :math:`\{f_1,\dots,f_{N_\ell}\}\subset[f_{\min},f_{\max}]`
is coarse-grained by dividing it into consecutive, disjoint subsets :math:`J_h`. Each
:math:`J_h` contains an **odd** number :math:`N_h` of Fourier frequencies, and :math:`\bar f_h`
denotes the *midpoint Fourier frequency* of :math:`J_h` (the middle member on the
discrete Fourier grid, not an average).

The binning logic
is implemented by :func:`log_psplines.coarse_grain.preprocess.compute_binning_structure`,
which returns :class:`log_psplines.coarse_grain.preprocess.CoarseGrainSpec`.
Only **linear, full-band** binning is supported. Exactly one construction mode
must be chosen:

- ``Nh``: fixed membership with equal-size (odd) bins that divide
  the retained frequency count exactly.
- ``Nc``: fixed bin count with sizes as equal as possible, all odd.

The spec stores

- the masks that select the retained frequencies,
- indices to group points into contiguous bins,
- the member count :math:`N_h` per bin, and
- bin widths (diagnostics only).

Aggregating FFT data
--------------------

:func:`log_psplines.coarse_grain.multivar.apply_coarse_grain_multivar_fft` takes the
:class:`log_psplines.datatypes.multivar.MultivarFFT` and :class:`CoarseGrainSpec`
and builds the coarse representation used during sampling. The frequencies are
grouped by bin across the **entire** retained band. Within each :math:`J_h`, the
individual Wishart matrices :math:`\mathbf{Y}(f)=\mathbf{U}(f)\mathbf{U}(f)^H` are **summed** to form
:math:`\bar{\mathbf{Y}}_h = \sum_{f\in J_h}\mathbf{Y}(f)`, and the sum is re-diagonalized to obtain a
single :math:`\bar{\mathbf{U}}_h` per bin. The function :func:`log_psplines.spectrum_utils.sum_wishart_outer_products`
implements the matrix sum, and :func:`numpy.linalg.eigh` recovers the eigenvectors
and eigenvalues that encode :math:`\bar{\mathbf{Y}}_h`.

The returned :class:`log_psplines.datatypes.multivar.MultivarFFT` now has
`len(spec.f_coarse)` frequencies. `apply_coarse_grain_multivar_fft` also returns a
scalar ``Nh`` giving the member count :math:`N_h` for each coarse bin (constant for
equal bins), and stores it on the FFT as ``Nh``.

Likelihood scaling
------------------

The scalar :math:`N_h` scales **only** the log-determinant term so that each bin behaves like
a Wishart observation with :math:`N_b N_h` degrees of freedom:

.. math::

    \log \mathcal{L}
    \;\propto\;
    - \sum_{h=1}^{N_c} N_b N_h \log \left|\mathbf{S}(\bar f_h)\right|
    - \sum_{h=1}^{N_c} \frac{1}{T}\,\mathrm{tr}\!\left[\mathbf{S}(\bar f_h)^{-1}\,\bar{\mathbf{Y}}_h\right].

When coarse graining is enabled, the multivariate sampler reads the scalar
``Nh`` from ``fft_data.Nh``. The NumPyro model
:func:`log_psplines.samplers.multivar.multivar_blocked_nuts.multivariate_psplines_model`
multiplies the summed `log_delta_sq` term by ``Nh``, ensuring the total log-det term
matches the aggregated :math:`N_b N_h` DOF. The trace term uses the **summed**
statistic :math:`\bar{\mathbf{Y}}_h` directly, so no additional :math:`N_h` factors appear.
Coarse graining therefore increases information linearly in :math:`N_h`.

No log-binning, hybrid schemes, or “preserve low frequencies” modes are
implemented.
