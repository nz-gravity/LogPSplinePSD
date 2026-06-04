Coarse-Graining
================

The multivariate PSD pipeline optionally groups nearby Fourier frequencies into
coarse bins so that each bin contributes a single aggregated Wishart statistic.
This documents the design of that discretization and shows how the implementation
realizes the theoretical approximations stated in the repository.

Frequency bins
--------------

The retained fine-frequency grid :math:`\{f_1,\dots,f_{N_\ell}\}\subset[f_{\min},f_{\max}]`
is coarse-grained by dividing it into consecutive, disjoint subsets
:math:`J_h`. Each :math:`J_h` contains :math:`N_h` Fourier frequencies, and
:math:`\bar f_h` denotes the midpoint Fourier frequency of :math:`J_h`. For
even :math:`N_h`, the implementation uses the lower-middle member of the
discrete Fourier grid.

The band limits :math:`[f_{\min}, f_{\max}]` are chosen upstream via
``model.fmin``/``model.fmax``. Coarse-grain configuration controls only binning
(``Nc``/``Nh``).

The binning logic is implemented by
:func:`log_psplines.preprocessing.coarse_grain.compute_binning_structure`, which
returns :class:`log_psplines.preprocessing.coarse_grain.CoarseGrainSpec`. Only
linear, full-band binning is supported. Exactly one construction mode must be
chosen:

- ``Nh``: fixed membership with equal-size bins that divide the retained
  frequency count exactly.
- ``Nc``: fixed bin count. If the retained count is not divisible by ``Nc``,
  trailing frequencies are trimmed so bins are equal sized.

The spec stores

- midpoint frequencies,
- start and midpoint indices for each contiguous bin,
- the constant member count :math:`N_h`, and
- the number of coarse bins :math:`N_c`.

Aggregating FFT data
--------------------

:func:`log_psplines.preprocessing.coarse_grain.apply_coarse_grain_multivar_fft`
takes the :class:`log_psplines.datatypes.multivar.MultivarFFT` and
:class:`log_psplines.preprocessing.coarse_grain.CoarseGrainSpec` and builds the
coarse representation used during sampling. The frequencies are
grouped by bin across the **entire** retained band. Within each :math:`J_h`, the
individual Wishart matrices
:math:`\mathbf{Y}(f)=\mathbf{U}(f)\mathbf{U}(f)^H` are summed to form
:math:`\bar{\mathbf{Y}}_h = \sum_{f\in J_h}\mathbf{Y}(f)`, and the sum is
re-factorized to obtain a single :math:`\bar{\mathbf{U}}_h` per bin. The helper
:func:`log_psplines.datatypes.multivar_utils.Y_to_U` performs the eigensystem
factorisation used by this step.

The returned :class:`log_psplines.datatypes.multivar.MultivarFFT` has
``len(spec.f_coarse)`` frequencies and stores the constant bin size on
``fft_data.Nh``.

Likelihood scaling
------------------

The scalar :math:`N_h` scales **only** the log-determinant term so that each bin behaves like
a Wishart observation with :math:`N_b N_h` degrees of freedom:

.. math::

    \log \mathcal{L}
    \;\propto\;
    - \sum_{h=1}^{N_c} N_b N_h \log \left|\mathbf{S}(\bar f_h)\right|
    - \sum_{h=1}^{N_c} \frac{1}{T}\,\mathrm{tr}\!\left[\mathbf{S}(\bar f_h)^{-1}\,\bar{\mathbf{Y}}_h\right].

When coarse graining is enabled, the pipeline model reads ``Nh`` from
``fft_data.Nh``. The NumPyro model
:func:`log_psplines.pipeline.models._blocked_channel_model` multiplies the
summed ``log_delta_sq`` term by ``Nh``, ensuring the total log-det term matches
the aggregated :math:`N_b N_h` degrees of freedom. The trace term uses the
summed statistic :math:`\bar{\mathbf{Y}}_h` directly, so no additional
:math:`N_h` factor appears there.

No log-binning, hybrid schemes, or “preserve low frequencies” modes are
implemented.
