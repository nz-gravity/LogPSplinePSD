Conventions
===========

This page records naming conventions used throughout the documentation and codebase.

Variable names
--------------

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Symbol
     - Meaning
   * - ``T``
     - Total duration.
   * - ``dt``
     - Sampling time interval.
   * - ``n``
     - Number of time points.
   * - ``N``
     - Number of frequency points (from the discrete FFT).
   * - ``p``
     - Number of channels.
   * - ``Nb``
     - Number of blocks.
   * - ``Lb``
     - Number of samples per block (n//Nb).
   * - ``Nc``
     - Number of coarse-grain bins.
   * - ``Nh``
     - Number of frequencies in each coarse-grain bin.
   * - ``Z``
     - Raw time series data, :math:`Z = (Z_1, \ldots, Z_n)^\top \in \mathbb{R}^{n \times p}`.
   * - ``d_k``
     - Fourier transform (or DFT) vector at frequency :math:`f_k`,
       :math:`d_k = (d_1(f_k), d_2(f_k), \ldots, d_p(f_k))^\top`.
   * - ``S(fk)``
     - :math:`p \times p` Hermitian positive semidefinite spectral density matrix at :math:`f_k`.
   * - ``I``
     - Periodogram.
   * - ``Ibar``
     - Block-averaged periodogram.
   * - ``Y``
     - Sum over all blocked FFT periodograms, :math:`Y = \bar{I}\,N_b` (not averaged).

Eigendecomposition conventions
------------------------------

For a Hermitian positive semidefinite matrix :math:`Y(f_k)` we use

.. math::

   Y(f_k) = \sum_{\nu=1}^p \lambda^{(k)}_\nu\, v^{(k)}_\nu v^{(k)*}_\nu
          = \sum_{\nu=1}^p u^{(k)}_\nu u^{(k)*}_\nu,
   \qquad
   u^{(k)}_\nu = \sqrt{\lambda^{(k)}_\nu}\, v^{(k)}_\nu,

where :math:`(\cdot)^*` denotes the conjugate transpose.

Coarse-graining conventions
---------------------------

We “coarse-grain” the Fourier frequencies by dividing them into :math:`N_c` subsequent disjoint subsets
:math:`J_h` (each containing an odd number :math:`N_h` of frequencies):

.. math::

   \{f_1, \ldots, f_N\} = \bigcup_{h=1}^{N_c} J_h.

Let :math:`\bar{f}_h` denote the midpoint Fourier frequency of interval :math:`J_h`, and define the
coarse-grained sum

.. math::

   \bar{Y}_h = \sum_{f \in J_h} Y(f),
   \qquad h=1,\ldots,N_c.

We also use the eigendecomposition of :math:`\bar{Y}_h`:

.. math::

   \bar{Y}_h = \sum_{\nu=1}^p \lambda^{(h)}_\nu\, v^{(h)}_\nu v^{(h)*}_\nu
             = \sum_{\nu=1}^p u^{(h)}_\nu u^{(h)*}_\nu,
   \qquad
   u^{(h)}_\nu = \sqrt{\lambda^{(h)}_\nu}\, v^{(h)}_\nu.

PSD terminology
---------------

``log_psplines.spectrum_utils`` centralises the conversion between Wishart
statistics and PSD matrices. The current conventions are:

- **Normalisation** – PSD matrices are **one-sided** and expressed per Hz. The
  helper :func:`wishart_matrix_to_psd` simply divides the summed Wishart
  matrices by the effective degrees of freedom, i.e., :math:`S(f) = Y(f) / N_b`.
- **Degrees of freedom** – :func:`compute_effective_Nb` multiplies the baseline
  block count by the coarse-grain bin counts, so the PSD conversion always uses
  the correct ``ν`` for each frequency bin.
- **Scaling factor** – The optional ``scaling_factor`` tracks variance
  adjustments applied during time-domain standardisation and is folded into the
  PSD conversion exactly once.
- **Terminology** – ``U(f)`` denotes the eigenvector-weighted replicates and
  ``Y(f) = U(f)U(f)^H`` denotes the summed Wishart matrices. Diagonal elements
  represent auto PSDs; off-diagonal elements are cross spectral densities.

These helpers ensure the frequency-domain likelihood, diagnostics, and plotting
code all consume spectra with the same units and sidedness.

Data flow
---------

The multivariate pipeline follows a fixed sequence of transformations:

1. **Timeseries** – raw or standardised time-domain data.
2. **MultivarFFT** – ``to_wishart_stats`` produces frequency grids, FFT means, and
   the eigenvector replicates ``U(f)`` on the positive-frequency grid.
3. **CoarseGrain** – optional **linear, full-band** binning combines nearby
   frequencies by summing :math:`\bar Y_h = \sum_{f\in J_h} Y(f)` and assigns each
   bin the member count :math:`Nh` for log-determinant scaling.
4. **Sampler** – NumPyro samplers consume the (possibly coarse) Wishart stats
   and spline models.
5. **ArviZ conversion** – ``wishart_u_to_psd`` populates
   ``observed_data['periodogram']`` using the canonical normalisation.
6. **Plotting** – visualisers consume the precomputed posterior quantiles and
   empirical PSD without re-deriving spectra.
