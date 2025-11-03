# Conventions

## PSD terminology

The refactor introduces `log_psplines.spectrum_utils`, which centralises the
mathematics used to convert between eigenvector-weighted Wishart statistics and
power spectral density matrices. The key conventions are:

- **Normalisation** – All PSD matrices are one-sided and use the
  `2/(2π)` scaling implemented by :func:`wishart_matrix_to_psd`.
- **Degrees of freedom** – Effective ν values are obtained via
  :func:`compute_effective_nu`, which multiplies the baseline block count by
  any coarse-bin multiplicities.
- **Scaling factor** – The optional `scaling_factor` tracks the variance
  adjustment applied during timeseries standardisation and is folded into the
  PSD conversion exactly once.

- **Terminology** – ``U(f)`` denotes the eigenvector-weighted replicates that
  arise from the Wishart factorisation, while ``Y(f) = U(f)U(f)^H`` denotes the
  summed Wishart matrices prior to normalisation. Diagonal elements are referred to
  as PSDs; off-diagonal elements are CSDs. coherences are computed from CSDs.

Using these helpers removes a number of duplicated `np.einsum` expressions and
makes it much easier to reason about the frequency-domain likelihood and the
diagnostic plots derived from it.

## Data flow

The multivariate pipeline follows a fixed sequence of transformations:

1. **Timeseries** – raw or standardised time-domain data.
2. **MultivarFFT** – `to_wishart_stats` produces frequency grids, FFT means, and
   the eigenvector replicates ``U(f)``.
3. **CoarseGrain** – optional binning combines nearby frequencies and adjusts
   the effective degrees of freedom.
4. **Sampler** – NumPyro samplers consume the (possibly coarse) Wishart stats
   and spline models.
5. **ArviZ conversion** – `wishart_u_to_psd` populates
   ``observed_data['periodogram']`` using the canonical normalisation.
6. **Plotting** – visualisers consume the precomputed posterior quantiles and
   empirical PSD without re-deriving spectra.
