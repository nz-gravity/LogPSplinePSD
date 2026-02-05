# Conventions

## PSD terminology

`log_psplines.spectrum_utils` centralises the conversion between Wishart
statistics and PSD matrices. The current conventions are:

- **Normalisation** – PSD matrices are **one-sided** and expressed per Hz. The
  helper :func:`wishart_matrix_to_psd` simply divides the summed Wishart
  matrices by the effective degrees of freedom, i.e., :math:`S(f) = Y(f) / \nu`.
- **Degrees of freedom** – :func:`compute_effective_nu` multiplies the baseline
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

## Data flow

The multivariate pipeline follows a fixed sequence of transformations:

1. **Timeseries** – raw or standardised time-domain data.
2. **MultivarFFT** – `to_wishart_stats` produces frequency grids, FFT means, and
   the eigenvector replicates ``U(f)`` on the positive-frequency grid.
3. **CoarseGrain** – optional **linear, full-band** binning combines nearby
   frequencies by summing \(\bar Y_h = \sum_{f\in J_h} Y(f)\) and assigns each
   bin the member count \(Nh\) for log-determinant scaling.
4. **Sampler** – NumPyro samplers consume the (possibly coarse) Wishart stats
   and spline models.
5. **ArviZ conversion** – `wishart_u_to_psd` populates
   ``observed_data['periodogram']`` using the canonical normalisation.
6. **Plotting** – visualisers consume the precomputed posterior quantiles and
   empirical PSD without re-deriving spectra.
