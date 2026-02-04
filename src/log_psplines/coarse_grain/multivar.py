from __future__ import annotations

from typing import Tuple

import numpy as np

from ..datatypes.multivar import MultivarFFT
from ..spectrum_utils import (
    sum_wishart_outer_products,
    u_to_wishart_matrix,
    wishart_matrix_to_psd,
)
from .preprocess import CoarseGrainSpec


def coarse_grain_multivar_fft(
    fft: MultivarFFT, spec: CoarseGrainSpec
) -> Tuple[MultivarFFT, np.ndarray]:
    """Coarse-grain a MultivarFFT using a precomputed binning spec.

    This aggregates the eigenvector-weighted periodogram components ``U(f)``
    within each bin by summing Y(f) = U(f) U(f)^H and recomputing the
    eigen-decomposition of the summed matrix. The representative frequency of
    each bin is provided by ``spec.f_coarse``. The returned frequency weights
    correspond to the number of fine frequencies in each coarse bin and should
    scale the log-determinant term in the likelihood (i.e. multiply ``nu``).

    Args:
        fft: Original multivariate FFT/Wishart statistics.
        spec: Coarse graining specification.

    Returns:
        (fft_coarse, weights), where ``weights`` has length ``len(spec.f_coarse)``
        and ``fft_coarse`` has matching frequency axis.
    """
    # Select the in-range frequencies from the original arrays
    selection = np.asarray(spec.selection_mask, dtype=bool)
    if selection.shape[0] != fft.n_freq:
        # Allow specs computed on already-trimmed frequency grids
        if selection.shape[0] != fft.freq.shape[0]:
            raise ValueError(
                "CoarseGrainSpec.selection_mask length does not match FFT frequencies"
            )
    # If selection is all True this is a no-op
    y_re_sel = fft.y_re[selection]
    y_im_sel = fft.y_im[selection]
    u_re_sel = fft.u_re[selection]
    u_im_sel = fft.u_im[selection]

    mask_high = np.asarray(spec.mask_high, dtype=bool)
    n_bins_high = int(spec.n_bins_high)
    if n_bins_high <= 0:
        raise ValueError("Coarse-graining spec has no bins.")

    u_high = (u_re_sel[mask_high] + 1j * u_im_sel[mask_high]).astype(
        np.complex128
    )
    y_re_high = y_re_sel[mask_high]
    y_im_high = y_im_sel[mask_high]

    sort_idx = np.asarray(spec.sort_indices, dtype=np.int64)
    bin_counts = np.asarray(spec.bin_counts, dtype=np.int64)

    # Sort high-frequency arrays to contiguous bins
    u_high_sorted = u_high[sort_idx]
    y_re_high_sorted = y_re_high[sort_idx]
    y_im_high_sorted = y_im_high[sort_idx]

    # Pre-allocate outputs
    n_dim = fft.n_dim
    u_bins = np.zeros((n_bins_high, n_dim, n_dim), dtype=np.complex128)
    y_re_bins = np.zeros((n_bins_high, n_dim), dtype=np.float64)
    y_im_bins = np.zeros((n_bins_high, n_dim), dtype=np.float64)

    # Iterate over bins using counts
    pos = 0
    for b in range(n_bins_high):
        count = int(bin_counts[b])
        if count <= 0:
            continue
        sl = slice(pos, pos + count)
        pos += count

        # Sum of y across member frequencies (for diagnostics only)
        y_re_bins[b] = np.sum(y_re_high_sorted[sl], axis=0)
        y_im_bins[b] = np.sum(y_im_high_sorted[sl], axis=0)

        # Sum Y = Σ U U^H then eigendecompose
        Y_sum = sum_wishart_outer_products(u_high_sorted[sl])
        # Numeric guard
        try:
            eigvals, eigvecs = np.linalg.eigh(Y_sum)
        except np.linalg.LinAlgError:
            # Fallback to symmetric part if numerical issues arise
            Ys = 0.5 * (Y_sum + Y_sum.conj().T)
            eigvals, eigvecs = np.linalg.eigh(Ys)

        eigvals = np.clip(eigvals.real, a_min=0.0, a_max=None)
        sqrt_eig = np.sqrt(eigvals).astype(np.float64)
        # Columns: v_j * sqrt(λ_j)
        u_bins[b] = eigvecs * sqrt_eig[np.newaxis, :]

    u_coarse = u_bins
    y_re_coarse = y_re_bins
    y_im_coarse = y_im_bins

    # Frequency weights: bin counts for each coarse bin
    weights = bin_counts.astype(float)

    # Build the coarse FFT structure and attach the aggregated PSD for diagnostics.
    u_re_coarse = u_coarse.real.astype(np.float64)
    u_im_coarse = u_coarse.imag.astype(np.float64)

    psd_coarse = wishart_matrix_to_psd(
        u_to_wishart_matrix(u_coarse),
        nu=int(fft.nu),
        scaling_factor=float(fft.scaling_factor or 1.0),
        weights=np.asarray(weights, dtype=np.float64),
    )

    fft_coarse = MultivarFFT(
        y_re=y_re_coarse.astype(np.float64),
        y_im=y_im_coarse.astype(np.float64),
        u_re=u_re_coarse,
        u_im=u_im_coarse,
        freq=np.asarray(spec.f_coarse, dtype=np.float64),
        n_freq=int(spec.f_coarse.shape[0]),
        n_dim=int(fft.n_dim),
        nu=int(fft.nu),
        scaling_factor=fft.scaling_factor,
        fs=fft.fs,
        raw_psd=psd_coarse.astype(np.complex128),
        raw_freq=np.asarray(spec.f_coarse, dtype=np.float64),
        channel_stds=fft.channel_stds,
        freq_bin_counts=np.asarray(weights, dtype=np.float64),
    )

    return fft_coarse, weights
