from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..datatypes import MultivarFFT
from .preprocess import CoarseGrainSpec


@dataclass(slots=True)
class CoarseGrainedMultivar:
    """Result of coarse-graining multivariate FFT data."""

    fft: MultivarFFT
    weights: np.ndarray


def apply_coarse_graining_multivar_fft(
    fft: MultivarFFT,
    spec: CoarseGrainSpec,
) -> CoarseGrainedMultivar:
    """Coarse-grain multivariate FFT data using ``CoarseGrainSpec``.

    The coarse-graining strategy mirrors the univariate case:
    - Frequencies in the low region are kept untouched with unit weights.
    - High-frequency bins are replaced with their arithmetic mean across
      member frequencies; the corresponding weight equals the number of
      fine frequencies represented by the bin.

    Parameters
    ----------
    fft:
        Original multivariate FFT data.
    spec:
        Coarse-graining specification derived from the FFT frequency grid.

    Returns
    -------
    CoarseGrainedMultivar
        Coarse-grained FFT data along with frequency weights.
    """

    selected_idx = np.flatnonzero(spec.selection_mask)
    if selected_idx.size == 0:
        raise ValueError("Coarse-graining selection mask produced no indices.")

    # Select frequencies within [f_min, f_max]
    y_re_selected = np.asarray(fft.y_re[selected_idx])
    y_im_selected = np.asarray(fft.y_im[selected_idx])
    Z_re_selected = np.asarray(fft.Z_re[selected_idx])
    Z_im_selected = np.asarray(fft.Z_im[selected_idx])
    freq_selected = np.asarray(fft.freq[selected_idx])

    # Low-frequency region: keep as-is.
    y_re_low = y_re_selected[spec.mask_low]
    y_im_low = y_im_selected[spec.mask_low]
    Z_re_low = Z_re_selected[spec.mask_low]
    Z_im_low = Z_im_selected[spec.mask_low]
    freq_low = freq_selected[spec.mask_low]
    weights_low = np.ones(freq_low.shape[0], dtype=np.float64)

    fft_selected = y_re_selected + 1j * y_im_selected
    fft_low = fft_selected[spec.mask_low]

    csd_low = (
        fft_low[:, :, None] * np.conjugate(fft_low[:, None, :])
        if fft_low.size
        else np.zeros((0, fft.n_dim, fft.n_dim), dtype=np.complex128)
    )

    if spec.n_bins_high == 0:
        coarse_fft = MultivarFFT(
            y_re=y_re_low,
            y_im=y_im_low,
            Z_re=Z_re_low,
            Z_im=Z_im_low,
            freq=freq_low,
            n_freq=freq_low.shape[0],
            n_dim=fft.n_dim,
            scaling_factor=fft.scaling_factor,
            fs=fft.fs,
            bin_weights=weights_low,
            csd_sums=csd_low,
        )
        return CoarseGrainedMultivar(coarse_fft, weights_low)

    counts = np.asarray(spec.bin_counts, dtype=np.int32)
    if counts.shape[0] != spec.n_bins_high:
        raise ValueError("bin_counts length must match n_bins_high")

    y_re_high = y_re_selected[spec.mask_high]
    y_im_high = y_im_selected[spec.mask_high]
    Z_re_high = Z_re_selected[spec.mask_high]
    Z_im_high = Z_im_selected[spec.mask_high]
    freq_high_means = spec.f_coarse[spec.n_low :]

    # Compute cumulative offsets for slicing the high-frequency members.
    offsets = np.concatenate([[0], np.cumsum(counts)])

    # Complex FFT vectors for stats aggregation
    fft_high = fft_selected[spec.mask_high]

    outer_high = (
        fft_high[:, :, None] * np.conjugate(fft_high[:, None, :])
        if fft_high.size
        else np.zeros((0, fft.n_dim, fft.n_dim), dtype=np.complex128)
    )
    csd_high = np.zeros(
        (spec.n_bins_high, fft.n_dim, fft.n_dim), dtype=np.complex128
    )
    if outer_high.size:
        np.add.at(csd_high, spec.bin_indices, outer_high)

    y_re_bins = []
    y_im_bins = []
    Z_re_bins = []
    Z_im_bins = []
    for bin_idx, count in enumerate(counts):
        if count == 0:
            continue
        start = offsets[bin_idx]
        end = offsets[bin_idx + 1]
        y_re_bins.append(np.mean(y_re_high[start:end], axis=0))
        y_im_bins.append(np.mean(y_im_high[start:end], axis=0))
        Z_re_bins.append(np.mean(Z_re_high[start:end], axis=0))
        Z_im_bins.append(np.mean(Z_im_high[start:end], axis=0))

    y_re_bins_arr = np.asarray(y_re_bins, dtype=np.float64)
    y_im_bins_arr = np.asarray(y_im_bins, dtype=np.float64)
    Z_re_bins_arr = np.asarray(Z_re_bins, dtype=np.float64)
    Z_im_bins_arr = np.asarray(Z_im_bins, dtype=np.float64)

    freq_coarse = np.concatenate((freq_low, freq_high_means))
    y_re_coarse = np.concatenate((y_re_low, y_re_bins_arr), axis=0)
    y_im_coarse = np.concatenate((y_im_low, y_im_bins_arr), axis=0)
    Z_re_coarse = np.concatenate((Z_re_low, Z_re_bins_arr), axis=0)
    Z_im_coarse = np.concatenate((Z_im_low, Z_im_bins_arr), axis=0)
    weights_coarse = np.concatenate(
        (weights_low, counts.astype(np.float64)), axis=0
    )
    csd_coarse = np.concatenate((csd_low, csd_high), axis=0)

    coarse_fft = MultivarFFT(
        y_re=y_re_coarse,
        y_im=y_im_coarse,
        Z_re=Z_re_coarse,
        Z_im=Z_im_coarse,
        freq=freq_coarse,
        n_freq=freq_coarse.shape[0],
        n_dim=fft.n_dim,
        scaling_factor=fft.scaling_factor,
        fs=fft.fs,
        bin_weights=weights_coarse,
        csd_sums=csd_coarse,
    )
    return CoarseGrainedMultivar(coarse_fft, weights_coarse)


__all__ = ["CoarseGrainedMultivar", "apply_coarse_graining_multivar_fft"]
