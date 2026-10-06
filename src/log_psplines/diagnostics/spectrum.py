"""Post-fit spectrum comparison; whitening checks can live here later."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import simpson

from log_psplines.models.matrix import SpectralMatrix

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def _real_components(values: np.ndarray) -> np.ndarray:
    """Encode diagonal, real cross-spectrum and imaginary cross-spectrum."""
    matrix = np.asarray(values, dtype=np.complex128)
    p = matrix.shape[-1]
    upper = np.triu(np.ones((p, p), dtype=bool))
    return np.where(upper, matrix.real, matrix.imag)


def spectrum_diagnostics(
    result: PSDResult, *, truth: np.ndarray | None = None
) -> dict[str, object]:
    """Compare posterior spectra with truth on the result's native grid.

    Stationary truth has shape (F, p, p), or (F,) for a scalar fit.
    Time-varying truth has shape (T, F, p, p), or (T, F) for a scalar fit.
    Frequency endpoints are excluded when at least four bins are available.
    Coherence error uses the median of per-draw coherence, using all-draw
    cached summaries when only a preview of spectra is retained.
    """
    if truth is None:
        truth = (
            result.truth
            if result.truth is not None
            else result.metadata.get("true_psd")
        )
    if truth is None:
        return {}

    time_varying = result.time is not None
    frequency = result.frequency
    selection = slice(1, -1) if frequency.size > 3 else slice(None)
    frequency = frequency[selection]
    quantiles = np.asarray(result.quantiles((5.0, 50.0, 95.0)))
    quantiles = (
        quantiles[:, :, selection] if time_varying else quantiles[:, selection]
    )
    median = quantiles[1]
    reference = np.asarray(truth, dtype=np.complex128)
    if reference.ndim == (2 if time_varying else 1):
        reference = reference[..., None, None]
    elif time_varying and reference.ndim == 3:
        reference = reference[..., :, None] * np.eye(reference.shape[-1])
    reference = (
        reference[:, selection] if time_varying else reference[selection]
    )
    if reference.shape != median.shape:
        raise ValueError(
            f"truth must match the fitted spectrum shape {median.shape} "
            f"after frequency endpoint removal; got {reference.shape}"
        )
    lower, upper = _real_components(quantiles[[0, 2]])

    # Integrate the full Frobenius norm over frequency, then sum over time.
    error_norm = np.linalg.norm(median - reference, axis=(-2, -1))
    truth_norm = np.linalg.norm(reference, axis=(-2, -1))
    frequency_axis = 1 if time_varying else 0
    error_integral = float(
        np.sum(simpson(error_norm, x=frequency, axis=frequency_axis))
    )
    truth_integral = float(
        np.sum(simpson(truth_norm, x=frequency, axis=frequency_axis))
    )
    error_sq = float(
        np.sum(simpson(error_norm**2, x=frequency, axis=frequency_axis))
    )
    truth_sq = float(
        np.sum(simpson(truth_norm**2, x=frequency, axis=frequency_axis))
    )
    encoded_truth = _real_components(reference)
    if result.metadata.get("likelihood") == "diagonal_power_whittle":
        lower = np.diagonal(lower, axis1=-2, axis2=-1)
        upper = np.diagonal(upper, axis1=-2, axis2=-1)
        encoded_truth = np.diagonal(encoded_truth, axis1=-2, axis2=-1)
    metrics: dict[str, object] = {
        "riae": error_integral / truth_integral
        if truth_integral > 0
        else float("nan"),
        "l2": float(np.sqrt(error_sq / truth_sq))
        if truth_sq > 0
        else float("nan"),
        "coverage": float(
            np.mean((encoded_truth >= lower) & (encoded_truth <= upper))
        ),
    }
    p = reference.shape[-1]
    channel_riae = []
    for channel in range(p):
        estimate = np.real(median[..., channel, channel])
        target = np.real(reference[..., channel, channel])
        numerator = float(
            np.sum(
                simpson(
                    np.abs(estimate - target),
                    x=frequency,
                    axis=frequency_axis,
                )
            )
        )
        denominator = float(
            np.sum(simpson(target, x=frequency, axis=frequency_axis))
        )
        channel_riae.append(
            numerator / denominator if denominator > 0 else float("nan")
        )
    metrics["channel_riae"] = channel_riae
    if p > 1 and result.metadata.get("likelihood") != "diagonal_power_whittle":
        offdiag = np.triu(np.ones((p, p), dtype=bool), k=1)
        summary = result.spectrum_summary
        if summary is not None and "coherence_quantiles" in summary:
            coherence = np.asarray(
                summary["coherence_quantiles"].sel(percentile=50)
            )
        else:
            coherence = np.median(result.coherence, axis=(0, 1))
        coherence = (
            coherence[:, selection] if time_varying else coherence[selection]
        )
        difference = np.abs(
            coherence[..., offdiag]
            - SpectralMatrix.coherence(reference)[..., offdiag]
        )
        metrics["coherence_mae"] = float(np.mean(difference))
    return metrics


__all__ = ["spectrum_diagnostics"]
