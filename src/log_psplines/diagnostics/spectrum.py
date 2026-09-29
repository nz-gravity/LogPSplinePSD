"""Post-fit spectrum comparison; whitening checks can live here later."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import simpson

if TYPE_CHECKING:
    from log_psplines.results import PSDResult


def _real_components(values: np.ndarray) -> np.ndarray:
    """Encode diagonal, real cross-spectrum and imaginary cross-spectrum."""
    matrix = np.asarray(values, dtype=np.complex128)
    p = matrix.shape[-1]
    upper = np.triu(np.ones((p, p), dtype=bool))
    return np.where(upper, matrix.real, matrix.imag)


def _coherence(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.real(np.diagonal(matrix, axis1=-2, axis2=-1))
    denominator = diagonal[..., :, None] * diagonal[..., None, :]
    return np.divide(
        np.abs(matrix) ** 2,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def spectrum_diagnostics(
    result: PSDResult, *, truth: np.ndarray | None = None
) -> dict[str, object]:
    """Compare posterior spectra with truth on the result's native grid.

    Stationary truth has shape (F, p, p), or (F,) for a scalar fit.
    Time-varying scalar truth has shape (T, F). Frequency endpoints are
    excluded when at least four bins are available.
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
    reference = np.asarray(truth)
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
        difference = np.abs(
            _coherence(median)[..., offdiag]
            - _coherence(reference)[..., offdiag]
        )
        metrics["coherence_mae"] = float(np.mean(difference))
    return metrics


__all__ = ["spectrum_diagnostics"]
