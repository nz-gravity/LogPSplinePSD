"""Dimensionless forward Whittle losses on declared quadrature grids.

These diagnostics do not construct observations or change the spline prior.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _weights(values: np.ndarray | None, size: int) -> np.ndarray:
    weights = (
        np.ones(size) if values is None else np.asarray(values, dtype=float)
    )
    if (
        weights.shape != (size,)
        or not size
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError(
            "weights must be finite, nonnegative and have positive sum"
        )
    return weights / weights.sum()


def whittle_loss(log_true: np.ndarray, log_estimate: np.ndarray) -> np.ndarray:
    """Forward loss d_W(true, estimate), broadcasting log spectra.

    Use expm1 to preserve the small-loss limit. Large ratios may return inf.
    """
    difference = np.asarray(log_true) - np.asarray(log_estimate)
    if not np.isfinite(difference).all():
        raise ValueError("log spectra must be finite")
    return np.expm1(difference) - difference


def stationarity_loss(
    log_spectrum: np.ndarray,
    *,
    time_weights: np.ndarray | None = None,
    frequency_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Jensen gap over full support; log_spectrum: (..., time, frequency).

    Weights are normalized separately. Missing/partial supports must be
    excluded by the caller; an empty or nonfinite surface is rejected.
    """
    logs = np.asarray(log_spectrum, dtype=float)
    if logs.ndim < 2 or not np.isfinite(logs).all():
        raise ValueError(
            "log_spectrum must be finite with time and frequency axes"
        )
    a = _weights(time_weights, logs.shape[-2])
    w = _weights(frequency_weights, logs.shape[-1])
    retained = a > 0
    # Shift before subtraction to avoid cancellation of arbitrary log scales.
    values = logs[..., retained, :]
    shift = values[..., :1, :]
    centered = values - shift
    log_mean = logsumexp(centered, b=a[retained, None], axis=-2)
    mean_log = np.sum(centered * a[retained, None], axis=-2)
    return np.asarray(np.sum((log_mean - mean_log) * w, axis=-1))


def horizon_summary(
    losses: np.ndarray,
    candidates: np.ndarray,
    *,
    epsilon: float,
    q: float = 0.95,
    prefix: bool = False,
) -> dict[str, np.ndarray]:
    """Summarize full-support losses (draw, ..., candidate).

    prefix=True takes a draw-wise cumulative maximum before probabilities.
    Empty acceptance is -1; largest-candidate acceptance is right-censored.
    This is a finite candidate/grid diagnostic, not a continuous guarantee.
    """
    losses = np.asarray(losses, dtype=float)
    candidates = np.asarray(candidates)
    if (
        losses.ndim < 2
        or losses.shape[0] == 0
        or candidates.ndim != 1
        or candidates.size == 0
        or losses.shape[-1] != candidates.size
        or np.any(np.diff(candidates) <= 0)
        or np.any(candidates <= 0)
        or not np.isfinite(candidates).all()
        or not np.isfinite(losses).all()
        or np.any(losses < -1e-12)
        or not np.isfinite(epsilon)
        or epsilon < 0
        or not 0 < q <= 1
    ):
        raise ValueError("invalid losses, candidates, epsilon or q")
    checked = np.maximum.accumulate(losses, axis=-1) if prefix else losses
    probability = np.mean(checked <= epsilon, axis=0)
    accepted = probability >= q
    return {
        "probability": probability,
        "selected": np.asarray(
            np.max(np.where(accepted, candidates, -1), axis=-1)
        ),
        "right_censored": np.asarray(accepted[..., -1]),
        "smallest_failed": np.asarray(~accepted[..., 0]),
        "none_acceptable": np.asarray(~np.any(accepted, axis=-1)),
    }
