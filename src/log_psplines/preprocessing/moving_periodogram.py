"""Tang zig-zag moving-periodogram preprocessing.

The raw transform follows Definition 1 of Tang et al. (2026): centred
``2*m+1``-sample windows are assigned one of ``m`` Fourier frequencies in a
zig-zag pattern.  Thinning is applied by retaining complete blocks, which is
the dependence-control step used by the dynamic-Whittle implementation.

For exact scattered-coordinate fitting, use
:func:`scattered_moving_periodogram`. :func:`moving_periodogram` instead pools
the ordinates onto a rectangular ``PowerSpectrum`` grid. Use
:func:`tang_moving_periodogram` when complex coefficients are also needed.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from log_psplines.data.spectral import PowerSpectrum, ScatteredPowerSpectrum


class MovingPeriodogram(TypedDict):
    """Raw scattered moving-periodogram ordinates."""

    u: np.ndarray
    omega: np.ndarray
    coeff: np.ndarray
    mi: np.ndarray


def tang_moving_periodogram(
    data: np.ndarray, *, m: int, thin: int = 2
) -> MovingPeriodogram:
    """Return thinned zig-zag moving-periodogram ordinates.

    ``data`` is a real series of length ``T``.  ``m`` is the half-width of
    each window, so each window has length ``2*m+1``.  The returned ``u`` is
    the true one-based window-centre divided by ``T`` and ``omega`` is angular
    frequency in radians per sample.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            "Moving-periodogram input data must be one-dimensional."
        )
    if not np.isfinite(x).all():
        raise ValueError("Moving-periodogram input data must be finite.")
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer.")
    if (
        isinstance(thin, bool)
        or not isinstance(thin, (int, np.integer))
        or thin < 1
    ):
        raise ValueError("thin must be a positive integer.")

    n_blocks = (x.size - 2 * m) // (thin * m)
    if n_blocks < 1:
        raise ValueError("Series too short for these (m, thin).")

    lam = 2.0 * np.arange(1, m + 1) / (2 * m + 1)
    omega = np.pi * lam
    nu = np.arange(2 * m + 1)
    phase = np.exp(-1j * np.pi * np.outer(nu, lam))
    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * m + 1)
    starts = (
        thin * m * np.arange(n_blocks)[:, None] + np.arange(m)[None, :]
    ).reshape(-1)
    freq_index = np.tile(np.arange(m), n_blocks)

    selected = windows[starts]
    coeff = np.einsum(
        "pn,pn->p", selected, phase.T[freq_index], optimize=True
    )
    coeff /= np.sqrt(2.0 * np.pi * (2 * m + 1))
    centres = starts + m + 1
    return {
        "u": centres / x.size,
        "omega": np.tile(omega, n_blocks),
        "coeff": coeff,
        "mi": np.abs(coeff) ** 2,
    }


def bin_tang_ordinates(
    ordinates: MovingPeriodogram,
    *,
    time_bin: int = 1,
    freq_bin: int = 1,
) -> dict[str, np.ndarray]:
    """Pool already-thinned ordinates into power/count observations.

    Each raw complex ordinate contributes ``summed_power=2*mi`` and
    ``counts=2``.  Pooling sums both quantities and averages the coordinates;
    thinning must happen before this operation.
    """
    for name, value in (("time_bin", time_bin), ("freq_bin", freq_bin)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer.")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    u = np.asarray(ordinates["u"], dtype=float)
    omega = np.asarray(ordinates["omega"], dtype=float)
    mi = np.asarray(ordinates["mi"], dtype=float)
    if any(a.ndim != 1 for a in (u, omega, mi)) or not u.size:
        raise ValueError(
            "Tang ordinate arrays must be non-empty and one-dimensional."
        )
    if not (u.size == omega.size == mi.size):
        raise ValueError("Tang ordinate arrays must have the same length.")
    if not np.isfinite(u).all() or not np.isfinite(omega).all():
        raise ValueError("Tang ordinate locations must be finite.")
    if not np.isfinite(mi).all() or np.any(mi < 0):
        raise ValueError("Tang powers must be finite and non-negative.")

    rungs = np.unique(omega)
    if u.size % rungs.size:
        raise ValueError(
            "Tang ordinate count must be divisible by frequencies."
        )
    n_blocks = u.size // rungs.size
    u = u.reshape(n_blocks, rungs.size)
    mi = mi.reshape(n_blocks, rungs.size)
    omega_2d = omega.reshape(n_blocks, rungs.size)
    if not np.allclose(omega_2d, rungs[None, :], atol=1e-14, rtol=0):
        raise ValueError(
            "Tang ordinates must use repeated block-major frequencies."
        )

    def pooled(values: np.ndarray) -> np.ndarray:
        rows = np.add.reduceat(
            values, np.arange(0, n_blocks, time_bin), axis=0
        )
        cols = np.add.reduceat(
            rows, np.arange(0, rungs.size, freq_bin), axis=1
        )
        return cols

    cells = pooled(np.ones_like(mi))
    return {
        "u": (pooled(u) / cells).reshape(-1),
        "omega": np.broadcast_to(
            np.add.reduceat(rungs, np.arange(0, rungs.size, freq_bin))
            / np.diff(np.r_[np.arange(0, rungs.size, freq_bin), rungs.size]),
            cells.shape,
        ).reshape(-1),
        "summed_power": (2.0 * pooled(mi)).reshape(-1),
        "counts": (2.0 * cells).reshape(-1),
    }


def moving_periodogram(
    data: np.ndarray,
    *,
    dt: float,
    m: int,
    thin: int = 2,
    time_bin: int = 1,
    freq_bin: int = 1,
) -> PowerSpectrum:
    """Prepare moving-periodogram powers for ``fit``.

    The returned frequency is in Hz and time is rescaled to the full record.
    Since :class:`~log_psplines.data.spectral.PowerSpectrum` is rectangular,
    time coordinates are the pooled centres of each retained block.  The
    exact scattered coordinates and complex coefficients remain available by
    calling :func:`tang_moving_periodogram` separately.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be strictly positive.")
    raw = tang_moving_periodogram(data, m=m, thin=thin)
    pooled = bin_tang_ordinates(raw, time_bin=time_bin, freq_bin=freq_bin)
    n_freq = np.unique(pooled["omega"]).size
    n_time = pooled["u"].size // n_freq
    power = pooled["summed_power"].reshape(n_time, n_freq)
    counts = pooled["counts"].reshape(n_time, n_freq)
    time = pooled["u"].reshape(n_time, n_freq).mean(axis=1)
    frequency = np.unique(pooled["omega"]) / (2.0 * np.pi * dt)
    return PowerSpectrum(
        power=power,
        counts=counts,
        frequency=frequency,
        time=time,
        units="moving-periodogram coefficient variance",
    )


__all__ = [
    "MovingPeriodogram",
    "bin_tang_ordinates",
    "moving_periodogram",
    "scattered_moving_periodogram",
    "tang_moving_periodogram",
]


def scattered_moving_periodogram(
    data: np.ndarray, *, dt: float, m: int, thin: int = 2
) -> ScatteredPowerSpectrum:
    """Prepare moving-periodogram powers at their exact (u, omega) ordinates.

    Unlike :func:`moving_periodogram`, no rectangular pooling is applied: each
    retained window keeps its own exact centre and rung frequency, so no
    cross-rung time-averaging is introduced. Pass the result to ``fit`` with
    an explicit ``LogPSpline``; the likelihood evaluates the spline at each
    ordinate's own time and frequency.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be strictly positive.")
    raw = tang_moving_periodogram(data, m=m, thin=thin)
    return ScatteredPowerSpectrum(
        power=2.0 * raw["mi"],
        counts=np.full_like(raw["mi"], 2.0),
        time=raw["u"],
        frequency=raw["omega"] / (2.0 * np.pi * dt),
        units="moving-periodogram coefficient variance",
    )
