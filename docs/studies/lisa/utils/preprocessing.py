"""Block structure and coarse-graining setup for LISA analysis."""

from __future__ import annotations

import numpy as np

from log_psplines.example_datasets.lisa_data import LIGHT_TRAVEL_TIME
from log_psplines.logger import logger
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    compute_binning_structure,
)

FMIN = 1e-4
FMAX = 1e-1


def setup_coarse_grain(
    Nl_analysis: int,
    target_Nc: int,
) -> CoarseGrainConfig:
    """Build a CoarseGrainConfig for the given analysis frequency count.

    Parameters
    ----------
    Nl_analysis : int
        Number of positive frequencies retained in [FMIN, FMAX].
    target_Nc : int
        Target number of coarse-grained bins. 0 = disabled.
    """
    if target_Nc <= 0:
        logger.info("Coarse graining disabled (full frequency resolution).")
        return CoarseGrainConfig(enabled=False)

    Nc = target_Nc
    if Nl_analysis % Nc != 0:
        candidates = [
            k for k in range(1, Nl_analysis + 1) if Nl_analysis % k == 0
        ]
        Nc = max(c for c in candidates if c <= min(Nc, Nl_analysis))
        logger.info(
            f"Adjusting coarse bins: Nl={Nl_analysis} not divisible by "
            f"{target_Nc}; using Nc={Nc}."
        )

    if Nc < 32:
        logger.warning(f"Coarse graining would use Nc={Nc} (<32); disabling.")
        return CoarseGrainConfig(enabled=False)

    logger.info(f"Coarse graining enabled with Nc={Nc}.")
    return CoarseGrainConfig(enabled=True, Nc=Nc)


def compute_Nl_analysis(
    Lb: int,
    dt: float,
    *,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> int:
    """Count positive frequencies in [fmin, fmax] for a given block length."""
    freq = np.fft.rfftfreq(Lb, d=dt)[1:]
    mask = (freq >= float(fmin)) & (freq <= float(fmax))
    Nl = int(np.count_nonzero(mask))
    if Nl < 1:
        raise ValueError(
            f"No positive frequencies in [{fmin}, {fmax}] for Lb={Lb}."
        )
    return Nl


def compute_analysis_frequencies(
    Lb: int,
    dt: float,
    *,
    fmin: float = FMIN,
    fmax: float = FMAX,
    coarse_cfg: CoarseGrainConfig | None = None,
) -> np.ndarray:
    """Return the retained analysis frequency grid used by the sampler.

    Parameters
    ----------
    Lb : int
        Samples per Wishart block.
    dt : float
        Sampling interval in seconds.
    fmin, fmax : float
        Analysis band limits in Hz.
    coarse_cfg : CoarseGrainConfig, optional
        If enabled, return the coarse representative frequencies instead of the
        fine Wishart grid.
    """
    freq = np.fft.rfftfreq(Lb, d=dt)[1:]
    mask = (freq >= float(fmin)) & (freq <= float(fmax))
    retained = np.asarray(freq[mask], dtype=np.float64)
    if retained.size == 0:
        raise ValueError(
            f"No positive frequencies in [{fmin}, {fmax}] for Lb={Lb}."
        )

    if coarse_cfg is None or not coarse_cfg.enabled:
        return retained

    Nc = coarse_cfg.Nc
    Nh = coarse_cfg.Nh
    if Nc is not None:
        if retained.size % int(Nc) != 0:
            candidates = [
                k
                for k in range(1, retained.size + 1)
                if retained.size % k == 0
            ]
            Nc = max(c for c in candidates if c <= min(int(Nc), retained.size))
    elif Nh is not None and retained.size % int(Nh) != 0:
        raise ValueError(
            f"Nh={Nh} must divide retained frequency count {retained.size}."
        )

    spec = compute_binning_structure(
        retained,
        Nc=Nc,
        Nh=Nh,
    )
    return np.asarray(spec.f_coarse, dtype=np.float64)


def compute_transfer_null_frequencies(
    *,
    fmin: float = FMIN,
    fmax: float = FMAX,
    light_travel_time: float = LIGHT_TRAVEL_TIME,
) -> np.ndarray:
    """Return TDI transfer-null frequencies k / (4 * light_travel_time)."""
    if light_travel_time <= 0.0:
        raise ValueError("light_travel_time must be positive.")
    if fmax < fmin:
        raise ValueError("fmax must be >= fmin.")

    base = 1.0 / (4.0 * float(light_travel_time))
    k_max = int(np.floor(float(fmax) / base))
    if k_max < 1:
        return np.zeros((0,), dtype=np.float64)
    freqs = base * np.arange(1, k_max + 1, dtype=np.float64)
    return freqs[(freqs >= float(fmin)) & (freqs <= float(fmax))]


def build_transfer_null_exclusion_bands(
    freq: np.ndarray,
    *,
    bins_per_side: int = 1,
    half_width: float | None = None,
    fmin: float = FMIN,
    fmax: float = FMAX,
    light_travel_time: float = LIGHT_TRAVEL_TIME,
) -> tuple[tuple[float, float], ...]:
    """Build null-centered exclusion bands for a retained frequency grid."""
    freq = np.asarray(freq, dtype=np.float64)
    if freq.ndim != 1 or freq.size == 0:
        raise ValueError("freq must be a non-empty 1-D array.")
    if bins_per_side < 0:
        raise ValueError("bins_per_side must be non-negative.")

    null_freqs = compute_transfer_null_frequencies(
        fmin=fmin,
        fmax=fmax,
        light_travel_time=light_travel_time,
    )
    if null_freqs.size == 0:
        return ()

    if half_width is None:
        if freq.size < 2:
            raise ValueError(
                "Need at least two frequency bins to infer exclusion width."
            )
        spacing = float(np.median(np.diff(np.sort(freq))))
        half_width = max(0.5 * spacing, bins_per_side * spacing)
    half_width = float(half_width)
    if half_width < 0.0:
        raise ValueError("half_width must be non-negative.")

    bands: list[tuple[float, float]] = []
    for f0 in null_freqs:
        low = max(float(fmin), float(f0 - half_width))
        high = min(float(fmax), float(f0 + half_width))
        bands.append((low, high))
    return tuple(bands)
