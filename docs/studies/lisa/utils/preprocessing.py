"""Block structure and coarse-graining setup for LISA analysis."""

from __future__ import annotations

import numpy as np

from log_psplines.logger import logger
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

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


def compute_Nl_analysis(Lb: int, dt: float) -> int:
    """Count positive frequencies in [FMIN, FMAX] for a given block length."""
    freq = np.fft.rfftfreq(Lb, d=dt)[1:]
    mask = (freq >= FMIN) & (freq <= FMAX)
    Nl = int(np.count_nonzero(mask))
    if Nl < 1:
        raise ValueError(
            f"No positive frequencies in [{FMIN}, {FMAX}] for Lb={Lb}."
        )
    return Nl
