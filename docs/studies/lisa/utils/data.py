"""LISA data generation for simulation study.

Wraps lisa_datagen.generate_lisatools_xyz_noise_timeseries() to produce
a MultivariateTimeseries trimmed to block-consistent shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project source is on the path.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from log_psplines.datatypes import MultivariateTimeseries  # noqa: E402
from log_psplines.example_datasets.lisa_data import LASER_FREQ  # noqa: E402
from log_psplines.logger import logger  # noqa: E402

# Import the generator from the sibling datagen script.
LISA_DIR = Path(__file__).resolve().parent.parent
if str(LISA_DIR) not in sys.path:
    sys.path.insert(0, str(LISA_DIR))

from lisa_datagen import generate_lisatools_xyz_noise_timeseries  # noqa: E402

SEC_IN_DAY = 86_400.0

# Fixed LISA noise generation parameters.
DELTA_T = 5.0  # seconds (0.2 Hz sampling rate)
MODEL = "scirdv1"
GENERATION_FMIN = 1e-5
GENERATION_FMAX = 1e-1
CHOLESKY_FLOOR_REL = 1e-12
CHOLESKY_FLOOR_ABS = 0.0
FREQ_CHUNK_SIZE = 200_000


def generate_lisa_data(
    seed: int,
    duration_days: float,
    block_days: float,
    fmin_generate: float = GENERATION_FMIN,
    fmax_generate: float = GENERATION_FMAX,
    absolute_freq_units: bool = False,
) -> tuple[MultivariateTimeseries, np.ndarray, np.ndarray, int, int, float]:
    """Generate LISA XYZ noise and return block-trimmed timeseries.

    Returns
    -------
    ts : MultivariateTimeseries
        Block-trimmed XYZ timeseries.
    freq_true : np.ndarray
        True PSD frequencies (full resolution, before coarse graining).
    S_true : np.ndarray
        True spectral matrix (N_freq, 3, 3), complex.
    Nb : int
        Number of time blocks.
    Lb : int
        Samples per block.
    dt : float
        Sampling interval in seconds.
    """
    duration = duration_days * SEC_IN_DAY
    dt = DELTA_T

    x_t, y_t, z_t, freq_true, S_true = generate_lisatools_xyz_noise_timeseries(
        duration=duration,
        delta_t=dt,
        model=MODEL,
        seed=seed,
        fmin_generate=float(fmin_generate),
        fmax_generate=float(fmax_generate),
        cholesky_floor_rel=CHOLESKY_FLOOR_REL,
        cholesky_floor_abs=CHOLESKY_FLOOR_ABS,
        freq_chunk_size=FREQ_CHUNK_SIZE,
    )

    n = len(x_t)
    Lb = int(round(block_days * SEC_IN_DAY / dt))
    Lb = min(Lb, n)
    Nb = max(1, n // Lb)
    n_used = Nb * Lb

    if n_used != n:
        n_trim = n - n_used
        logger.info(
            f"Trimming {n_trim} samples to fit {Nb} blocks of {Lb} samples "
            f"({Lb * dt:.0f} s each)."
        )

    x_t = x_t[:n_used]
    y_t = y_t[:n_used]
    z_t = z_t[:n_used]

    y_full = np.vstack((x_t, y_t, z_t)).T.astype(np.float64)
    if absolute_freq_units:
        # lisatools XYZ outputs are fractional-frequency-like observables.
        # Scaling the time series by the laser carrier frequency converts them
        # to absolute-frequency fluctuations, so the PSD scales by nu0^2.
        y_full = y_full * float(LASER_FREQ)
        S_true = (
            np.asarray(S_true, dtype=np.complex128) * float(LASER_FREQ) ** 2
        )
    t_full = np.arange(n_used, dtype=np.float64) * dt
    ts = MultivariateTimeseries(y=y_full, t=t_full)

    logger.info(
        f"Generated LISA data: seed={seed}, {duration_days:.0f} days, "
        f"Nb={Nb}, Lb={Lb}, n={n_used}."
    )

    return ts, freq_true, S_true, Nb, Lb, dt
