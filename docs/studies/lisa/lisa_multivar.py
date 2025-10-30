import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lisa_data import TEN_DAYS, LISAData, download

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "src").exists()
)
sys.path.insert(0, str(REPO_ROOT / "src"))

try:  # pragma: no cover - optional dependency shim for docs
    import loguru  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types

    loguru = types.ModuleType("loguru")

    class _StubLogger:
        def __getattr__(self, name):
            def _(*args, **kwargs):
                pass

            return _

    loguru.logger = _StubLogger()
    sys.modules["loguru"] = loguru

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import plot_psd_matrix
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines

set_level("DEBUG")

HERE = os.path.dirname(os.path.abspath(__file__))

RESULT_FN = f"{HERE}/results/lisa/inference_data.nc"
DATA_FPATH = f"{HERE}/data/tdi.h5"
if not os.path.exists(DATA_FPATH):
    download(TEN_DAYS, dest_folder="data")
lisa_data = LISAData.from_hdf5(DATA_FPATH)


t = lisa_data.t
y = lisa_data.data
y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

# detrmine number of time blocks based on data length
n = y.shape[0]

# make n_blocks so each block is ~ 1week long, power of 2
target_blocks = max(1, 2 ** int(np.round(np.log2(n / (24 * 7)))))
while target_blocks > 1 and n % target_blocks != 0:
    target_blocks //= 2

# Require each block to contain at least one quarter of the samples.
while target_blocks > 4:
    target_blocks //= 2

n_blocks = target_blocks
n_inside_block = n // n_blocks
logger.info(
    f"Using n_blocks={n_blocks} x {n_inside_block} (n_time={n})",
)


timeseries = MultivariateTimeseries(y=y, t=t)
logger.info(timeseries)

FMIN, FMAX = 10**-4, 5 * 10**-1

fft_data = timeseries.to_wishart_stats(
    n_blocks=n_blocks,
    fmin=FMIN,
    fmax=FMAX,
)
logger.info(fft_data)

coarse_cfg = CoarseGrainConfig(
    enabled=True,
    f_transition=5e-3,
    n_log_bins=200,
    f_min=FMIN,
    f_max=FMAX,
)


if os.path.exists(RESULT_FN) and False:
    logger.info(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(RESULT_FN)

else:

    idata = run_mcmc(
        data=fft_data,
        sampler="multivar_blocked_nuts",
        n_samples=1000,
        n_warmup=1000,
        n_knots=20,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir=f"{HERE}/results/lisa",
        verbose=True,
        coarse_grain_config=coarse_cfg,
        fmin=FMIN,
        fmax=FMAX,
    )

logger.info(idata)


freq_plot = np.asarray(idata.posterior_psd["freq"].values)

plot_psd_matrix(
    idata=idata,
    freq=freq_plot,
    empirical_psd=fft_data.empirical_psd,
    outdir=f"{HERE}/results/lisa",
    filename=f"psd_matrix.png",
    diag_yscale="log",
    xscale="log",
)
