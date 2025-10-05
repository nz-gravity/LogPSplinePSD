import os

import matplotlib.pyplot as plt
import numpy as np
from lisa_data import TEN_DAYS, LISAData, download

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


# lets also only save every 2 datapoint to reduce size
t = lisa_data.t[::2]
y = lisa_data.data[::2, :]

# lets truncate to only first 10% of data to reduce size
n_trunc = int(0.1 * len(t))
t = t[:n_trunc]
y = y[:n_trunc, :]

# standardise y
y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

timeseries = MultivariateTimeseries(y=y, t=t)
logger.info(timeseries)

FMIN, FMAX = 5**-5, 6**-2

fft_data = timeseries.to_cross_spectral_density()  # fmin=FMIN, fmax=FMAX)
logger.info(fft_data)


if os.path.exists(RESULT_FN) and False:
    logger.info(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(RESULT_FN)

else:

    idata = run_mcmc(
        data=fft_data,
        sampler="multivar-blocked-nuts",
        n_samples=100,
        n_warmup=100,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir=f"{HERE}/results/lisa",
        verbose=True,
    )

logger.info(idata)


plot_psd_matrix(
    idata=idata,
    n_channels=3,
    freq=fft_data.freq,
    empirical_psd=fft_data.empirical_psd,
    outdir=f"{HERE}/results/lisa",
    filename=f"psd_matrix.png",
    diag_yscale="log",
    xscale="log",
)
