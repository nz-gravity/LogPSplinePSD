import os

import matplotlib.pyplot as plt
import numpy as np
from examples.plot_dcov_test import n_samples
from lisa_data import TEN_DAYS, LISAData, download

from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.psd_matrix import (
    plot_psd_matrix,
)
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines
from log_psplines.samplers.multivar.multivar_nuts import (
    MultivarNUTSConfig,
    MultivarNUTSSampler,
)

RESULT_FN = "results/lisa/inference_data.nc"
DATA_FPATH = "data/tdi.h5"
if not os.path.exists(DATA_FPATH):
    download(TEN_DAYS, dest_folder="data")
lisa_data = LISAData.from_hdf5(DATA_FPATH)

timeseries = MultivariateTimeseries(
    t=lisa_data.t,
    y=lisa_data.data,
)
print(timeseries)

FMIN, FMAX = 5**-5, 6**-2


if os.path.exists(RESULT_FN):
    print(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az

    idata = az.from_netcdf(RESULT_FN)

else:

    idata = run_mcmc(
        data=timeseries,
        sampler="nuts",
        n_samples=100,
        n_warmup=100,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(strategy="log"),
        outdir="results/lisa",
        verbose=True,
    )


print(idata)
