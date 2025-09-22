from lisa_data import LISAData, download, TEN_DAYS
from log_psplines.mcmc import run_mcmc
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines
from log_psplines.samplers.multivar.multivar_nuts import MultivarNUTSSampler, MultivarNUTSConfig
import os
import matplotlib.pyplot as plt
from log_psplines.plotting.psd_matrix import plot_psd_matrix, compute_empirical_psd

import numpy as np

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

timeseries = timeseries.standardise()
fft_data = timeseries.to_cross_spectral_density(fmin=FMIN, fmax=FMAX)
print(fft_data)


if os.path.exists(RESULT_FN):
    print(f"Found existing results at {RESULT_FN}, loading...")
    import arviz as az
    idata = az.from_netcdf(RESULT_FN)

else:

    sampler = MultivarNUTSSampler(
        fft_data,
        MultivariateLogPSplines.from_multivar_fft(
            fft_data,
            n_knots=10,
            degree=3,
            diffMatrixOrder=2,
            knot_kwargs=dict(method='log')
        ),
        MultivarNUTSConfig(
            verbose=True,
            outdir="results/lisa",
        )
    )
    idata = sampler.sample(n_samples=200, n_warmup=200)


print(idata)

plot_psd_matrix(
    idata=idata,
    n_channels=3,
    freq=fft_data.freq,
    empirical_psd=compute_empirical_psd(fft_data.y_re, fft_data.y_im, n_channels=3),
    outdir="results/lisa",
    filename="psd_matrix_posterior.png",
    diag_yscale='log',
    xscale='log'
)