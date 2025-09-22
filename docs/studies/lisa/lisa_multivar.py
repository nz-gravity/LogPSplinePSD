from lisa_data import LISAData, download, TEN_DAYS
from log_psplines.mcmc import run_mcmc
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines
from log_psplines.samplers.multivar.multivar_nuts import MultivarNUTSSampler, MultivarNUTSConfig
import os
import matplotlib.pyplot as plt
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
            n_knots=5,
            degree=3,
            diffMatrixOrder=2,
            knot_kwargs=dict(method='log')
        ),
        MultivarNUTSConfig(
            verbose=True,
            outdir="results/lisa",
        )
    )
    idata = sampler.sample(n_samples=50, n_warmup=50)


print(idata)


def _plot_psd_matrix(idata, fft_data, outdir=None):
    """
    Plot the posterior mean PSD matrix for each channel pair in log-log scale.
    Args:
        idata: ArviZ InferenceData containing posterior samples
        fft_data: MultivarFFT object (for frequency axis)
        outdir: Optional directory to save the plot
    """
    # Extract frequency axis
    freqs = fft_data.freq
    # Extract PSD samples: shape (n_samples, n_freq, n_channels, n_channels)
    psd_samples = idata.posterior['psd_matrix'].values  # adjust key if needed
    # Take mean over samples
    psd_mean = np.mean(psd_samples, axis=0)  # shape (n_freq, n_channels, n_channels)
    n_channels = psd_mean.shape[1]
    # Plot each channel pair
    fig, axes = plt.subplots(n_channels, n_channels, figsize=(4*n_channels, 4*n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j] if n_channels > 1 else axes
            ax.plot(freqs, psd_mean[:, i, j])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('PSD')
            ax.set_title(f'PSD: Ch {i+1} vs Ch {j+1}')
            ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    if outdir:
        plt.savefig(f'{outdir}/psd_matrix_loglog.png')
    else:
        plt.show()

# Call the plotting function after print(idata)
_plot_psd_matrix(idata, fft_data, outdir='results/lisa')
