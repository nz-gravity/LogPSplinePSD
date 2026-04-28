import os

import arviz as az
import numpy as np

from log_psplines.datatypes import Periodogram, Timeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.example_datasets.lvk_data import LVKData
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.psplines import LogPSplines


def _plot_univariate_periodogram(
    pdgrm: Periodogram,
    spline_model: LogPSplines,
):
    freq = np.asarray(pdgrm.freqs, dtype=np.float64)
    model = np.exp(np.asarray(spline_model(), dtype=np.float64))
    empirical = EmpiricalPSD(
        freq=freq,
        psd=np.asarray(pdgrm.power, dtype=np.complex128)[:, None, None],
        coherence=np.zeros((freq.size, 1, 1), dtype=np.float64),
        channels=np.asarray(["1"]),
    )
    return plot_psd_matrix(
        PSDMatrixPlotSpec(
            freq=freq,
            ci_dict={
                "psd": {(0, 0): (model, model, model)},
                "coh": {},
                "re": {},
                "im": {},
                "mag": {},
            },
            empirical_psd=empirical,
            save=False,
            close=False,
            show_knots=False,
        )
    )


FMIN, FMAX = 20, 1024
DURATION = 4.0

out = os.path.join("out_lvk_mcmc_nuts")
os.makedirs(out, exist_ok=True)
lvk_data = LVKData.download_data(
    detector="L1",
    gps_start=1126259462,
    duration=DURATION,
    fmin=FMIN,
    fmax=FMAX,
    threshold=10,
)
lvk_data.plot_psd(fname=os.path.join(out, "lvk_psd_analysis.png"))
# rescale the PSD to a better scale to work with
power = lvk_data.psd / np.nanmax(lvk_data.psd) * 1e-3
pdgrm = Periodogram(
    freqs=lvk_data.freqs,
    power=power,
)
pdgrm = pdgrm.cut(FMIN, FMAX)
n = lvk_data.strain.shape[0]
t = np.linspace(0.0, DURATION, n, endpoint=False)
ts = Timeseries(t=t, y=lvk_data.strain)

idata_fname = os.path.join(out, "inference_data.nc")
if os.path.exists(idata_fname):
    print(f"Loading existing inference data from {idata_fname}")
    idata = az.from_netcdf(idata_fname)
else:
    spline_model = LogPSplines.from_periodogram(
        pdgrm,
        n_knots=len(lvk_data.knots_locations),
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs=dict(knots=lvk_data.knots_locations),
    )
    # plot initial fit with optimised weights
    fig, axes = _plot_univariate_periodogram(pdgrm, spline_model)
    ax = axes[0, 0]
    ax.set_xscale("linear")
    fig.savefig(os.path.join(out, "test_spline_init.png"))

    idata = run_mcmc(
        ts,
        n_samples=2000,
        n_warmup=2000,
        outdir=out,
        rng_key=42,
        knot_kwargs=dict(knots=lvk_data.knots_locations),
        fmin=FMIN,
        fmax=FMAX,
    )

    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(idata=idata, save=False, close=False)
    )
    ax = axes[0, 0]
    ax.set_xscale("linear")
    fig.savefig(os.path.join(out, "test_mcmc.png"))

    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(idata=idata, save=False, close=False)
    )
    ax = axes[0, 0]
    ax.set_xscale("log")
    fig.savefig(os.path.join(out, "test_mcmc_log.png"))

    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(
            idata=idata, save=False, close=False, show_knots=False
        )
    )
    ax = axes[0, 0]
    ax.set_xscale("linear")
    fig.savefig(os.path.join(out, "test_mcmc_no_knots.png"))

    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(
            idata=idata, save=False, close=False, show_knots=False
        )
    )
    ax = axes[0, 0]
    ax.set_xscale("log")
    fig.savefig(os.path.join(out, "test_mcmc_log_no_knots.png"))


fig, axes = plot_psd_matrix(
    PSDMatrixPlotSpec(idata=idata, save=False, close=False, show_knots=True)
)
ax = axes[0, 0]
ax.set_xscale("log")
fig.savefig(os.path.join(out, "test_mcmc_log_no_knots.png"))
# plt.show()
