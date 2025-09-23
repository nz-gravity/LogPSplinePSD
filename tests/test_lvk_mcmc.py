import os
from unittest.mock import patch

import numpy as np
from gwpy.timeseries import TimeSeries

from log_psplines.datatypes import Periodogram
from log_psplines.example_datasets.lvk_data import LVKData
from log_psplines.mcmc import run_mcmc
from log_psplines.psplines.knots_locator import init_knots


def mock_gwpy_timeseries_from_simulation(detector, gps_start, gps_end, **kwargs):
    duration = gps_end - gps_start
    fs = 4096  # or use kwargs.get('sample_rate', 4096) if needed
    lvk_sim = LVKData.from_simulation(duration=duration, fs=fs)
    return TimeSeries(lvk_sim.strain, sample_rate=fs)


def test_lvk_mcmc(outdir, test_mode):
    out = os.path.join(outdir, "out_lvk_mcmc")
    os.makedirs(out, exist_ok=True)
    with patch('gwpy.timeseries.TimeSeries.fetch_open_data', side_effect=mock_gwpy_timeseries_from_simulation):
        lvk_data = LVKData.download_data(
            detector="L1", gps_start=1126259462, duration=2, fmin=256, fmax=512
        )
    # TODO: mock download TimeSeries.fetch_open_data(detector, gps_start, gps_end)
    lvk_data.plot_psd(fname=os.path.join(out, "lvk_psd_analysis.png"))
    # rescale the PSD to a better scale to work with
    power = lvk_data.psd / np.nanmax(lvk_data.psd) * 1e-3
    pdgrm = Periodogram(
        freqs=lvk_data.freqs,
        power=power,
    )
    pdgrm = pdgrm.cut(256, 512)

    lvk_knots = init_knots(
        n_knots=50,
        periodogram=pdgrm,
        method="lvk",
        knots_plotfn=os.path.join(out, "lvk_psd_analysis.png"),
    )
    assert lvk_knots is not None

    kwgs = dict(
        n_samples=200,
        n_warmup=200,
        n_knots=50,
        outdir = out,
        rng_key = 42,
        knot_kwargs = dict(
            method="uniform",
        ),
    )

    if test_mode == "fast":
        kwgs.update(
            n_samples=5,
            n_warmup=5,
            n_knots=4,
        )
    run_mcmc(pdgrm, **kwgs, sampler="nuts")
