import os
from unittest.mock import patch

import numpy as np
import pytest

from log_psplines.datatypes import Periodogram, Timeseries
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    RunMCMCConfig,
    run_mcmc,
)
from log_psplines.psplines.knots_locator import init_knots

try:
    from gwpy.timeseries import TimeSeries

    from log_psplines.example_datasets.lvk_data import LVKData

    _LVK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    TimeSeries = None  # type: ignore[assignment]
    LVKData = None  # type: ignore[assignment]
    _LVK_IMPORT_ERROR = exc


def mock_gwpy_timeseries_from_simulation(
    detector, gps_start, gps_end, **kwargs
):
    sample_rate = 256
    n_samples = int((gps_end - gps_start) * sample_rate)
    rng = np.random.default_rng(12345)
    data = rng.normal(scale=1e-21, size=n_samples)
    return TimeSeries(data, sample_rate=sample_rate, t0=gps_start)


def test_lvk_knots_from_periodogram():
    if LVKData is None or TimeSeries is None:
        pytest.skip(
            f"LVK dataset dependencies unavailable: {_LVK_IMPORT_ERROR}"
        )

    duration = 1.0
    with patch(
        "gwpy.timeseries.TimeSeries.fetch_open_data",
        side_effect=mock_gwpy_timeseries_from_simulation,
    ):
        lvk_data = LVKData.download_data(
            detector="L1",
            gps_start=1126259462,
            duration=duration,
            fmin=64,
            fmax=128,
        )

    power = lvk_data.psd / np.nanmax(lvk_data.psd)
    pdgrm = Periodogram(freqs=lvk_data.freqs, power=power).cut(64, 128)
    knots = init_knots(
        n_knots=6,
        periodogram=pdgrm,
        method="lvk",
    )

    assert knots.size >= 3
    assert knots[0] == 0.0
    assert knots[-1] == 1.0
    assert np.all(np.diff(knots) >= 0.0)
    assert np.all((knots >= 0.0) & (knots <= 1.0))


@pytest.mark.slow
def test_lvk_mcmc(outdir, test_mode):
    if LVKData is None or TimeSeries is None:
        pytest.skip(
            f"LVK dataset dependencies unavailable: {_LVK_IMPORT_ERROR}"
        )

    duration = 1.0
    out = os.path.join(outdir, "out_lvk_mcmc")
    os.makedirs(out, exist_ok=True)
    with patch(
        "gwpy.timeseries.TimeSeries.fetch_open_data",
        side_effect=mock_gwpy_timeseries_from_simulation,
    ):
        lvk_data = LVKData.download_data(
            detector="L1",
            gps_start=1126259462,
            duration=duration,
            fmin=64,
            fmax=128,
        )
    # TODO: mock download TimeSeries.fetch_open_data(detector, gps_start, gps_end)
    lvk_data.plot_psd(fname=os.path.join(out, "lvk_psd_analysis.png"))
    # rescale the PSD to a better scale to work with
    power = lvk_data.psd / np.nanmax(lvk_data.psd) * 1e-3
    pdgrm = Periodogram(
        freqs=lvk_data.freqs,
        power=power,
    )
    pdgrm = pdgrm.cut(64, 128)
    n = lvk_data.strain.shape[0]
    t = np.linspace(0.0, duration, n, endpoint=False)
    ts = Timeseries(t=t, y=lvk_data.strain)

    base_knots = 12
    if test_mode == "fast":
        base_knots = 4
    lvk_knots = init_knots(
        n_knots=base_knots,
        periodogram=pdgrm,
        method="lvk",
        knots_plotfn=os.path.join(out, "lvk_psd_analysis.png"),
    )
    assert lvk_knots is not None

    n_samples = 30
    n_warmup = 30
    n_knots = 12
    if test_mode == "fast":
        n_samples = 4
        n_warmup = 4
        n_knots = 4

    model_cfg = ModelConfig(
        n_knots=n_knots,
        knot_kwargs={"method": "uniform"},
    )
    diagnostics_cfg = DiagnosticsConfig(outdir=out)
    run_cfg = RunMCMCConfig(
        n_samples=n_samples,
        n_warmup=n_warmup,
        rng_key=42,
        model=model_cfg,
        diagnostics=diagnostics_cfg,
    )
    run_mcmc(ts, config=run_cfg)
