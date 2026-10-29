"""Saved preprocessing must describe the supplied statistic, not fit options."""

from dataclasses import replace

import numpy as np
import pytest

from log_psplines import PSDResult, StationaryConfig, fit
from log_psplines.preprocessing.coarse_grain import (
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.preprocessing.periodogram import compute_wishart


@pytest.mark.parametrize("known_history", [True, False])
def test_precomputed_statistics_preserve_actual_history(
    known_history, tmp_path
):
    data = compute_wishart(
        np.random.default_rng(41).normal(size=(256, 1)) + 7.0,
        fs=16.0,
        Nb=4,
        window="hann",
    )
    if not known_history:
        data = replace(data, preprocessing={})
    expected = dict(data.preprocessing)
    data = apply_coarse_grain_multivar_fft(
        data, compute_binning_structure(data.freq, Nh=2)
    )
    data = data.cut(float(data.freq[1]), float(data.freq[-2]))
    assert data.preprocessing == expected

    # These options do not describe the already computed Hann statistics.
    result = fit(
        data,
        StationaryConfig(
            wishart_window=("tukey", 0.1),
            method="vi",
            n_knots=4,
            degree=2,
            vi_steps=2,
            vi_posterior_draws=3,
            vi_early_stopping=False,
            verbose=False,
            vi_progress_bar=False,
        ),
    )
    mean_removal = "global_per_channel" if known_history else "unknown"
    window = "hann" if known_history else "unknown"
    assert result.metadata["mean_removal"] == mean_removal
    assert result.metadata["window"] == window
    path = tmp_path / "preprocessing.nc"
    result.to_netcdf(path)
    restored = PSDResult.from_netcdf(path)
    assert restored.metadata["mean_removal"] == mean_removal
    assert restored.metadata["window"] == window
