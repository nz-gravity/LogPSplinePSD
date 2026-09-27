"""Native result plotting contracts."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix
from log_psplines.plotting.results import plot_posterior_spectrum
from log_psplines.results import PSDResult


def _result(
    channels: int = 2, *, vi: bool = False, observed: bool = False
) -> PSDResult:
    frequency = np.linspace(0.1, 0.9, 7)
    draws = np.empty((1, 4, frequency.size, channels, channels), dtype=complex)
    for draw in range(4):
        for index, freq in enumerate(frequency):
            matrix = np.eye(channels) * (2.0 + freq + 0.1 * draw)
            draws[0, draw, index] = matrix
            if channels > 1:
                draws[0, draw, index, 0, 1] = 0.2 + 0.05j
                draws[0, draw, index, 1, 0] = 0.2 - 0.05j
    spectrum = xr.DataArray(
        draws,
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={
            "frequency": frequency,
            "channel": np.arange(channels),
            "channel_aux": np.arange(channels),
        },
    )
    empirical = (
        xr.Dataset(
            {
                "periodogram": (
                    ("frequency", "channel", "channel_aux"),
                    draws[0, 0],
                )
            },
            coords={
                "frequency": frequency,
                "channel": np.arange(channels),
                "channel_aux": np.arange(channels),
            },
        )
        if observed
        else None
    )
    return PSDResult(
        posterior=xr.Dataset(),
        spectrum=spectrum,
        vi_spectrum=spectrum * 1.05 if vi else None,
        observed_data=empirical,
    )


@pytest.mark.parametrize("mode", ["coherence", "magnitude", "components"])
def test_multivariate_modes_and_overlays(mode, tmp_path) -> None:
    result = _result(vi=True, observed=True)
    filename = f"{mode}.png"
    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(
            result=result,
            show_coherence=mode == "coherence",
            show_csd_magnitude=mode == "magnitude",
            true_psd=result.spectrum.values[0, 0],
            overlay_vi=True,
            knot_frequencies=np.array([0.2, 0.4]),
            excluded_bands=((0.3, 0.4),),
            channel_labels=["x", "y"],
            outdir=tmp_path,
            filename=filename,
            close=False,
        )
    )
    assert axes.shape == (2, 2)
    assert (tmp_path / filename).exists()
    assert len(axes[0, 0].lines) >= 4  # posterior, empirical, truth, VI
    if mode == "components":
        # Upper imaginary panel follows the stored Hermitian orientation.
        np.testing.assert_allclose(axes[0, 1].lines[0].get_ydata(), 0.05)
        np.testing.assert_allclose(axes[1, 0].lines[0].get_ydata(), 0.2)
    else:
        assert not axes[0, 1].axison
    fig.canvas.draw()
    plt.close(fig)


def test_scalar_and_time_varying_surface(tmp_path) -> None:
    result = _result(channels=1)
    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(result=result, save=False, close=False)
    )
    assert axes.shape == (1, 1)
    fig.canvas.draw()
    plt.close(fig)
    tv = PSDResult(
        posterior=xr.Dataset(),
        spectrum=result.spectrum.expand_dims(time=[0.0, 1.0]).transpose(
            "chain", "draw", "time", "frequency", "channel", "channel_aux"
        ),
        metadata={"units": "power"},
    )
    plot_posterior_spectrum(tv, tmp_path)
    assert (tmp_path / "posterior_spectrum.png").exists()


def test_invalid_plot_inputs() -> None:
    result = _result()
    with pytest.raises(ValueError, match="either coherence"):
        plot_psd_matrix(
            PSDMatrixPlotSpec(
                result=result,
                show_coherence=True,
                show_csd_magnitude=True,
            )
        )
    with pytest.raises(ValueError, match="non-negative"):
        plot_psd_matrix(PSDMatrixPlotSpec(result=result, psd_scale=-1.0))
    with pytest.raises(ValueError, match="shape"):
        plot_psd_matrix(PSDMatrixPlotSpec(result=result, true_psd=np.ones(7)))
    with pytest.raises(ValueError, match="stationary"):
        tv = PSDResult(
            posterior=xr.Dataset(),
            spectrum=result.spectrum.expand_dims(time=[0.0]).transpose(
                "chain", "draw", "time", "frequency", "channel", "channel_aux"
            ),
        )
        plot_psd_matrix(PSDMatrixPlotSpec(result=tv))
